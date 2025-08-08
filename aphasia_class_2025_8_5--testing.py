# -*- coding: utf-8 -*-
"""
Advanced Multi-Modal Aphasia Classification System
With Adaptive Learning Rate and Comprehensive Reporting
"""

import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
import numpy as np
import os
import random
import csv
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer, TrainerCallback,
    EarlyStoppingCallback, get_cosine_schedule_with_warmup,
    default_data_collator, set_seed
)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
import gc
from scipy import stats

# Environment setup for stability
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
json_file = '/workspace/SH001/aphasia_data_augmented.json'

# Set seeds for reproducibility
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(42)

# Configuration
@dataclass
class ModelConfig:
    # Model architecture
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    max_length: int = 512
    hidden_size: int = 768
    
    # Feature dimensions
    pos_vocab_size: int = 150
    pos_emb_dim: int = 64
    grammar_dim: int = 3
    grammar_hidden_dim: int = 64
    duration_hidden_dim: int = 128
    prosody_dim: int = 32
    
    # Multi-head attention
    num_attention_heads: int = 8
    attention_dropout: float = 0.3
    
    # Classification head
    classifier_hidden_dims: List[int] = None
    dropout_rate: float = 0.3
    activation_fn: str = "tanh"
    
    # Training
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    batch_size: int = 10
    num_epochs: int = 500
    gradient_accumulation_steps: int = 4
    
    # Adaptive Learning Rate Parameters
    adaptive_lr: bool = True
    lr_patience: int = 3  # Patience for learning rate adjustment
    lr_factor: float = 0.8  # Factor to multiply learning rate
    lr_increase_factor: float = 1.2  # Factor to increase learning rate
    min_lr: float = 1e-6
    max_lr: float = 1e-3
    oscillation_amplitude: float = 0.1  # For sinusoidal oscillation
    
    # Advanced techniques
    use_focal_loss: bool = True
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_label_smoothing: bool = True
    label_smoothing: float = 0.1
    
    def __post_init__(self):
        if self.classifier_hidden_dims is None:
            self.classifier_hidden_dims = [512, 256]

# Utility functions
def log_message(message):
    timestamp = datetime.datetime.now().isoformat()
    full_message = f"{timestamp}: {message}"
    log_file = "./training_log.txt"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")
    print(full_message, flush=True)

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def normalize_type(t):
    return t.strip().upper() if isinstance(t, str) else t

# Adaptive Learning Rate Scheduler
class AdaptiveLearningRateScheduler:
    """智能學習率調度器，結合多種策略"""
    def __init__(self, optimizer, config: ModelConfig, total_steps: int):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        
        # 歷史記錄
        self.loss_history = []
        self.f1_history = []
        self.accuracy_history = []
        self.lr_history = []
        
        # 狀態追蹤
        self.plateau_counter = 0
        self.best_f1 = 0.0
        self.best_loss = float('inf')
        self.step_count = 0
        
        # 初始學習率
        self.base_lr = config.learning_rate
        self.current_lr = self.base_lr
        
        log_message(f"Adaptive LR Scheduler initialized with base_lr={self.base_lr}")
    
    def calculate_slope(self, values, window=3):
        """計算近期數值的斜率"""
        if len(values) < window:
            return 0.0
        
        recent_values = values[-window:]
        x = np.arange(len(recent_values))
        slope, _, _, _, _ = stats.linregress(x, recent_values)
        return slope
    
    def exponential_adjustment(self, current_value, target_value, base_factor=1.1):
        """指數調整函數"""
        ratio = current_value / target_value if target_value != 0 else 1.0
        factor = math.exp(-ratio) * base_factor
        return factor
    
    def logarithmic_adjustment(self, current_value, threshold=0.1):
        """對數調整函數"""
        if current_value <= 0:
            return 1.0
        factor = math.log(1 + current_value / threshold)
        return max(0.5, min(2.0, factor))
    
    def sinusoidal_oscillation(self, step, amplitude=None):
        """正弦波動調整"""
        if amplitude is None:
            amplitude = self.config.oscillation_amplitude
        
        # 基於步數的正弦波動
        phase = 2 * math.pi * step / (self.total_steps / 4)  # 4個週期
        oscillation = 1 + amplitude * math.sin(phase)
        return oscillation
    
    def cosine_decay(self, step):
        """餘弦衰減"""
        progress = step / self.total_steps
        decay = 0.5 * (1 + math.cos(math.pi * progress))
        return decay
    
    def adaptive_lr_calculation(self, current_loss, current_f1, current_acc):
        """智能學習率計算"""
        # 記錄歷史
        self.loss_history.append(current_loss)
        self.f1_history.append(current_f1)
        self.accuracy_history.append(current_acc)
        
        # 計算斜率
        loss_slope = self.calculate_slope(self.loss_history)
        f1_slope = self.calculate_slope(self.f1_history)
        acc_slope = self.calculate_slope(self.accuracy_history)
        
        # 基礎學習率調整因子
        adjustment_factor = 1.0
        
        # 1. 基於Loss斜率的調整
        if abs(loss_slope) < 0.001:  # Loss plateau
            log_message(f"Loss plateau detected (slope: {loss_slope:.6f})")
            # 指數增加學習率
            exp_factor = self.exponential_adjustment(abs(loss_slope), 0.01, 1.15)
            adjustment_factor *= exp_factor
            
        elif current_loss > 2.0:  # Loss太高
            log_message(f"High loss detected: {current_loss:.4f}")
            # 對數調整
            log_factor = self.logarithmic_adjustment(current_loss, 1.0)
            adjustment_factor *= log_factor
        
        # 2. 基於F1分數的調整
        if current_f1 < 0.3:  # F1太低
            log_message(f"Low F1 detected: {current_f1:.4f}")
            # 指數增加學習率
            exp_factor = self.exponential_adjustment(0.3, current_f1, 1.2)
            adjustment_factor *= exp_factor
            
        elif abs(f1_slope) < 0.001:  # F1 plateau
            log_message(f"F1 plateau detected (slope: {f1_slope:.6f})")
            adjustment_factor *= 1.1
        
        # 3. 添加正弦波動性
        sin_factor = self.sinusoidal_oscillation(self.step_count)
        
        # 4. 添加餘弦衰減
        cos_factor = self.cosine_decay(self.step_count)
        
        # 綜合調整
        final_factor = adjustment_factor * sin_factor * (0.3 + 0.7 * cos_factor)
        
        # 計算新的學習率
        new_lr = self.current_lr * final_factor
        
        # 限制學習率範圍
        new_lr = max(self.config.min_lr, min(self.config.max_lr, new_lr))
        
        # 更新學習率
        if abs(new_lr - self.current_lr) > 1e-7:  # 只有變化足夠大才更新
            self.current_lr = new_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            log_message(f"Learning rate adjusted: {new_lr:.2e} (factor: {final_factor:.3f})")
            log_message(f"  - Loss slope: {loss_slope:.6f}, F1 slope: {f1_slope:.6f}")
            log_message(f"  - Sin factor: {sin_factor:.3f}, Cos factor: {cos_factor:.3f}")
        
        self.lr_history.append(self.current_lr)
        self.step_count += 1
        
        return self.current_lr

# Training History Tracker
class TrainingHistoryTracker:
    """訓練歷史記錄器"""
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'eval_loss': [],
            'train_accuracy': [],
            'eval_accuracy': [],
            'train_f1': [],
            'eval_f1': [],
            'learning_rate': [],
            'train_precision': [],
            'eval_precision': [],
            'train_recall': [],
            'eval_recall': []
        }
    
    def update(self, epoch, metrics):
        """更新歷史記錄"""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save_history(self, output_dir):
        """保存歷史記錄"""
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
        return df
    
    def plot_training_curves(self, output_dir):
        """繪製訓練曲線"""
        if not self.history['epoch']:
            return
        
        # 設置圖表樣式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = self.history['epoch']
        
        # 1. Loss曲線
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['eval_loss'], 'r-', label='Eval Loss', linewidth=2)
        axes[0, 0].set_title('Loss Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 準確率曲線
        axes[0, 1].plot(epochs, self.history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.history['eval_accuracy'], 'r-', label='Eval Accuracy', linewidth=2)
        axes[0, 1].set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. F1分數曲線
        axes[0, 2].plot(epochs, self.history['train_f1'], 'b-', label='Train F1', linewidth=2)
        axes[0, 2].plot(epochs, self.history['eval_f1'], 'r-', label='Eval F1', linewidth=2)
        axes[0, 2].set_title('F1 Score Over Time', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 學習率曲線
        axes[1, 0].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Precision曲線
        axes[1, 1].plot(epochs, self.history['train_precision'], 'b-', label='Train Precision', linewidth=2)
        axes[1, 1].plot(epochs, self.history['eval_precision'], 'r-', label='Eval Precision', linewidth=2)
        axes[1, 1].set_title('Precision Over Time', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Recall曲線
        axes[1, 2].plot(epochs, self.history['train_recall'], 'b-', label='Train Recall', linewidth=2)
        axes[1, 2].plot(epochs, self.history['eval_recall'], 'r-', label='Eval Recall', linewidth=2)
        axes[1, 2].set_title('Recall Over Time', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Recall')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()

# Focal loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Stable positional encoding
class StablePositionalEncoding(nn.Module):
    """Simplified but stable positional encoding"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Traditional sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Simple learnable component
        self.learnable_pe = nn.Parameter(torch.randn(max_len, d_model) * 0.01)
    
    def forward(self, x):
        seq_len = x.size(1)
        sinusoidal = self.pe[:, :seq_len, :].to(x.device)
        learnable = self.learnable_pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)
        return x + 0.1 * (sinusoidal + learnable)

# Stable multi-head attention
class StableMultiHeadAttention(nn.Module):
    """Stable multi-head attention for feature fusion"""
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        
        output = self.output_proj(context)
        return self.layer_norm(output + x)

# Stable linguistic feature extractor
class StableLinguisticFeatureExtractor(nn.Module):
    """Stable linguistic feature processing"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # POS embeddings
        self.pos_embedding = nn.Embedding(config.pos_vocab_size, config.pos_emb_dim, padding_idx=0)
        self.pos_attention = StableMultiHeadAttention(config.pos_emb_dim, num_heads=4)
        
        # Grammar feature processing
        self.grammar_projection = nn.Sequential(
            nn.Linear(config.grammar_dim, config.grammar_hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(config.grammar_hidden_dim),
            nn.Dropout(config.dropout_rate * 0.3)
        )
        
        # Duration processing
        self.duration_projection = nn.Sequential(
            nn.Linear(1, config.duration_hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(config.duration_hidden_dim)
        )
        
        # Prosody processing
        self.prosody_projection = nn.Sequential(
            nn.Linear(config.prosody_dim, config.prosody_dim),
            nn.ReLU(),
            nn.LayerNorm(config.prosody_dim)
        )
        
        # Feature fusion
        total_feature_dim = (config.pos_emb_dim + config.grammar_hidden_dim + 
                           config.duration_hidden_dim + config.prosody_dim)
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_feature_dim, total_feature_dim // 2),
            nn.Tanh(),
            nn.LayerNorm(total_feature_dim // 2),
            nn.Dropout(config.dropout_rate)
        )
        
    def forward(self, pos_ids, grammar_ids, durations, prosody_features, attention_mask):
        batch_size, seq_len = pos_ids.size()
        
        # Process POS features with clamping
        pos_ids_clamped = pos_ids.clamp(0, self.config.pos_vocab_size - 1)
        pos_embeds = self.pos_embedding(pos_ids_clamped)
        pos_features = self.pos_attention(pos_embeds, attention_mask)
        
        # Process grammar features
        grammar_features = self.grammar_projection(grammar_ids.float())
        
        # Process duration features
        duration_features = self.duration_projection(durations.unsqueeze(-1).float())
        
        # Process prosodic features
        prosody_features = self.prosody_projection(prosody_features.float())
        
        # Combine features
        combined_features = torch.cat([
            pos_features, grammar_features, duration_features, prosody_features
        ], dim=-1)
        
        # Feature fusion
        fused_features = self.feature_fusion(combined_features)
        
        # Global pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled_features = torch.sum(fused_features * mask_expanded, dim=1) / torch.sum(mask_expanded, dim=1)
        
        return pooled_features

# Main classifier with stability improvements
class StableAphasiaClassifier(nn.Module):
    """Stable aphasia classification model"""
    def __init__(self, config: ModelConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # Pre-trained model
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.bert_config = self.bert.config
        
        # Freeze embeddings for stability
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Positional encoding
        self.positional_encoder = StablePositionalEncoding(
            d_model=self.bert_config.hidden_size,
            max_len=config.max_length
        )
        
        # Linguistic feature extractor
        self.linguistic_extractor = StableLinguisticFeatureExtractor(config)
        
        # Calculate dimensions
        bert_dim = self.bert_config.hidden_size
        linguistic_dim = (config.pos_emb_dim + config.grammar_hidden_dim + 
                         config.duration_hidden_dim + config.prosody_dim) // 2
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(bert_dim + linguistic_dim, bert_dim),
            nn.LayerNorm(bert_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Classifier
        self.classifier = self._build_classifier(bert_dim, num_labels)
        
        # Multi-task heads (simplified)
        self.severity_head = nn.Sequential(
            nn.Linear(bert_dim, 4),
            nn.Softmax(dim=-1)
        )
        
        self.fluency_head = nn.Sequential(
            nn.Linear(bert_dim, 1),
            nn.Sigmoid()
        )
        
    def _build_classifier(self, input_dim: int, num_labels: int):
        layers = []
        current_dim = input_dim
        
        for hidden_dim in self.config.classifier_hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Dropout(self.config.dropout_rate)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, num_labels))
        return nn.Sequential(*layers)
    
    def forward(self, input_ids, attention_mask, labels=None,
                word_pos_ids=None, word_grammar_ids=None, word_durations=None,
                prosody_features=None, **kwargs):
        
        # BERT encoding
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        # Apply positional encoding
        position_enhanced = self.positional_encoder(sequence_output)
        
        # Attention pooling
        pooled_output = self._attention_pooling(position_enhanced, attention_mask)
        
        # Process linguistic features
        if all(x is not None for x in [word_pos_ids, word_grammar_ids, word_durations]):
            if prosody_features is None:
                batch_size, seq_len = input_ids.size()
                prosody_features = torch.zeros(
                    batch_size, seq_len, self.config.prosody_dim,
                    device=input_ids.device
                )
            
            linguistic_features = self.linguistic_extractor(
                word_pos_ids, word_grammar_ids, word_durations,
                prosody_features, attention_mask
            )
        else:
            linguistic_features = torch.zeros(
                input_ids.size(0), 
                (self.config.pos_emb_dim + self.config.grammar_hidden_dim + 
                 self.config.duration_hidden_dim + self.config.prosody_dim) // 2,
                device=input_ids.device
            )
        
        # Feature fusion
        combined_features = torch.cat([pooled_output, linguistic_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # Predictions
        logits = self.classifier(fused_features)
        severity_pred = self.severity_head(fused_features)
        fluency_pred = self.fluency_head(fused_features)
        
        # Loss computation
        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)
        
        return {
            "logits": logits,
            "severity_pred": severity_pred,
            "fluency_pred": fluency_pred,
            "loss": loss
        }
    
    def _attention_pooling(self, sequence_output, attention_mask):
        """Attention-based pooling"""
        attention_weights = torch.softmax(
            torch.sum(sequence_output, dim=-1, keepdim=True), dim=1
        )
        attention_weights = attention_weights * attention_mask.unsqueeze(-1).float()
        attention_weights = attention_weights / (torch.sum(attention_weights, dim=1, keepdim=True) + 1e-9)
        pooled = torch.sum(sequence_output * attention_weights, dim=1)
        return pooled
    
    def _compute_loss(self, logits, labels):
        if self.config.use_focal_loss:
            focal_loss = FocalLoss(
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
                reduction='mean'
            )
            return focal_loss(logits, labels)
        else:
            if self.config.use_label_smoothing:
                return F.cross_entropy(
                    logits, labels, 
                    label_smoothing=self.config.label_smoothing
                )
            else:
                return F.cross_entropy(logits, labels)

# Stable dataset class
class StableAphasiaDataset(Dataset):
    """Stable dataset with simplified processing"""
    def __init__(self, sentences, tokenizer, aphasia_types_mapping, config: ModelConfig):
        self.samples = []
        self.tokenizer = tokenizer
        self.config = config
        self.aphasia_types_mapping = aphasia_types_mapping
        
        # Add special tokens
        special_tokens = ["[DIALOGUE]", "[TURN]", "[PAUSE]", "[REPEAT]", "[HESITATION]"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        for idx, item in enumerate(sentences):
            sentence_id = item.get("sentence_id", f"S{idx}")
            aphasia_type = normalize_type(item.get("aphasia_type", ""))
            
            if aphasia_type not in aphasia_types_mapping:
                log_message(f"Skipping Sentence {sentence_id}: Invalid aphasia type '{aphasia_type}'")
                continue
            
            self._process_sentence(item, sentence_id, aphasia_type)
        
        if not self.samples:
            raise ValueError("No valid samples found in dataset!")
        
        log_message(f"Dataset created with {len(self.samples)} samples")
        self._print_class_distribution()
    
    def _process_sentence(self, item, sentence_id, aphasia_type):
        """Process sentence with stable approach"""
        all_tokens, all_pos, all_grammar, all_durations = [], [], [], []
        
        for dialogue_idx, dialogue in enumerate(item.get("dialogues", [])):
            if dialogue_idx > 0:
                all_tokens.append("[DIALOGUE]")
                all_pos.append(0)
                all_grammar.append([0, 0, 0])
                all_durations.append(0.0)
            
            for par in dialogue.get("PAR", []):
                if "tokens" in par and par["tokens"]:
                    tokens = par["tokens"]
                    pos_ids = par.get("word_pos_ids", [0] * len(tokens))
                    grammar_ids = par.get("word_grammar_ids", [[0, 0, 0]] * len(tokens))
                    durations = par.get("word_durations", [0.0] * len(tokens))
                    
                    all_tokens.extend(tokens)
                    all_pos.extend(pos_ids)
                    all_grammar.extend(grammar_ids)
                    all_durations.extend(durations)
        
        if not all_tokens:
            return
        
        # Create sample
        self._create_sample(all_tokens, all_pos, all_grammar, all_durations, 
                          sentence_id, aphasia_type)
    
    def _create_sample(self, tokens, pos_ids, grammar_ids, durations, 
                      sentence_id, aphasia_type):
        """Create training sample"""
        # Tokenize
        text = " ".join(tokens)
        encoded = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Align features
        aligned_pos, aligned_grammar, aligned_durations = self._align_features(
            tokens, pos_ids, grammar_ids, durations, encoded
        )
        
        # Create prosody features
        prosody_features = self._extract_prosodic_features(durations, tokens)
        prosody_tensor = torch.tensor(prosody_features).unsqueeze(0).repeat(
            self.config.max_length, 1
        )
        
        label = self.aphasia_types_mapping[aphasia_type]
        
        sample = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "word_pos_ids": torch.tensor(aligned_pos, dtype=torch.long),
            "word_grammar_ids": torch.tensor(aligned_grammar, dtype=torch.long),
            "word_durations": torch.tensor(aligned_durations, dtype=torch.float),
            "prosody_features": prosody_tensor.float(),
            "sentence_id": sentence_id
        }
        self.samples.append(sample)
    
    def _align_features(self, tokens, pos_ids, grammar_ids, durations, encoded):
        """Align features with BERT subtokens"""
        subtoken_to_token = []
        
        for token_idx, token in enumerate(tokens):
            subtokens = self.tokenizer.tokenize(token)
            subtoken_to_token.extend([token_idx] * len(subtokens))
        
        aligned_pos = [0]  # [CLS]
        aligned_grammar = [[0, 0, 0]]  # [CLS]
        aligned_durations = [0.0]  # [CLS]
        
        for subtoken_idx in range(1, self.config.max_length - 1):
            if subtoken_idx - 1 < len(subtoken_to_token):
                original_idx = subtoken_to_token[subtoken_idx - 1]
                aligned_pos.append(pos_ids[original_idx] if original_idx < len(pos_ids) else 0)
                aligned_grammar.append(grammar_ids[original_idx] if original_idx < len(grammar_ids) else [0, 0, 0])
                raw = durations[original_idx] if original_idx < len(durations) else 0.0
                if isinstance(raw, list) and (isinstance(raw[1], int) and isinstance(raw[0], int)):
                    if len(raw) >= 2:
                        duration_val = int(raw[1]) - int(raw[0])
                    else:
                        duration_val = raw[0]
                else:
                    duration_val = 0.0
                aligned_durations.append(duration_val)
            else:
                aligned_pos.append(0)
                aligned_grammar.append([0, 0, 0])
                aligned_durations.append(0.0)
        
        aligned_pos.append(0)  # [SEP]
        aligned_grammar.append([0, 0, 0])  # [SEP]
        aligned_durations.append(0.0)  # [SEP]
        
        return aligned_pos, aligned_grammar, aligned_durations
    
    def _extract_prosodic_features(self, durations, tokens):
        """Extract prosodic features"""
        if not durations:
            return [0.0] * self.config.prosody_dim
        
        valid_durations = [d for d in durations if isinstance(d, (int, float)) and d > 0]
        if not valid_durations:
            return [0.0] * self.config.prosody_dim
        
        features = [
            np.mean(valid_durations),
            np.std(valid_durations),
            np.median(valid_durations),
            len([d for d in valid_durations if d > np.mean(valid_durations) * 1.5])
        ]
        
        # Pad to prosody_dim
        while len(features) < self.config.prosody_dim:
            features.append(0.0)
        
        return features[:self.config.prosody_dim]
    
    def _print_class_distribution(self):
        """Print class distribution"""
        label_counts = Counter(sample["labels"].item() for sample in self.samples)
        reverse_mapping = {v: k for k, v in self.aphasia_types_mapping.items()}
        
        log_message("\nClass Distribution:")
        for label_id, count in sorted(label_counts.items()):
            class_name = reverse_mapping.get(label_id, f"Unknown_{label_id}")
            log_message(f"  {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# Stable data collator
def stable_collate_fn(batch):
    """Stable data collation"""
    if not batch or batch[0] is None:
        return None
    
    try:
        max_length = batch[0]["input_ids"].size(0)
        
        collated_batch = {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "sentence_ids": [item.get("sentence_id", "N/A") for item in batch],
            "word_pos_ids": torch.stack([item.get("word_pos_ids", torch.zeros(max_length, dtype=torch.long)) for item in batch]),
            "word_grammar_ids": torch.stack([item.get("word_grammar_ids", torch.zeros(max_length, 3, dtype=torch.long)) for item in batch]),
            "word_durations": torch.stack([item.get("word_durations", torch.zeros(max_length, dtype=torch.float)) for item in batch]),
            "prosody_features": torch.stack([item.get("prosody_features", torch.zeros(max_length, 32, dtype=torch.float)) for item in batch])
        }
        return collated_batch
    except Exception as e:
        log_message(f"Collation error: {e}")
        return None

# Enhanced Training callback with adaptive learning rate
class AdaptiveTrainingCallback(TrainerCallback):
    """Enhanced training callback with adaptive learning rate and comprehensive tracking"""
    def __init__(self, config: ModelConfig, patience=5, min_delta=0.8):
        self.config = config
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float('-inf')
        self.patience_counter = 0
        
        # Learning rate scheduler
        self.lr_scheduler = None
        
        # History tracker
        self.history_tracker = TrainingHistoryTracker()
        
        # Metrics for current epoch
        self.current_train_metrics = {}
        self.current_eval_metrics = {}

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize learning rate scheduler"""
        if self.config.adaptive_lr:
            model = kwargs.get('model')
            optimizer = kwargs.get('optimizer')
            if optimizer and model:
                total_steps = state.max_steps if state.max_steps > 0 else len(kwargs.get('train_dataloader', [])) * args.num_train_epochs
                self.lr_scheduler = AdaptiveLearningRateScheduler(optimizer, self.config, total_steps)
                log_message("Adaptive learning rate scheduler initialized")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture training metrics"""
        if logs:
            # Store training metrics
            if 'train_loss' in logs:
                self.current_train_metrics['loss'] = logs['train_loss']
            if 'learning_rate' in logs:
                self.current_train_metrics['lr'] = logs['learning_rate']

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Handle evaluation and learning rate adjustment"""
        if logs is not None:
            current_metric = logs.get('eval_f1', 0)
            current_loss = logs.get('eval_loss', float('inf'))
            current_acc = logs.get('eval_accuracy', 0)
            
            # Store evaluation metrics
            self.current_eval_metrics = {
                'loss': current_loss,
                'f1': current_metric,
                'accuracy': current_acc,
                'precision': logs.get('eval_precision_macro', 0),
                'recall': logs.get('eval_recall_macro', 0)
            }
            
            # Update history
            epoch_metrics = {
                'train_loss': self.current_train_metrics.get('loss', 0),
                'eval_loss': current_loss,
                'train_accuracy': 0,  # Will be computed separately if needed
                'eval_accuracy': current_acc,
                'train_f1': 0,  # Will be computed separately if needed
                'eval_f1': current_metric,
                'learning_rate': self.current_train_metrics.get('lr', self.config.learning_rate),
                'train_precision': 0,
                'eval_precision': logs.get('eval_precision_macro', 0),
                'train_recall': 0,
                'eval_recall': logs.get('eval_recall_macro', 0)
            }
            
            self.history_tracker.update(state.epoch, epoch_metrics)
            
            # Adaptive learning rate adjustment
            if self.lr_scheduler and self.config.adaptive_lr:
                new_lr = self.lr_scheduler.adaptive_lr_calculation(current_loss, current_metric, current_acc)
            if current_acc > 0.84:
                log_message(f"Target accuracy reached ({current_acc:.2%}) → stopping and saving model")
                control.should_save = True
                control.should_training_stop = True
                return control    
            # Early stopping logic
            if current_metric > self.best_metric + self.min_delta:
                self.best_metric = current_metric
                self.patience_counter = 0
                log_message(f"New best F1 score: {current_metric:.4f}")
            else:
                self.patience_counter += 1
                log_message(f"No improvement for {self.patience_counter} evaluations")

                if self.patience_counter >= self.patience:
                    log_message("Early stopping triggered")
                    control.should_training_stop = True

        clear_memory()

    def on_train_end(self, args, state, control, **kwargs):
        """Save training history at the end"""
        output_dir = args.output_dir
        self.history_tracker.save_history(output_dir)
        self.history_tracker.plot_training_curves(output_dir)
        log_message("Training history and curves saved")

# Metrics computation
def compute_comprehensive_metrics(pred):
    """Compute comprehensive evaluation metrics"""
    predictions = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    labels = pred.label_ids
    
    preds = np.argmax(predictions, axis=1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    
    # Per-class metrics
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)

    return {
        "accuracy": acc,
        "f1": f1_weighted,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_std": np.std(f1_per_class),
        "precision_std": np.std(precision_per_class),
        "recall_std": np.std(recall_per_class)
    }

# Enhanced analysis and visualization
def generate_comprehensive_reports(trainer, eval_dataset, aphasia_types_mapping, tokenizer, output_dir):
    """Generate comprehensive analysis reports and visualizations"""
    log_message("Generating comprehensive reports...")
    
    model = trainer.model
    if hasattr(model, 'module'):
        model = model.module
    
    model.eval()
    device = next(model.parameters()).device
    
    predictions = []
    true_labels = []
    sentence_ids = []
    severity_preds = []
    fluency_preds = []
    prediction_probs = []
    
    # Evaluation
    dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=stable_collate_fn)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            # Move to device
            for key in ['input_ids', 'attention_mask', 'word_pos_ids', 
                       'word_grammar_ids', 'word_durations', 'labels', 'prosody_features']:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            outputs = model(**batch)
            
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(batch["labels"].cpu().numpy())
            sentence_ids.extend(batch["sentence_ids"])
            severity_preds.extend(outputs["severity_pred"].cpu().numpy())
            fluency_preds.extend(outputs["fluency_pred"].cpu().numpy())
            prediction_probs.extend(probs.cpu().numpy())
    
    # Analysis
    reverse_mapping = {v: k for k, v in aphasia_types_mapping.items()}
    
    # 1. 詳細預測結果
    log_message("=== DETAILED PREDICTIONS (First 20) ===")
    for i in range(min(20, len(predictions))):
        true_type = reverse_mapping.get(true_labels[i], 'Unknown')
        pred_type = reverse_mapping.get(predictions[i], 'Unknown')
        severity_level = np.argmax(severity_preds[i])
        fluency_score = fluency_preds[i][0] if isinstance(fluency_preds[i], np.ndarray) else fluency_preds[i]
        confidence = np.max(prediction_probs[i])
        
        log_message(f"ID: {sentence_ids[i]} | True: {true_type} | Pred: {pred_type} | "
                   f"Confidence: {confidence:.3f} | Severity: {severity_level} | Fluency: {fluency_score:.3f}")
    
    # 2. 混淆矩陣
    cm = confusion_matrix(true_labels, predictions)
    
    # Enhanced confusion matrix plot
    plt.figure(figsize=(14, 12))
    
    # Calculate percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotation array
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap="Blues",
                xticklabels=list(aphasia_types_mapping.keys()),
                yticklabels=list(aphasia_types_mapping.keys()),
                cbar_kws={'label': 'Count'})
    
    plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
    plt.ylabel("True Label", fontsize=12, fontweight='bold')
    plt.title("Enhanced Confusion Matrix\n(Count and Percentage)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "enhanced_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 分類報告
    all_label_ids = list(aphasia_types_mapping.values())
    report_dict = classification_report(
        true_labels,
        predictions,
        labels=all_label_ids,
        target_names=list(aphasia_types_mapping.keys()),
        output_dict=True,
        zero_division=0
    )
    
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(os.path.join(output_dir, "comprehensive_classification_report.csv"))
    
    # 4. Per-class performance visualization
    class_names = list(aphasia_types_mapping.keys())
    metrics_data = []
    
    for i, class_name in enumerate(class_names):
        if class_name in report_dict:
            metrics_data.append({
                'Class': class_name,
                'Precision': report_dict[class_name]['precision'],
                'Recall': report_dict[class_name]['recall'],
                'F1-Score': report_dict[class_name]['f1-score'],
                'Support': report_dict[class_name]['support']
            })
    
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)
    
    # Plot per-class performance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Precision
    axes[0, 0].bar(df_metrics['Class'], df_metrics['Precision'], color='skyblue', alpha=0.8)
    axes[0, 0].set_title('Precision by Class', fontweight='bold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[0, 1].bar(df_metrics['Class'], df_metrics['Recall'], color='lightcoral', alpha=0.8)
    axes[0, 1].set_title('Recall by Class', fontweight='bold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1-Score
    axes[1, 0].bar(df_metrics['Class'], df_metrics['F1-Score'], color='lightgreen', alpha=0.8)
    axes[1, 0].set_title('F1-Score by Class', fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Support
    axes[1, 1].bar(df_metrics['Class'], df_metrics['Support'], color='gold', alpha=0.8)
    axes[1, 1].set_title('Support by Class', fontweight='bold')
    axes[1, 1].set_ylabel('Support (Number of Samples)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Prediction confidence distribution
    confidences = [np.max(prob) for prob in prediction_probs]
    correct_predictions = [pred == true for pred, true in zip(predictions, true_labels)]
    
    plt.figure(figsize=(12, 8))
    
    # Separate correct and incorrect predictions
    correct_confidences = [conf for conf, correct in zip(confidences, correct_predictions) if correct]
    incorrect_confidences = [conf for conf, correct in zip(confidences, correct_predictions) if not correct]
    
    plt.hist(correct_confidences, bins=30, alpha=0.7, label='Correct Predictions', color='green', density=True)
    plt.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect Predictions', color='red', density=True)
    
    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Prediction Confidence', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 特徵分析
    log_message("=== FEATURE ANALYSIS ===")
    avg_severity = np.mean(severity_preds, axis=0)
    avg_fluency = np.mean(fluency_preds)
    std_fluency = np.std(fluency_preds)
    
    log_message(f"Average Severity Distribution: {avg_severity}")
    log_message(f"Average Fluency Score: {avg_fluency:.3f} ± {std_fluency:.3f}")
    
    # 7. 詳細結果保存
    results_df = pd.DataFrame({
        'sentence_id': sentence_ids,
        'true_label': [reverse_mapping[label] for label in true_labels],
        'predicted_label': [reverse_mapping[pred] for pred in predictions],
        'prediction_confidence': confidences,
        'correct_prediction': correct_predictions,
        'severity_level': [np.argmax(severity) for severity in severity_preds],
        'fluency_score': [fluency[0] if isinstance(fluency, np.ndarray) else fluency for fluency in fluency_preds]
    })
    
    # Add probability columns for each class
    for i, class_name in enumerate(aphasia_types_mapping.keys()):
        results_df[f'prob_{class_name}'] = [prob[i] for prob in prediction_probs]
    
    results_df.to_csv(os.path.join(output_dir, "comprehensive_results.csv"), index=False)
    
    # 8. 統計摘要
    summary_stats = {
        'Overall Accuracy': accuracy_score(true_labels, predictions),
        'Macro F1': f1_score(true_labels, predictions, average='macro'),
        'Weighted F1': f1_score(true_labels, predictions, average='weighted'),
        'Macro Precision': precision_score(true_labels, predictions, average='macro'),
        'Macro Recall': recall_score(true_labels, predictions, average='macro'),
        'Average Confidence': np.mean(confidences),
        'Confidence Std': np.std(confidences),
        'Average Severity': avg_severity.tolist(),
        'Average Fluency': avg_fluency,
        'Fluency Std': std_fluency
    }
    
    serializable_summary = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in summary_stats.items()
    }
    with open(os.path.join(output_dir, "summary_statistics.json"), "w") as f:
        json.dump(serializable_summary, f, indent=2)
    
    log_message("Comprehensive Classification Report:")
    log_message(df_report.to_string())
    log_message(f"Comprehensive results saved to {output_dir}")
    
    return results_df, df_report, summary_stats

# Main training function with adaptive learning rate
def train_adaptive_model(json_file: str, output_dir: str = "./adaptive_aphasia_model"):
    """Main training function with adaptive learning rate"""
    
    log_message("Starting Adaptive Aphasia Classification Training")
    log_message("=" * 60)
    
    # Setup
    config = ModelConfig()
    os.makedirs(output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")
    
    # Load data
    log_message("Loading dataset...")
    with open(json_file, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    
    sentences = dataset_json.get("sentences", [])
    
    # Normalize aphasia types
    for item in sentences:
        if "aphasia_type" in item:
            item["aphasia_type"] = normalize_type(item["aphasia_type"])
    
    # Aphasia types mapping
    aphasia_types_mapping = {
        "BROCA": 0,
        "TRANSMOTOR": 1,
        "NOTAPHASICBYWAB": 2,
        "CONDUCTION": 3,
        "WERNICKE": 4,
        "ANOMIC": 5,
        "GLOBAL": 6,
        "ISOLATION": 7,
        "TRANSSENSORY": 8
    }
    
    log_message(f"Aphasia Types Mapping: {aphasia_types_mapping}")
    
    num_labels = len(aphasia_types_mapping)
    log_message(f"Number of labels: {num_labels}")
    
    # Filter sentences
    filtered_sentences = []
    for item in sentences:
        aphasia_type = item.get("aphasia_type", "")
        if aphasia_type in aphasia_types_mapping:
            filtered_sentences.append(item)
        else:
            log_message(f"Excluding sentence with invalid type: {aphasia_type}")
    
    log_message(f"Filtered dataset: {len(filtered_sentences)} sentences")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    random.shuffle(filtered_sentences)
    dataset_all = StableAphasiaDataset(
        filtered_sentences, tokenizer, aphasia_types_mapping, config
    )
    
    # Split dataset
    total_samples = len(dataset_all)
    train_size = int(0.8 * total_samples)
    eval_size = total_samples - train_size
    
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset_all, [train_size, eval_size]
    )
    
    log_message(f"Train size: {train_size}, Eval size: {eval_size}")
    
    # Setup weighted sampling for class imbalance
    train_labels = [dataset_all.samples[idx]["labels"].item() for idx in train_dataset.indices]
    label_counts = Counter(train_labels)
    sample_weights = [1.0 / label_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    # Model initialization
    def model_init():
        model = StableAphasiaClassifier(config, num_labels)
        model.bert.resize_token_embeddings(len(tokenizer))
        return model.to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_strategy="steps",
        logging_steps=50,
        seed=42,
        dataloader_num_workers=0,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=1.0,
        fp16=False,
        dataloader_drop_last=True,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=3,
        remove_unused_columns=False,
    )
    
    # Initialize trainer with adaptive callback
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_comprehensive_metrics,
        data_collator=stable_collate_fn,
        callbacks=[AdaptiveTrainingCallback(config, patience=5, min_delta=0.8)]
    )
    
    # Start training
    log_message("Starting adaptive training...")
    try:
        trainer.train()
        log_message("Training completed successfully!")
    except Exception as e:
        log_message(f"Training error: {str(e)}")
        import traceback
        log_message(traceback.format_exc())
        raise
    
    # Final evaluation
    log_message("Starting final evaluation...")
    eval_results = trainer.evaluate()
    log_message(f"Final evaluation results: {eval_results}")
    
    # Generate comprehensive reports
    results_df, report_df, summary_stats = generate_comprehensive_reports(
        trainer, eval_dataset, aphasia_types_mapping, tokenizer, output_dir
    )
    
    # Save model
    model_to_save = trainer.model
    if hasattr(model_to_save, 'module'):
        model_to_save = model_to_save.module
    
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration
    config_dict = {
        "model_name": config.model_name,
        "num_labels": num_labels,
        "aphasia_types_mapping": aphasia_types_mapping,
        "training_args": training_args.to_dict(),
        "adaptive_lr_config": {
            "adaptive_lr": config.adaptive_lr,
            "lr_patience": config.lr_patience,
            "lr_factor": config.lr_factor,
            "lr_increase_factor": config.lr_increase_factor,
            "min_lr": config.min_lr,
            "max_lr": config.max_lr,
            "oscillation_amplitude": config.oscillation_amplitude
        }
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    log_message(f"Adaptive model and comprehensive reports saved to {output_dir}")
    clear_memory()
    
    return trainer, eval_results, results_df

# Cross-validation with adaptive learning rate
def train_adaptive_cross_validation(json_file: str, output_dir: str = "./adaptive_cv_results", n_folds: int = 5):
    """Cross-validation training with adaptive learning rate"""
    log_message("Starting Adaptive Cross-Validation Training")
    
    config = ModelConfig()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    with open(json_file, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    
    sentences = dataset_json.get("sentences", [])
    
    # Normalize and filter
    for item in sentences:
        if "aphasia_type" in item:
            item["aphasia_type"] = normalize_type(item["aphasia_type"])
    
    aphasia_types_mapping = {
        "BROCA": 0, "TRANSMOTOR": 1, "NOTAPHASICBYWAB": 2,
        "CONDUCTION": 3, "WERNICKE": 4, "ANOMIC": 5,
        "GLOBAL": 6, "ISOLATION": 7, "TRANSSENSORY": 8
    }
    
    filtered_sentences = [s for s in sentences if s.get("aphasia_type") in aphasia_types_mapping]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create full dataset
    full_dataset = StableAphasiaDataset(
        filtered_sentences, tokenizer, aphasia_types_mapping, config
    )
    
    # Extract labels for stratification
    sample_labels = [sample["labels"].item() for sample in full_dataset.samples]
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    all_predictions = []
    all_true_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(sample_labels)), sample_labels)):
        log_message(f"\n=== Fold {fold + 1}/{n_folds} ===")
        
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        # Train single fold
        fold_trainer, fold_results_dict, fold_predictions = train_adaptive_single_fold(
            train_subset, val_subset, config, aphasia_types_mapping, 
            tokenizer, fold, output_dir
        )
        
        fold_results.append({
            'fold': fold + 1,
            **fold_results_dict
        })
        
        # Collect predictions for ensemble analysis
        all_predictions.extend(fold_predictions['predictions'])
        all_true_labels.extend(fold_predictions['true_labels'])
        
        clear_memory()
    
    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(output_dir, "adaptive_cv_summary.csv"), index=False)
    
    # Cross-validation summary statistics
    cv_summary = {
        'mean_accuracy': results_df['accuracy'].mean(),
        'std_accuracy': results_df['accuracy'].std(),
        'mean_f1': results_df['f1'].mean(),
        'std_f1': results_df['f1'].std(),
        'mean_f1_macro': results_df['f1_macro'].mean(),
        'std_f1_macro': results_df['f1_macro'].std(),
        'mean_precision': results_df['precision_macro'].mean(),
        'std_precision': results_df['precision_macro'].std(),
        'mean_recall': results_df['recall_macro'].mean(),
        'std_recall': results_df['recall_macro'].std()
    }
    
    with open(os.path.join(output_dir, "cv_statistics.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)
    
    # Overall confusion matrix across all folds
    overall_cm = confusion_matrix(all_true_labels, all_predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(overall_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(aphasia_types_mapping.keys()),
                yticklabels=list(aphasia_types_mapping.keys()))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label") 
    plt.title("Overall Confusion Matrix (All Folds)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cross-validation results visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy across folds
    axes[0, 0].bar(range(1, n_folds + 1), results_df['accuracy'], color='skyblue', alpha=0.8)
    axes[0, 0].axhline(y=results_df['accuracy'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {results_df["accuracy"].mean():.3f}')
    axes[0, 0].set_title('Accuracy Across Folds')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score across folds
    axes[0, 1].bar(range(1, n_folds + 1), results_df['f1'], color='lightgreen', alpha=0.8)
    axes[0, 1].axhline(y=results_df['f1'].mean(), color='red', linestyle='--',
                       label=f'Mean: {results_df["f1"].mean():.3f}')
    axes[0, 1].set_title('F1 Score Across Folds')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision across folds
    axes[1, 0].bar(range(1, n_folds + 1), results_df['precision_macro'], color='coral', alpha=0.8)
    axes[1, 0].axhline(y=results_df['precision_macro'].mean(), color='red', linestyle='--',
                       label=f'Mean: {results_df["precision_macro"].mean():.3f}')
    axes[1, 0].set_title('Precision Across Folds')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall across folds
    axes[1, 1].bar(range(1, n_folds + 1), results_df['recall_macro'], color='gold', alpha=0.8)
    axes[1, 1].axhline(y=results_df['recall_macro'].mean(), color='red', linestyle='--',
                       label=f'Mean: {results_df["recall_macro"].mean():.3f}')
    axes[1, 1].set_title('Recall Across Folds')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_performance_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    log_message("\n=== Adaptive Cross-Validation Summary ===")
    log_message(results_df.to_string(index=False))
    
    # Statistics
    log_message(f"\nMean F1: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    log_message(f"Mean Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    log_message(f"Mean F1 Macro: {results_df['f1_macro'].mean():.4f} ± {results_df['f1_macro'].std():.4f}")
    
    return results_df, cv_summary

def train_adaptive_single_fold(train_dataset, val_dataset, config, aphasia_types_mapping, 
                              tokenizer, fold, output_dir):
    """Train a single fold with adaptive learning rate"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = len(aphasia_types_mapping)
    
    # Setup weighted sampling
    train_labels = [train_dataset[i]["labels"].item() for i in range(len(train_dataset))]
    label_counts = Counter(train_labels)
    sample_weights = [1.0 / label_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Model initialization
    def model_init():
        model = StableAphasiaClassifier(config, num_labels)
        model.bert.resize_token_embeddings(len(tokenizer))
        return model.to(device)
    
    # Training arguments
    fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=50,
        seed=42,
        dataloader_num_workers=0,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=1.0,
        fp16=False,
        dataloader_drop_last=True,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=1,
        remove_unused_columns=False,
    )
    
    # Trainer with adaptive callback
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_comprehensive_metrics,
        data_collator=stable_collate_fn,
        callbacks=[AdaptiveTrainingCallback(config, patience=5, min_delta=0.8)]
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    # Get predictions for ensemble analysis
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    fold_predictions = {
        'predictions': pred_labels.tolist(),
        'true_labels': true_labels.tolist()
    }
    
    # Save fold model
    model_to_save = trainer.model
    if hasattr(model_to_save, 'module'):
        model_to_save = model_to_save.module
    
    torch.save(model_to_save.state_dict(), os.path.join(fold_output_dir, "pytorch_model.bin"))
    
    return trainer, eval_results, fold_predictions

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Learning Rate Aphasia Classification Training")
    parser.add_argument("--output_dir", type=str, default="./adaptive_aphasia_model", help="Output directory")
    parser.add_argument("--cross_validation", action="store_true", help="Use cross-validation")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--json_file", type=str, default=json_file, help="Path to JSON dataset file")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--adaptive_lr", action="store_true", default=True, help="Use adaptive learning rate")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = ModelConfig()
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.adaptive_lr = args.adaptive_lr
    
    try:
        clear_memory()
        
        log_message(f"Starting training with adaptive_lr={config.adaptive_lr}")
        log_message(f"Config: lr={config.learning_rate}, batch_size={config.batch_size}, epochs={config.num_epochs}")
        
        if args.cross_validation:
            results_df, cv_summary = train_adaptive_cross_validation(args.json_file, args.output_dir, args.n_folds)
            log_message("Cross-validation training completed!")
        else:
            trainer, eval_results, results_df = train_adaptive_model(args.json_file, args.output_dir)
            log_message("Single model training completed!")
            
        log_message("All adaptive training completed successfully!")
        
    except Exception as e:
        log_message(f"Training failed: {str(e)}")
        import traceback
        log_message(traceback.format_exc())
    finally:
        clear_memory()