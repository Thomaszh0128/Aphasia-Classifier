# -*- coding: utf-8 -*-
"""
失語症分類推理系統
用於載入訓練好的模型並對新的語音數據進行分類預測
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

# 重新定義模型結構（與訓練程式碼一致）
@dataclass
class ModelConfig:
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    max_length: int = 512
    hidden_size: int = 768
    pos_vocab_size: int = 150
    pos_emb_dim: int = 64
    grammar_dim: int = 3
    grammar_hidden_dim: int = 64
    duration_hidden_dim: int = 128
    prosody_dim: int = 32
    num_attention_heads: int = 8
    attention_dropout: float = 0.3
    classifier_hidden_dims: List[int] = None
    dropout_rate: float = 0.3
    
    def __post_init__(self):
        if self.classifier_hidden_dims is None:
            self.classifier_hidden_dims = [512, 256]

class StablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        self.learnable_pe = nn.Parameter(torch.randn(max_len, d_model) * 0.01)
    
    def forward(self, x):
        seq_len = x.size(1)
        sinusoidal = self.pe[:, :seq_len, :].to(x.device)
        learnable = self.learnable_pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)
        return x + 0.1 * (sinusoidal + learnable)

class StableMultiHeadAttention(nn.Module):
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

class StableLinguisticFeatureExtractor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.pos_embedding = nn.Embedding(config.pos_vocab_size, config.pos_emb_dim, padding_idx=0)
        self.pos_attention = StableMultiHeadAttention(config.pos_emb_dim, num_heads=4)
        
        self.grammar_projection = nn.Sequential(
            nn.Linear(config.grammar_dim, config.grammar_hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(config.grammar_hidden_dim),
            nn.Dropout(config.dropout_rate * 0.3)
        )
        
        self.duration_projection = nn.Sequential(
            nn.Linear(1, config.duration_hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(config.duration_hidden_dim)
        )
        
        self.prosody_projection = nn.Sequential(
            nn.Linear(config.prosody_dim, config.prosody_dim),
            nn.ReLU(),
            nn.LayerNorm(config.prosody_dim)
        )
        
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
        
        pos_ids_clamped = pos_ids.clamp(0, self.config.pos_vocab_size - 1)
        pos_embeds = self.pos_embedding(pos_ids_clamped)
        pos_features = self.pos_attention(pos_embeds, attention_mask)
        
        grammar_features = self.grammar_projection(grammar_ids.float())
        duration_features = self.duration_projection(durations.unsqueeze(-1).float())
        prosody_features = self.prosody_projection(prosody_features.float())
        
        combined_features = torch.cat([
            pos_features, grammar_features, duration_features, prosody_features
        ], dim=-1)
        
        fused_features = self.feature_fusion(combined_features)
        
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled_features = torch.sum(fused_features * mask_expanded, dim=1) / torch.sum(mask_expanded, dim=1)
        
        return pooled_features

class StableAphasiaClassifier(nn.Module):
    def __init__(self, config: ModelConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.bert_config = self.bert.config
        
        self.positional_encoder = StablePositionalEncoding(
            d_model=self.bert_config.hidden_size,
            max_len=config.max_length
        )
        
        self.linguistic_extractor = StableLinguisticFeatureExtractor(config)
        
        bert_dim = self.bert_config.hidden_size
        linguistic_dim = (config.pos_emb_dim + config.grammar_hidden_dim + 
                         config.duration_hidden_dim + config.prosody_dim) // 2
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(bert_dim + linguistic_dim, bert_dim),
            nn.LayerNorm(bert_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout_rate)
        )
        
        self.classifier = self._build_classifier(bert_dim, num_labels)
        
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
        
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        position_enhanced = self.positional_encoder(sequence_output)
        pooled_output = self._attention_pooling(position_enhanced, attention_mask)
        
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
        
        combined_features = torch.cat([pooled_output, linguistic_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        logits = self.classifier(fused_features)
        severity_pred = self.severity_head(fused_features)
        fluency_pred = self.fluency_head(fused_features)
        
        return {
            "logits": logits,
            "severity_pred": severity_pred,
            "fluency_pred": fluency_pred,
            "loss": None
        }
    
    def _attention_pooling(self, sequence_output, attention_mask):
        attention_weights = torch.softmax(
            torch.sum(sequence_output, dim=-1, keepdim=True), dim=1
        )
        attention_weights = attention_weights * attention_mask.unsqueeze(-1).float()
        attention_weights = attention_weights / (torch.sum(attention_weights, dim=1, keepdim=True) + 1e-9)
        pooled = torch.sum(sequence_output * attention_weights, dim=1)
        return pooled


class AphasiaInferenceSystem:
    """失語症分類推理系統"""
    
    def __init__(self, model_dir: str):
        """
        初始化推理系統
        Args:
            model_dir: 訓練好的模型目錄路徑
        """
        self.model_dir = '/workspace/SH001/adaptive_aphasia_model'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 失語症類型描述
        self.aphasia_descriptions = {
            "BROCA": {
                "name": "Broca's Aphasia (Non-fluent)",
                "description": "Characterized by limited speech output, difficulty with grammar and sentence formation, but relatively preserved comprehension. Speech is typically effortful and halting.",
                "features": ["Non-fluent speech", "Preserved comprehension", "Grammar difficulties", "Word-finding problems"]
            },
            "TRANSMOTOR": {
                "name": "Trans-cortical Motor Aphasia",
                "description": "Similar to Broca's aphasia but with preserved repetition abilities. Speech is non-fluent with good comprehension.",
                "features": ["Non-fluent speech", "Good repetition", "Preserved comprehension", "Grammar difficulties"]
            },
            "NOTAPHASICBYWAB": {
                "name": "Not Aphasic by WAB",
                "description": "Individuals who do not meet the criteria for aphasia according to the Western Aphasia Battery assessment.",
                "features": ["Normal language function", "No significant language impairment", "Good comprehension", "Fluent speech"]
            },
            "CONDUCTION": {
                "name": "Conduction Aphasia",
                "description": "Characterized by fluent speech with good comprehension but severely impaired repetition. Often involves phonemic paraphasias.",
                "features": ["Fluent speech", "Good comprehension", "Poor repetition", "Phonemic errors"]
            },
            "WERNICKE": {
                "name": "Wernicke's Aphasia (Fluent)",
                "description": "Fluent but often meaningless speech with poor comprehension. Speech may contain neologisms and jargon.",
                "features": ["Fluent speech", "Poor comprehension", "Jargon speech", "Neologisms"]
            },
            "ANOMIC": {
                "name": "Anomic Aphasia",
                "description": "Primarily characterized by word-finding difficulties with otherwise relatively preserved language abilities.",
                "features": ["Word-finding difficulties", "Good comprehension", "Fluent speech", "Circumlocution"]
            },
            "GLOBAL": {
                "name": "Global Aphasia",
                "description": "Severe impairment in all language modalities - comprehension, production, repetition, and naming.",
                "features": ["Severe comprehension deficit", "Non-fluent speech", "Poor repetition", "Severe naming difficulties"]
            },
            "ISOLATION": {
                "name": "Isolation Syndrome",
                "description": "Rare condition with preserved repetition but severely impaired comprehension and spontaneous speech.",
                "features": ["Good repetition", "Poor comprehension", "Limited spontaneous speech", "Echolalia"]
            },
            "TRANSSENSORY": {
                "name": "Trans-cortical Sensory Aphasia",
                "description": "Fluent speech with good repetition but impaired comprehension, similar to Wernicke's but with preserved repetition.",
                "features": ["Fluent speech", "Good repetition", "Poor comprehension", "Semantic errors"]
            }
        }
        
        # 載入模型配置和映射
        self.load_configuration()
        
        # 載入模型
        self.load_model()
        
        print(f"推理系統初始化完成，使用設備: {self.device}")
    
    def load_configuration(self):
        """載入模型配置"""
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            self.aphasia_types_mapping = config_data.get("aphasia_types_mapping", {
                "BROCA": 0, "TRANSMOTOR": 1, "NOTAPHASICBYWAB": 2,
                "CONDUCTION": 3, "WERNICKE": 4, "ANOMIC": 5,
                "GLOBAL": 6, "ISOLATION": 7, "TRANSSENSORY": 8
            })
            self.num_labels = config_data.get("num_labels", 9)
            self.model_name = config_data.get("model_name", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        else:
            # 預設配置
            self.aphasia_types_mapping = {
                "BROCA": 0, "TRANSMOTOR": 1, "NOTAPHASICBYWAB": 2,
                "CONDUCTION": 3, "WERNICKE": 4, "ANOMIC": 5,
                "GLOBAL": 6, "ISOLATION": 7, "TRANSSENSORY": 8
            }
            self.num_labels = 9
            self.model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        
        # 建立反向映射
        self.id_to_aphasia_type = {v: k for k, v in self.aphasia_types_mapping.items()}
        
    def load_model(self):
        """載入訓練好的模型"""
        # 載入tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        added_tokens_path = os.path.join(self.model_dir, "added_tokens.json")
        if os.path.exists(added_tokens_path):
            with open(added_tokens_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        # 如果是 dict，就取出所有 key 當作要新增的 token 清單
            if isinstance(data, dict):
                tokens = list(data.keys())
            else:
                tokens = data  # 萬一已經是 list，就直接用
            num_added = self.tokenizer.add_tokens(tokens)
            print(f"新增到 tokenizer 的 token 數量: {num_added}")
        # 建立模型配置
        self.config = ModelConfig()
        self.config.model_name = self.model_name
        
        # 建立模型
        self.model = StableAphasiaClassifier(self.config, self.num_labels)
        self.model.bert.resize_token_embeddings(len(self.tokenizer))
        # 載入模型權重
        model_path = os.path.join(self.model_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.load_state_dict(state_dict)
            print("模型權重載入成功")
        else:
            raise FileNotFoundError(f"模型權重文件不存在: {model_path}")
        
        # 調整tokenizer尺寸
        self.model.bert.resize_token_embeddings(len(self.tokenizer))
        
        # 移動到設備並設置為評估模式
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_sentence(self, sentence_data: dict) -> dict:
        """預處理單個句子數據"""
        all_tokens, all_pos, all_grammar, all_durations = [], [], [], []
        
        # 處理對話數據
        for dialogue_idx, dialogue in enumerate(sentence_data.get("dialogues", [])):
            if dialogue_idx > 0:
                all_tokens.append("[DIALOGUE]")
                all_pos.append(0)
                all_grammar.append([0, 0, 0])
                all_durations.append(0.0)
            
            # 處理參與者的語音
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
            return None
        
        # 文本tokenization
        text = " ".join(all_tokens)
        encoded = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 對齊特徵
        aligned_pos, aligned_grammar, aligned_durations = self._align_features(
            all_tokens, all_pos, all_grammar, all_durations, encoded
        )
        
        # 建立韻律特徵
        prosody_features = self._extract_prosodic_features(all_durations, all_tokens)
        prosody_tensor = torch.tensor(prosody_features).unsqueeze(0).repeat(
            self.config.max_length, 1
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "word_pos_ids": torch.tensor(aligned_pos, dtype=torch.long),
            "word_grammar_ids": torch.tensor(aligned_grammar, dtype=torch.long),
            "word_durations": torch.tensor(aligned_durations, dtype=torch.float),
            "prosody_features": prosody_tensor.float(),
            "sentence_id": sentence_data.get("sentence_id", "unknown"),
            "original_tokens": all_tokens,
            "text": text
        }
    
    def _align_features(self, tokens, pos_ids, grammar_ids, durations, encoded):
        """對齊特徵與BERT子詞"""
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
                
                # 處理duration數據
                raw_duration = durations[original_idx] if original_idx < len(durations) else 0.0
                if isinstance(raw_duration, list) and len(raw_duration) >= 2:
                    try:
                        duration_val = float(raw_duration[1]) - float(raw_duration[0])
                    except (ValueError, TypeError):
                        duration_val = 0.0
                elif isinstance(raw_duration, (int, float)):
                    duration_val = float(raw_duration)
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
        """提取韻律特徵"""
        if not durations:
            return [0.0] * self.config.prosody_dim
        
        # 處理duration數據並提取數值
        processed_durations = []
        for d in durations:
            if isinstance(d, list) and len(d) >= 2:
                try:
                    processed_durations.append(float(d[1]) - float(d[0]))
                except (ValueError, TypeError):
                    continue
            elif isinstance(d, (int, float)):
                processed_durations.append(float(d))
        
        if not processed_durations:
            return [0.0] * self.config.prosody_dim
        
        # 計算基本統計特徵
        features = [
            np.mean(processed_durations),
            np.std(processed_durations),
            np.median(processed_durations),
            len([d for d in processed_durations if d > np.mean(processed_durations) * 1.5])
        ]
        
        # 填充至所需維度
        while len(features) < self.config.prosody_dim:
            features.append(0.0)
        
        return features[:self.config.prosody_dim]
    
    def predict_single(self, sentence_data: dict) -> dict:
        """對單個句子進行預測"""
        # 預處理數據
        processed_data = self.preprocess_sentence(sentence_data)
        if processed_data is None:
            return {
                "error": "無法處理輸入數據",
                "sentence_id": sentence_data.get("sentence_id", "unknown")
            }
        
        # 準備輸入數據
        input_data = {
            "input_ids": processed_data["input_ids"].unsqueeze(0).to(self.device),
            "attention_mask": processed_data["attention_mask"].unsqueeze(0).to(self.device),
            "word_pos_ids": processed_data["word_pos_ids"].unsqueeze(0).to(self.device),
            "word_grammar_ids": processed_data["word_grammar_ids"].unsqueeze(0).to(self.device),
            "word_durations": processed_data["word_durations"].unsqueeze(0).to(self.device),
            "prosody_features": processed_data["prosody_features"].unsqueeze(0).to(self.device)
        }
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(**input_data)
            
            logits = outputs["logits"]
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class_id = np.argmax(probabilities)
            
            severity_pred = outputs["severity_pred"].cpu().numpy()[0]
            fluency_pred = outputs["fluency_pred"].cpu().numpy()[0][0]
        
        # 建立結果
        predicted_type = self.id_to_aphasia_type[predicted_class_id]
        confidence = float(probabilities[predicted_class_id])
        
        # 建立機率分佈
        probability_distribution = {}
        for aphasia_type, type_id in self.aphasia_types_mapping.items():
            probability_distribution[aphasia_type] = {
                "probability": float(probabilities[type_id]),
                "percentage": f"{probabilities[type_id]*100:.2f}%"
            }
        
        # 排序機率分佈
        sorted_probabilities = sorted(
            probability_distribution.items(), 
            key=lambda x: x[1]["probability"], 
            reverse=True
        )
        
        result = {
            "sentence_id": processed_data["sentence_id"],
            "input_text": processed_data["text"],
            "original_tokens": processed_data["original_tokens"],
            "prediction": {
                "predicted_class": predicted_type,
                "confidence": confidence,
                "confidence_percentage": f"{confidence*100:.2f}%"
            },
            "class_description": self.aphasia_descriptions.get(predicted_type, {
                "name": predicted_type,
                "description": "Description not available",
                "features": []
            }),
            "probability_distribution": dict(sorted_probabilities),
            "additional_predictions": {
                "severity_distribution": {
                    "level_0": float(severity_pred[0]),
                    "level_1": float(severity_pred[1]), 
                    "level_2": float(severity_pred[2]),
                    "level_3": float(severity_pred[3])
                },
                "predicted_severity_level": int(np.argmax(severity_pred)),
                "fluency_score": float(fluency_pred),
                "fluency_rating": "High" if fluency_pred > 0.7 else "Medium" if fluency_pred > 0.4 else "Low"
            }
        }
        
        return result
    
    def predict_batch(self, input_file: str, output_file: str = None) -> List[dict]:
        """批次預測JSON文件中的所有句子"""
        # 載入輸入文件
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        sentences = data.get("sentences", [])
        results = []
        
        print(f"開始處理 {len(sentences)} 個句子...")
        
        for i, sentence in enumerate(sentences):
            print(f"處理第 {i+1}/{len(sentences)} 個句子...")
            result = self.predict_single(sentence)
            results.append(result)
        
        # 建立摘要統計
        summary = self._generate_summary(results)
        
        final_output = {
            "summary": summary,
            "total_sentences": len(results),
            "predictions": results
        }
        
        # 保存結果
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
            print(f"結果已保存到: {output_file}")
        
        return final_output
    
    def _generate_summary(self, results: List[dict]) -> dict:
        """生成預測結果摘要"""
        if not results:
            return {}
        
        # 統計各類別預測數量
        class_counts = defaultdict(int)
        confidence_scores = []
        fluency_scores = []
        severity_levels = defaultdict(int)
        
        for result in results:
            if "error" not in result:
                predicted_class = result["prediction"]["predicted_class"]
                confidence = result["prediction"]["confidence"]
                fluency = result["additional_predictions"]["fluency_score"]
                severity = result["additional_predictions"]["predicted_severity_level"]
                
                class_counts[predicted_class] += 1
                confidence_scores.append(confidence)
                fluency_scores.append(fluency)
                severity_levels[severity] += 1
        
        # 計算統計數據
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        avg_fluency = np.mean(fluency_scores) if fluency_scores else 0
        
        summary = {
            "classification_distribution": dict(class_counts),
            "classification_percentages": {
                k: f"{v/len(results)*100:.1f}%" 
                for k, v in class_counts.items()
            },
            "average_confidence": f"{avg_confidence:.3f}",
            "average_fluency_score": f"{avg_fluency:.3f}",
            "severity_distribution": dict(severity_levels),
            "confidence_statistics": {
                "mean": f"{np.mean(confidence_scores):.3f}",
                "std": f"{np.std(confidence_scores):.3f}",
                "min": f"{np.min(confidence_scores):.3f}",
                "max": f"{np.max(confidence_scores):.3f}"
            } if confidence_scores else {},
            "most_common_prediction": max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else "None"
        }
        
        return summary
    
    def generate_detailed_report(self, results: List[dict], output_dir: str = "./inference_results"):
        """生成詳細的分析報告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 建立詳細的CSV報告
        report_data = []
        for result in results:
            if "error" not in result:
                row = {
                    "sentence_id": result["sentence_id"],
                    "predicted_class": result["prediction"]["predicted_class"],
                    "confidence": result["prediction"]["confidence"],
                    "class_name": result["class_description"]["name"],
                    "severity_level": result["additional_predictions"]["predicted_severity_level"],
                    "fluency_score": result["additional_predictions"]["fluency_score"],
                    "fluency_rating": result["additional_predictions"]["fluency_rating"],
                    "input_text": result["input_text"]
                }
                
                # 添加各類別機率
                for aphasia_type in self.aphasia_types_mapping.keys():
                    row[f"prob_{aphasia_type}"] = result["probability_distribution"][aphasia_type]["probability"]
                
                report_data.append(row)
        
        # 保存CSV
        if report_data:
            df = pd.DataFrame(report_data)
            df.to_csv(os.path.join(output_dir, "detailed_predictions.csv"), index=False, encoding='utf-8')
            
            # 生成統計摘要
            summary_stats = {
                "total_predictions": len(report_data),
                "class_distribution": df["predicted_class"].value_counts().to_dict(),
                "average_confidence": df["confidence"].mean(),
                "confidence_std": df["confidence"].std(),
                "average_fluency": df["fluency_score"].mean(),
                "fluency_std": df["fluency_score"].std(),
                "severity_distribution": df["severity_level"].value_counts().to_dict()
            }
            
            with open(os.path.join(output_dir, "summary_statistics.json"), "w", encoding="utf-8") as f:
                json.dump(summary_stats, f, ensure_ascii=False, indent=2)
            
            print(f"詳細報告已生成並保存到: {output_dir}")
            return df
        
        return None


def main():
    """主程式 - 命令行介面"""
    import argparse
    
    parser = argparse.ArgumentParser(description="失語症分類推理系統")
    parser.add_argument("--model_dir", type=str, default = '/workspace/SH001/adaptive_aphasia_model',
                       help="訓練好的模型目錄路徑")
    parser.add_argument("--input_file", type=str, default = '/workspace/SH001/website/sample.input.json',
                       help="輸入JSON文件路徑")
    parser.add_argument("--output_file", type=str, default="./aphasia_predictions.json",
                       help="輸出JSON文件路徑")
    parser.add_argument("--report_dir", type=str, default="./inference_results",
                       help="詳細報告輸出目錄")
    parser.add_argument("--generate_report", action="store_true",
                       help="是否生成詳細的CSV報告")
    
    args = parser.parse_args()
    
    try:
        # 初始化推理系統
        print("正在初始化推理系統...")
        inference_system = AphasiaInferenceSystem(args.model_dir)
        
        # 執行批次預測
        print("開始執行批次預測...")
        results = inference_system.predict_batch(args.input_file, args.output_file)
        
        # 生成詳細報告
        if args.generate_report:
            print("生成詳細報告...")
            inference_system.generate_detailed_report(results["predictions"], args.report_dir)
        
        # 顯示摘要
        print("\n=== 預測摘要 ===")
        summary = results["summary"]
        print(f"總句子數: {results['total_sentences']}")
        print(f"平均信心度: {summary.get('average_confidence', 'N/A')}")
        print(f"平均流利度: {summary.get('average_fluency_score', 'N/A')}")
        print(f"最常見預測: {summary.get('most_common_prediction', 'N/A')}")
        
        print("\n類別分佈:")
        for class_name, count in summary.get("classification_distribution", {}).items():
            percentage = summary.get("classification_percentages", {}).get(class_name, "0%")
            print(f"  {class_name}: {count} ({percentage})")
        
        print(f"\n結果已保存到: {args.output_file}")
        
    except Exception as e:
        print(f"錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


# 使用範例
def example_usage():
    """使用範例"""
    
    # 1. 基本使用
    print("=== 失語症分類推理系統使用範例 ===\n")
    
    # 範例輸入數據
    sample_input = {
        "sentences": [
            {
                "sentence_id": "S1",
                "aphasia_type": "BROCA",  # 這在推理時會被忽略
                "dialogues": [
                    {
                        "INV": [
                            {
                                "tokens": ["how", "are", "you", "feeling"],
                                "word_pos_ids": [9, 10, 5, 6],
                                "word_grammar_ids": [[1, 4, 11], [2, 4, 2], [3, 4, 1], [4, 0, 3]],
                                "word_durations": [["how", 300], ["are", 200], ["you", 150], ["feeling", 500]]
                            }
                        ],
                        "PAR": [
                            {
                                "tokens": ["I", "feel", "good"],
                                "word_pos_ids": [1, 6, 8],
                                "word_grammar_ids": [[1, 2, 1], [2, 3, 2], [3, 4, 8]],
                                "word_durations": [["I", 200], ["feel", 400], ["good", 600]]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    # 保存範例輸入
    with open("sample_input.json", "w", encoding="utf-8") as f:
        json.dump(sample_input, f, ensure_ascii=False, indent=2)
    
    print("範例輸入文件已創建: sample_input.json")
    
    # 顯示使用說明
    usage_instructions = """
使用方法:

1. 命令行使用:
   python aphasia_inference.py \\
       --model_dir ./adaptive_aphasia_model \\
       --input_file sample_input.json \\
       --output_file predictions.json \\
       --generate_report \\
       --report_dir ./results

2. Python代碼使用:
   from aphasia_inference import AphasiaInferenceSystem
   
   # 初始化系統
   system = AphasiaInferenceSystem("./adaptive_aphasia_model")
   
   # 單個預測
   with open("sample_input.json", "r") as f:
       data = json.load(f)
   result = system.predict_single(data["sentences"][0])
   
   # 批次預測
   results = system.predict_batch("sample_input.json", "output.json")

3. 輸出格式:
   - JSON格式包含詳細的預測結果和機率分佈
   - CSV格式包含表格化的預測數據
   - 統計摘要包含整體分析結果

4. 支援的失語症類型:
   - BROCA: 布若卡失語症
   - WERNICKE: 韋尼克失語症  
   - ANOMIC: 命名性失語症
   - CONDUCTION: 傳導性失語症
   - GLOBAL: 全面性失語症
   - 以及其他類型...
"""
    
    print(usage_instructions)


if __name__ == "__main__":
    # 如果作為腳本執行，運行主程式
    main()
    
    # 如果想看使用範例，取消下面這行的註釋
    # example_usage()