#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cha2json.py ── 將單一 CLAN .cha 轉成 JSON（強化 %mor/%wor 對齊）

只要：
$ python3 cha2json.py
"""

# ────────── 這兩行改成你的固定路徑 ──────────
INPUT_CHA   = "/workspace/SH001/website/ACWT01a(4).cha"
OUTPUT_JSON = "/workspace/SH001/website/Output.json"
# ──────────────────────────────────────────

import re, json, sys
from pathlib import Path
from collections import defaultdict

TAG_PREFIXES = ("*PAR:", "*INV:", "%mor:", "%gra:", "%wor:", "@")
WORD_RE      = re.compile(r"[A-Za-z0-9]+")

# ────────── 同義集合（加速對齊） ──────────
SYN_SETS = [
    {"be", "am", "is", "are", "was", "were"},
    {"have", "has", "had"},
    {"do", "does", "did"},
    {"go", "going", "went", "gone"},
]
def same_syn(a, b):     # 同詞彙不同形態視為相同
    return any(a in s and b in s for s in SYN_SETS)

def canonical(txt):     # token/word → 比對用字串
    head = re.split(r"[~\-\&|]", txt, 1)[0]
    m = WORD_RE.search(head)
    return m.group(0).lower() if m else ""

def merge_multiline(block):   # 合併跨行 %mor/%wor/%gra
    merged, buf = [], None
    for raw in block:
        ln = raw.rstrip("\n").replace("\x15", "")
        if ln.lstrip().startswith("%") and ":" in ln:
            if buf: merged.append(buf)
            buf = ln
        else:
            if buf and ln.strip(): buf += " " + ln.strip()
            else:                  merged.append(ln)
    if buf: merged.append(buf)
    return "\n".join(merged)

# ────────── 主轉換 ──────────
def cha_to_json(lines):
    pos_map     = defaultdict(lambda: len(pos_map)     + 1)
    gra_map     = defaultdict(lambda: len(gra_map)     + 1)
    aphasia_map = defaultdict(lambda: len(aphasia_map))

    data, sent, i = [], None, 0
    while i < len(lines):
        line = lines[i]

        # --- 標頭 / 結尾 ---
        if line.startswith("@UTF8"):
            sent = {"sentence_id": f"S{len(data)+1}",
                    "sentence_pid": None,
                    "aphasia_type": None,
                    "dialogues": []}
            i += 1; continue
        if line.startswith("@End"):
            if sent and sent["aphasia_type"] and sent["dialogues"]:
                data.append(sent)
            sent = None; i += 1; continue

        # --- 句子屬性 ---
        if sent and line.startswith("@PID:"):
            parts = line.split("\t")
            if len(parts) > 1:
                sent["sentence_pid"] = parts[1].strip()
            i += 1; continue
        if sent and line.startswith("@ID:") and "|PAR|" in line:
            aph = line.split("|")[5].strip().upper()
            aphasia_map[aph]
            sent["aphasia_type"] = aph
            i += 1; continue

        # --- 對話行 ---
        if sent and (line.startswith("*INV:") or line.startswith("*PAR:")):
            role = "INV" if line.startswith("*INV:") else "PAR"
            if not sent["dialogues"]:
                sent["dialogues"].append({"INV": [], "PAR": []})
            if role == "INV" and sent["dialogues"][-1]["PAR"]:
                sent["dialogues"].append({"INV": [], "PAR": []})
            sent["dialogues"][-1][role].append(
                {"tokens": [], "word_pos_ids": [], "word_grammar_ids": [], "word_durations": []})
            i += 1; continue

        # --- %mor ---
        if sent and line.startswith("%mor:"):
            blk = [line]; i += 1
            while i < len(lines) and not lines[i].lstrip().startswith(TAG_PREFIXES):
                blk.append(lines[i]); i += 1
            units = merge_multiline(blk).replace("%mor:", "").strip().split()

            toks, pos_ids = [], []
            for u in units:
                if "|" in u:
                    pos, rest = u.split("|", 1)
                    toks.append(rest.split("|", 1)[0])
                    pos_ids.append(pos_map[pos])

            dlg = sent["dialogues"][-1]
            tgt = dlg["PAR"][-1] if dlg["PAR"] else dlg["INV"][-1]
            tgt["tokens"], tgt["word_pos_ids"] = toks, pos_ids
            continue

        # --- %wor ---
        if sent and line.startswith("%wor:"):
            blk = [line]; i += 1
            while i < len(lines) and not lines[i].lstrip().startswith(TAG_PREFIXES):
                blk.append(lines[i]); i += 1
            merged = merge_multiline(blk).replace("%wor:", "").strip()
            raw = re.findall(r"(\S+)\s+(\d+)\D+(\d+)", merged)
            wor = [(w, int(e)-int(s)) for w,s,e in raw]

            dlg = sent["dialogues"][-1]
            tgt = dlg["PAR"][-1] if dlg["PAR"] else dlg["INV"][-1]

            aligned, j = [], 0
            for tok in tgt["tokens"]:
                c_tok = canonical(tok); match = None
                for k in range(j, len(wor)):
                    c_w = canonical(wor[k][0])
                    if (c_tok == c_w or c_w.startswith(c_tok) or c_tok.startswith(c_w)
                        or same_syn(c_tok, c_w)):
                        match = wor[k]; j = k+1; break
                aligned.append([tok, match[1] if match else 0])
            tgt["word_durations"] = aligned
            continue

        # --- %gra ---
        if sent and line.startswith("%gra:"):
            blk = [line]; i += 1
            while i < len(lines) and not lines[i].lstrip().startswith(TAG_PREFIXES):
                blk.append(lines[i]); i += 1
            units = merge_multiline(blk).replace("%gra:", "").strip().split()

            triples = []
            for u in units:
                a,b,r = u.split("|")
                if a.isdigit() and b.isdigit():
                    triples.append([int(a), int(b), gra_map[r]])

            dlg = sent["dialogues"][-1]
            (dlg["PAR"][-1] if dlg["PAR"] else dlg["INV"][-1])["word_grammar_ids"] = triples
            continue

        i += 1  # 其他行

    return {"sentences": data,
            "pos_mapping": dict(pos_map),
            "grammar_mapping": dict(gra_map),
            "aphasia_types": dict(aphasia_map)}

# ────────── 執行 ──────────
def main():
    in_path  = Path(INPUT_CHA)
    out_path = Path(OUTPUT_JSON)

    if not in_path.exists():
        sys.exit(f"❌ 找不到檔案: {in_path}")

    with in_path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()

    dataset = cha_to_json(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=4)

    print(f"✅ 轉換完成 → {out_path}")

if __name__ == "__main__":
    main()
