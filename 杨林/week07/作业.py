序列标注模型训练脚本（BERT + Linear / BE

功能：
  - 支持从 eval_sft.json 提取数据生成训
  - 实现 BERT + Linear 和 BERT + CRF 两
  - 完整的训练/验证/测试流程
  - 实体级 F1 评估

使用方式：
  python train_ner.py                   
  python train_ner.py --use_crf         
  python train_ner.py --epochs 3 --batch
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB

import json
import time
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, Da
from transformers import BertTokenizer, 
from tqdm import tqdm

# 实体类型定义
ENTITY_TYPES = ["PER", "ORG", "LOC"]

def build_label_schema():
    """构建 BIO 标签体系"""
    labels = ["O"]
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    label2id = {lbl: i for i, lbl in enu
    id2label = {i: lbl for lbl, i in lab
    return labels, label2id, id2label

def gold_to_bio(text, entities):
    """将实体列表转换为 BIO 标签序列"""
    labels = ["O"] * len(text)
    for ent in entities:
        ent_text = ent["text"]
        ent_type = ent["type"]
        start = text.find(ent_text)
        if start != -1:
            end = start + len(ent_text)
            labels[start] = f"B-{ent_typ
            for i in range(start + 1, en
                labels[i] = f"I-{ent_typ
    return labels

def load_data_from_eval(eval_path):
    """从 eval_sft.json 提取训练数据"""
    with open(eval_path, "r", encoding="
        eval_data = json.load(f)
    
    samples = []
    for item in eval_data["detail"]:
        text = item["text"]
        entities = item["gold"]
        tokens = list(text)
        ner_tags = gold_to_bio(text, ent
        samples.append({"tokens": tokens
    
    random.seed(42)
    random.shuffle(samples)
    n_train = int(len(samples) * 0.8)
    n_val = int(len(samples) * 0.1)
    return samples[:n_train], samples[n_

class NERDataset(Dataset):
    """序列标注数据集"""
    def __init__(self, records, tokenize
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        tokens = row["tokens"]
        ner_tags = row["ner_tags"]
        
        char_labels = [self.label2id.get
        
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        word_ids = encoding.word_ids(bat
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-1
            elif wid != prev_word_id:
                aligned_labels.append(ch
                prev_word_id = wid
            else:
                aligned_labels.append(-1
        
        return {
            "input_ids": encoding["input
            "attention_mask": encoding["
            "token_type_ids": encoding["
            "labels": torch.tensor(align
        }

class BertNER(nn.Module):
    """BERT + Linear 分类头"""
    def __init__(self, bert_path, num_la
        super().__init__()
        self.bert = BertModel.from_pretr
        self.dropout = nn.Dropout(dropou
        self.classifier = nn.Linear(self
        self.num_labels = num_labels

    def forward(self, input_ids, attenti
        outputs = self.bert(input_ids, a
        logits = self.classifier(self.dr
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logit
        return logits, loss

class BertCRFNER(nn.Module):
    """BERT + CRF 层"""
    def __init__(self, bert_path, num_la
        super().__init__()
        from torchcrf import CRF
        self.bert = BertModel.from_pretr
        self.dropout = nn.Dropout(dropou
        self.classifier = nn.Linear(self
        self.crf = CRF(num_labels, batch
        self.num_labels = num_labels

    def _get_emissions(self, input_ids, 
        outputs = self.bert(input_ids, a
        return self.classifier(self.drop

    def forward(self, input_ids, attenti
        emissions = self._get_emissions(
        mask = attention_mask.bool()
        
        loss = None
        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels_crf == -10
            loss = -self.crf(emissions, 
        return emissions, loss

    def decode(self, input_ids, attentio
        emissions = self._get_emissions(
        mask = attention_mask.bool()
        return self.crf.decode(emissions

def evaluate_epoch(model, loader, id2lab
    """评估模型在验证/测试集上的表现"""
    from seqeval.metrics import f1_score
    
    model.eval()
    total_loss = 0.0
    all_preds, all_golds = [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids
            attention_mask = batch["atte
            token_type_ids = batch["toke
            labels = batch["labels"].to(
            
            if use_crf:
                emissions, loss = model(
                pred_ids_list = model.de
            else:
                logits, loss = model(inp
                pred_ids_list = logits.a
            
            total_loss += loss.item()
            labels_np = labels.cpu().tol
            
            for i in range(len(input_ids
                gold_seq, pred_seq = [],
                token_labels = labels_np
                pred_ids = pred_ids_list
                
                for j, gold_id in enumer
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2l
                    if use_crf:
                        pred_seq.append(
                    else:
                        pred_seq.append(
                
                all_golds.append(gold_se
                all_preds.append(pred_se
    
    return total_loss / len(loader), f1_

def train_one_epoch(model, loader, optim
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc=f"Epoch {ep
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].t
        attention_mask = batch["attentio
        token_type_ids = batch["token_ty
        labels = batch["labels"].to(devi
        
        _, loss = model(input_ids, atten
        loss.backward()
        total_loss += loss.item()
        
        nn.utils.clip_grad_norm_(model.p
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        pbar.set_postfix(loss=f"{loss.it
    
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(des
    parser.add_argument("--use_crf", act
    parser.add_argument("--bert_path", t
    parser.add_argument("--epochs", type
    parser.add_argument("--batch_size", 
    parser.add_argument("--max_length", 
    parser.add_argument("--lr", type=flo
    parser.add_argument("--head_lr_mult"
    parser.add_argument("--warmup_ratio"
    parser.add_argument("--dropout", typ
    args = parser.parse_args()
    
    # 设备配置
    device = torch.device("cuda" if torc
    print(f"设备: {device}")
    
    # 构建标签体系
    labels, label2id, id2label = build_l
    print(f"BIO 标签: {labels}")
    
    # 加载数据
    eval_path = Path(__file__).parent / 
    if not eval_path.exists():
        print(f"错误: 未找到 {eval_path}
        return
    
    train_records, val_records, test_rec
    print(f"数据规模: 训练={len(train_re
    
    # 构建 DataLoader
    tokenizer = BertTokenizer.from_pretr
    train_loader = DataLoader(NERDataset
                              batch_size
    val_loader = DataLoader(NERDataset(v
                            batch_size=a
    test_loader = DataLoader(NERDataset(
                             batch_size=
    
    # 构建模型
    num_labels = len(labels)
    model_cls = BertCRFNER if args.use_c
    model = model_cls(bert_path=args.ber
    print(f"模型: {'BERT+CRF' if args.us
    
    # 优化器
    bert_params = list(model.bert.parame
    head_params = list(model.classifier.
    if args.use_crf:
        head_params += list(model.crf.pa
    
    optimizer = AdamW([
        {"params": bert_params, "lr": ar
        {"params": head_params, "lr": ar
    ], weight_decay=0.01)
    
    total_steps = len(train_loader) * ar
    scheduler = get_linear_schedule_with
        optimizer, 
        num_warmup_steps=int(total_steps
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 
        t0 = time.time()
        train_loss = train_one_epoch(mod
        val_loss, val_f1 = evaluate_epoc
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch}/{args.epoc
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "state_dict": model.stat
                "label2id": label2id,
                "id2label": id2label,
                "args": vars(args)
            }, f"best_{'crf' if args.use
            print(f"  ★ 保存最优模型, F
    
    # 测试集评估
    test_loss, test_f1 = evaluate_epoch(
    print(f"\n测试集结果: test_loss={tes

if __name__ == "__main__":
    main()
