import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import os

# ============== 配置参数 ==============
class Config:
    vocab_size = 10000
    d_model = 256
    nhead = 8
    num_decoder_layers = 4
    dim_feedforward = 512
    dropout = 0.1
    max_seq_length = 128
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# ============== 位置编码 ==============
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ============== 掩码生成 ==============
def create_causal_mask(seq_len, device):
    """创建因果掩码，确保只能看到当前位置之前的token"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask

# ============== Transformer Decoder Block ==============
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, tgt_mask=None):
        # 自注意力层
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络
        ff_output = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

# ============== Transformer单向语言模型 ==============
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(x, tgt_mask)
        
        x = self.norm(x)
        logits = self.fc(x)
        return logits

# ============== 简单文本数据集 ==============
class TextDataset(Dataset):
    def __init__(self, text, vocab, max_seq_length):
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.tokens = self.tokenize(text)
    
    def tokenize(self, text):
        words = text.split()
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab['<unk>'])
        return tokens
    
    def __len__(self):
        return max(0, len(self.tokens) - self.max_seq_length)
    
    def __getitem__(self, idx):
        src = self.tokens[idx:idx + self.max_seq_length]
        tgt = self.tokens[idx + 1:idx + self.max_seq_length + 1]
        
        if len(src) < self.max_seq_length:
            src = src + [self.vocab['<pad>']] * (self.max_seq_length - len(src))
            tgt = tgt + [self.vocab['<pad>']] * (self.max_seq_length - len(tgt))
        
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# ============== 训练函数 ==============
def train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        
        # 创建因果掩码
        tgt_mask = create_causal_mask(src.size(1), device)
        
        optimizer.zero_grad()
        logits = model(src, tgt_mask)
        
        # 计算交叉熵损失
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

# ============== 文本生成函数 ==============
def generate_text(model, start_text, vocab, idx_to_word, config, max_length=100, temperature=1.0, top_k=50):
    """使用Transformer语言模型生成文本"""
    model.eval()
    
    # Tokenize start text
    words = start_text.split()
    tokens = []
    for word in words:
        if word in vocab:
            tokens.append(vocab[word])
        else:
            tokens.append(vocab['<unk>'])
    
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(config.device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # 创建因果掩码
            tgt_mask = create_causal_mask(input_ids.size(1), config.device)
            
            # 前向传播
            logits = model(input_ids, tgt_mask)
            
            # 获取最后一个位置的logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k采样
            if top_k > 0:
                values, indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, indices, values)
            
            # 转换为概率分布
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 采样
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 如果是结束符，停止生成
            if next_token.item() == vocab['<eos>']:
                break
            
            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    # 转换回文本
    generated_words = []
    for token_id in input_ids[0].tolist():
        if token_id in idx_to_word and idx_to_word[token_id] not in ['<pad>', '<bos>', '<eos>']:
            generated_words.append(idx_to_word[token_id])
    
    return ' '.join(generated_words)

# ============== 主程序 ==============
def main():
    print(f"使用设备: {config.device}")
    
    # 训练文本 - 使用中文语料
    train_text = """
    深度学习是机器学习的一个分支，它使用多层神经网络来分析各种层次的特征。
    Transformer模型是一种基于注意力机制的神经网络架构，广泛应用于自然语言处理任务。
    语言模型是自然语言处理中的核心技术，它用于预测下一个词或字符的概率分布。
    文本生成是语言模型的重要应用之一，可以用于写作辅助、机器翻译、对话系统等领域。
    注意力机制允许模型在处理序列数据时关注相关的输入部分，从而更好地理解上下文。
    GPT是基于Transformer解码器的单向语言模型，它通过大规模预训练和微调来实现各种NLP任务。
    预训练是指在大规模无标签数据上训练模型，使其学习通用的语言表示。
    微调是指在特定任务的标注数据上对预训练模型进行额外训练，以提高任务性能。
    自回归语言模型逐步生成文本，每个词的生成都依赖于之前生成的词。
    大语言模型通常拥有数十亿参数，能够学习丰富的语言知识和世界知识。
    """
    
    # 构建简单词汇表
    words = list(set(train_text.split()))
    word_to_idx = {word: idx + 3 for idx, word in enumerate(words)}
    word_to_idx['<pad>'] = 0
    word_to_idx['<bos>'] = 1
    word_to_idx['<eos>'] = 2
    word_to_idx['<unk>'] = 3
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # 扩展词汇表以达到配置的vocab_size
    while len(word_to_idx) < config.vocab_size:
        idx = len(word_to_idx)
        word_to_idx[f'<extra_{idx}>'] = idx
        idx_to_word[idx] = f'<extra_{idx}>'
    
    print(f"词汇表大小: {len(word_to_idx)}")
    
    # 创建数据集
    dataset = TextDataset(train_text, word_to_idx, config.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    print(f"训练样本数: {len(dataset)}")
    
    # 初始化模型
    model = TransformerLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout
    ).to(config.device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 50)
    
    for epoch in range(1, config.num_epochs + 1):
        avg_loss = train(model, dataloader, optimizer, criterion, config.device, epoch)
        scheduler.step()
        print(f"Epoch {epoch}/{config.num_epochs} | 平均损失: {avg_loss:.4f}")
        print("-" * 50)
    
    # 保存模型
    model_path = "transformer_lm.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'config': {
            'vocab_size': config.vocab_size,
            'd_model': config.d_model,
            'nhead': config.nhead,
            'num_decoder_layers': config.num_decoder_layers,
            'dim_feedforward': config.dim_feedforward,
            'max_seq_length': config.max_seq_length
        }
    }, model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 文本生成测试
    print("\n" + "=" * 50)
    print("文本生成测试")
    print("=" * 50)
    
    test_prompts = [
        "深度学习是",
        "Transformer模型是",
        "语言模型是"
    ]
    
    for prompt in test_prompts:
        print(f"\n提示词: '{prompt}'")
        generated = generate_text(model, prompt, word_to_idx, idx_to_word, config, max_length=50)
        print(f"生成结果: '{prompt}{generated}'")
    
    print("\n训练完成!")

if __name__ == "__main__":
    main()
