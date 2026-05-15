导入 torch
导入torch.nn作为nn
导入torch.nn.functional作为F
import math


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        断言embed_dim % num_heads ==
        self.heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

matmulq, k.-, -))/ 自我。

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)

        context = context.transpose(1, 2).reshape(batch_size, seq_len, -1)
返回自身。proj(上下文)


类TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_expanded):
        super().__init__()
        self.attention = SelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.ModuleList([
            nn.Linear(d_model, d_expanded),
            nn.GELU(),
            nn.Linear(d_expanded, d_model)
        ])

    def forward(self, x, padding_mask=None):
        attn_out = self.attention(x, attention_mask=padding_mask)
        x = self.norm1(x + attn_out)

        ffn_out = x
        for layer in self.ffn:
            ffn_out = layer(ffn_out)

        x = self.norm2(x + ffn_out)
        return x


if __name__ == "__main__":
    block = TransformerEncoderBlock(d_model=512, n_heads=8, d_expanded=2048)
    inputs = torch.randn(2, 16, 512)
    output = block(inputs)
    print(f"Output shape: {output.shape}")
