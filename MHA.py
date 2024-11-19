import torch
from torch import nn

# multi head attention
class MHA(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_size = embed_dim // n_heads
        self.scale = self.head_size ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):
        batch, seq_len, embed_dim = x.shape

        # (batch, seq_len, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (batch, seq_len, n_heads, head_size)
        q, k, v = map(lambda t: t.reshape(batch, seq_len, self.n_heads, self.head_size), (q, k, v))
        # (batch, n_heads, seq_len, head_size)
        q, k, v = map(lambda t: t.permute(0, 2, 1, 3), (q, k, v))

        # (batch, n_heads, seq_len, seq_len)
        scores = (self.scale * q) @ k.transpose(-1, -2)
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), -torch.inf)

        # (batch, n_heads, seq_len, seq_len)
        weights = self.dropout(scores.softmax(-1))
        # (batch, n_heads, seq_len, head_size)
        attn = weights @ v
        # (batch, seq_len, n_heads, head_size)
        attn = attn.permute(0, 2, 1, 3)
        # (batch, seq_len, embed_dim)
        attn = attn.reshape(batch, seq_len, self.embed_dim)
        return self.to_out(attn)


if __name__ == '__main__':
    batch_size = 2
    seq_len = 512
    embed_dim = 1024
    num_heads = 8

    # 随机生成输入数据
    hidden_state = torch.randn(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, hidden_size)

    # 创建多头注意力模块
    mha = MHA(embed_dim, num_heads)

    # 计算多头注意力输出
    output = mha(hidden_state)
    print("Input shape:", hidden_state.shape)
    print("Output shape:", output.shape)