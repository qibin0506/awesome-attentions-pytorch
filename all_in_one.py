import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, hidden_size, n_heads, *, n_key_value_heads=None):
        """
        :param hidden_size: 隐藏层dim
        :param n_heads: 多少个头
        :param n_key_value_heads: 分组大小，
            当n_key_value_heads=None时，n_groups==head_size，此时是普通的multi heads attention
            当n_key_value_heads=1时，此时是multi query attention
            当n_key_value_heads>1时，此时是group query attention
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_key_value_heads = n_key_value_heads if n_key_value_heads is not None else n_heads
        self.n_key_value_groups = self.n_heads // self.n_key_value_heads
        self.head_size = hidden_size // n_heads
        self.scale = self.head_size ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_key_value_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_key_value_heads * self.head_size, bias=False)
        self.to_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):
        batch, seq_len, hidden_size = x.shape

        # (batch, seq_len, hidden_size)
        q = self.q_proj(x)
        # (batch, seq_len, n_key_value_heads*head_size)
        k = self.k_proj(x)
        # (batch, seq_len, n_key_value_heads*head_size)
        v = self.v_proj(x)

        # q (batch, seq_len, n_heads, head_size)
        # k,v (batch, seq_len, n_key_value_heads, head_size)
        q, k, v = map(lambda t: t.reshape(batch, seq_len, -1, self.head_size), (q, k, v))
        # q (batch, n_heads, seq_len, head_size)
        # k,v (batch, n_key_value_heads, seq_len, head_size)
        q, k, v = map(lambda t: t.permute(0, 2, 1, 3), (q, k, v))

        if self.n_key_value_heads == 1:
            # k,v (batch, 1, seq_len, head_size)
            pass
        else:
            # k,v (batch, n_key_value_heads, 1, seq_len, head_size)
            k, v = map(lambda t: t[:, :, None, :, :], (k, v))
            # k,v (batch, n_key_value_heads, n_key_value_groups=n_heads//n_key_value_heads, seq_len, head_size)
            k, v = map(
                lambda t: t.expand(batch, self.n_key_value_heads, self.n_key_value_groups, seq_len, self.head_size),
                (k, v))
            # k,v (batch, n_heads=n_key_value_heads*n_key_value_groups, seq_len, head_size)
            k, v = map(
                lambda t: t.reshape(batch, self.n_key_value_heads * self.n_key_value_groups, seq_len, self.head_size),
                (k, v))

        # (batch, n_heads, seq_len, seq_len)
        scores = (self.scale * q) @ k.transpose(-1, -2)
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), -torch.inf)

        # (batch, n_heads, seq_len, seq_len)
        weights = self.dropout(scores.softmax(-1, dtype=torch.float32))
        # (batch, n_heads, seq_len, head_size)
        attn = weights @ v
        # (batch, seq_len, n_heads, head_size)
        attn = attn.permute(0, 2, 1, 3)
        # (batch, seq_len, hidden_size)
        attn = attn.reshape(batch, seq_len, self.hidden_size)
        return self.to_out(attn)
