import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, embed_dim, n_heads, *, n_groups=None):
        """
        :param head_size: 头大小
        :param n_heads: 多少个头
        :param n_groups: 分组大小，
            当n_groups=None时，n_groups==head_size，此时是普通的multi heads attention
            当n_groups=1时，此时是multi query attention
            当n_groups>1时，此时是group query attention
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_groups = n_groups if n_groups is not None else n_heads
        self.head_size = embed_dim // n_heads
        self.scale = self.head_size ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.n_groups * self.head_size, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.n_groups * self.head_size, bias=False)
        self.to_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):
        batch, seq_len, embed_dim = x.shape

        # (batch, seq_len, embed_dim)
        q = self.q_proj(x)
        # (batch, seq_len, n_groups*head_size)
        k = self.k_proj(x)
        # (batch, seq_len, n_groups*head_size)
        v = self.v_proj(x)

        # q (batch, seq_len, n_heads, head_size)
        # k,v (batch, seq_len, n_groups, head_size)
        q, k, v = map(lambda t: t.reshape(batch, seq_len, -1, self.head_size), (q, k, v))
        # q (batch, n_heads, seq_len, head_size)
        # k,v (batch, n_groups, seq_len, head_size)
        q, k, v = map(lambda t: t.permute(0, 2, 1, 3), (q, k, v))

        if self.n_groups == 1:
            # k,v (batch, 1, seq_len, head_size)
            pass
        else:
            # k,v (batch, n_groups, 1, seq_len, head_size)
            k, v = map(lambda t: t[:, :, None, :, :], (k, v))
            # k,v (batch, n_groups, n_heads//n_groups, seq_len, head_size)
            k, v = map(
                lambda t: t.expand(batch, self.n_groups, self.n_heads // self.n_groups, seq_len, self.head_size),
                (k, v))
            # k,v (batch, n_heads, seq_len, head_size)
            k, v = map(lambda t: t.reshape(batch, self.n_heads, seq_len, self.head_size), (k, v))

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
    n_heads = 8

    inputs = torch.randn(batch_size, seq_len, embed_dim)
    print(f'inputs shape: {inputs.shape}')

    multi_heads_attention = Attention(embed_dim, n_heads)
    multi_heads_result = multi_heads_attention(inputs)
    print(f'multi_heads_result shape: {multi_heads_result.shape}')

    multi_query_attention = Attention(embed_dim, n_heads, n_groups=1)
    multi_query_attention_result = multi_heads_attention(inputs)
    print(f'multi_query_attention_result shape: {multi_query_attention_result.shape}')

    group_query_attention = Attention(embed_dim, n_heads, n_groups=4)
    group_query_attention_result = multi_heads_attention(inputs)
    print(f'group_query_attention_result shape: {group_query_attention_result.shape}')