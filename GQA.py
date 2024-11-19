import torch
from torch import nn


# group query attention
class GQA(nn.Module):
    def __init__(self, embed_dim, n_heads, n_groups):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.head_size = embed_dim // n_heads
        self.scale = self.head_size ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # !!!修改点：embed_dim -> self.head_size * n_groups
        self.k_proj = nn.Linear(embed_dim, self.head_size * n_groups, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_size * n_groups, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    # def split_head(self, x, group_num=None):
    #     batch_size, seq_len = x.size()[:2]  # 获取批量大小和序列长度
    #
    #     if group_num is None:
    #         return x.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    #     else:
    #         # 将 hidden_size 分割为 group_num 和 head_dim
    #         x = x.reshape(batch_size, -1, group_num, self.head_dim).transpose(1, 2)
    #         # 再将其手动 expand 到相同大小
    #         x = x[:, :, None, :, :].expand(batch_size, group_num, self.num_heads // group_num, seq_len,
    #                                        self.head_dim).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
    #         return x  # 形状: (batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask=None):
        batch, seq_len, embed_dim = x.shape

        # (batch, seq_len, embed_dim)
        q = self.q_proj(x)
        # (batch, seq_len, n_groups * head_size)
        k = self.k_proj(x)
        # (batch, seq_len, n_groups * head_size)
        v = self.v_proj(x)

        # q (batch, seq_len, n_heads, head_size)
        q = q.reshape(batch, seq_len, -1, self.head_size)
        # q (batch, n_heads, seq_len, head_size)
        q = q.permute(0, 2, 1, 3)

        # !!! k v (batch, seq_len, n_groups, head_size)
        k, v = map(lambda t: t.reshape(batch, -1, self.n_groups, self.head_size), (k, v))
        # !!! k v (batch, n_groups, seq_len, head_size)
        k, v = map(lambda t: t.transpose(1, 2), (k, v))
        # !!! k v (batch, n_groups, 1, seq_len, head_size)
        k, v = map(lambda t: t[:, :, None, :, :], (k, v))
        # !!! k v (batch, n_groups, n_heads//n_groups, seq_len, head_size)
        k, v = map(lambda t: t.expand(batch, self.n_groups, self.n_heads // self.n_groups, seq_len, self.head_size), (k, v))
        # !!! k v (batch, n_heads, seq_len, head_size)
        k, v = map(lambda t: t.reshape(batch, self.n_heads, seq_len, self.head_size), (k, v))

        # (batch, n_heads, seq_len, seq_len)
        scores: torch.Tensor = (self.scale * q) @ k.transpose(-1, -2)
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
    mha = GQA(embed_dim, num_heads, 4)

    # 计算多头注意力输出
    output = mha(hidden_state)
    print("Input shape:", hidden_state.shape)
    print("Output shape:", output.shape)