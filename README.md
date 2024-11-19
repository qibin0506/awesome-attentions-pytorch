# awesome-attentions-pytorch
Implement Multi Head Attention, Multi Query Attention, Group Query Attention in pytorch

## Description
MHA.py: Multi Head Attention

MQA.py: Multi Query Attention

GQA.py: Group Query Attention

all_in_one.py: put everything together


## Usage
``` python
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
```
