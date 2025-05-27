import torch
import torch.nn as nn

class BatchCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, A, B, key_padding_mask=None):
        # A: (batch_size, seq_len_A, embed_dim) 作为Query
        # B: (batch_size, seq_len_B, embed_dim) 作为Key/Value
        attn_output, attn_weights = self.multihead_attn(
            query=A,
            key=B,
            value=B,
            key_padding_mask=key_padding_mask  # 处理B的填充位置
        )
        return attn_output, attn_weights

# # 示例配置
# batch_size = 4
# embed_dim = 128
# max_seq_len = 10  # 统一填充后的最大长度

# # 生成4对变长数据（A和B长度不同）
# data_pairs = [
#     (torch.randn(5, embed_dim), torch.randn(7, embed_dim)),  # Pair1: A_len=5, B_len=7
#     (torch.randn(3, embed_dim), torch.randn(8, embed_dim)),  # Pair2: A_len=3, B_len=8
#     (torch.randn(6, embed_dim), torch.randn(4, embed_dim)),  # Pair3: A_len=6, B_len=4
#     (torch.randn(9, embed_dim), torch.randn(2, embed_dim)),  # Pair4: A_len=9, B_len=2
# ]

# 填充对齐（统一到max_seq_len）
# def pad_sequences(pairs, max_len):
#     padded_A = [torch.cat([a, torch.zeros(max_len - a.size(0), embed_dim)]) for a, _ in pairs]
#     padded_B = [torch.cat([b, torch.zeros(max_len - b.size(0), embed_dim)]) for _, b in pairs]
#     return torch.stack(padded_A), torch.stack(padded_B)

# A_padded, B_padded = pad_sequences(data_pairs, max_seq_len)  # (4, 10, 128)

# # 生成B的填充掩码（True表示填充位置）
# B_lengths = [b.size(0) for _, b in data_pairs]
# B_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
# for i, l in enumerate(B_lengths):
#     B_mask[i, l:] = True  # 填充位置标记为True

# # 初始化模型
# cross_attn = BatchCrossAttention(embed_dim)

# # 前向计算
# output, attn_weights = cross_attn(A_padded, B_padded, key_padding_mask=B_mask)

# print(f"输出形状: {output.shape}")        # (4, 10, 128)
# print(f"注意力权重形状: {attn_weights.shape}")  # (4, 4, 10, 10)