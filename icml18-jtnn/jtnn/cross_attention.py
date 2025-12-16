import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, query_embed_dim, key_value_embed_dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_embed_dim // num_heads
        
        # 查询向量的线性变换（来自序列A）
        self.query = nn.Linear(query_embed_dim, query_embed_dim)
        
        # 键和值向量的线性变换（来自序列B）
        self.key = nn.Linear(key_value_embed_dim, query_embed_dim)
        self.value = nn.Linear(key_value_embed_dim, query_embed_dim)
        
        # 最终输出线性层 
        self.fc = nn.Linear(query_embed_dim, query_embed_dim)

    def forward(self, query, key_value, mask=None):
        batch_size_q, seq_len_q, embed_dim_q = query.size()
        batch_size_kv, seq_len_kv, embed_dim_kv = key_value.size()
        
        # 确保输入维度匹配
        assert embed_dim_q == self.query.out_features, "Query embedding dim mismatch"
        assert embed_dim_kv == self.key.in_features, "Key/Value embedding dim mismatch"
        
        # 生成Q/K/V 
        Q = self.query(query)  # (B_q, L_q, E_q)
        K = self.key(key_value)  # (B_kv, L_kv, E_kv) -> 需要调整维度
        V = self.value(key_value)  # (B_kv, L_kv, E_kv) -> 需要调整维度
        
        # 调整K/V的batch维度以匹配Q（假设B_q=1, B_kv=1）
        K = K.unsqueeze(0) if batch_size_kv == 1 else K  # (1, L_kv, E_kv) if B_kv=1
        V = V.unsqueeze(0) if batch_size_kv == 1 else V
        
        # 分头处理 
        Q = Q.view(batch_size_q, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B_q, H, L_q, D_h)
        K = K.view(1, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (1, H, L_kv, D_h)
        V = V.view(1, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (1, H, L_kv, D_h)

        # 计算缩放点积注意力分数 
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B_q, H, L_q, L_kv)
        
        # 应用掩码（可选）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax归一化 
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 加权求和 
        attn_output = torch.matmul(attn_weights, V)  # (B_q, H, L_q, D_h)
        
        # 拼接多头输出 
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size_q, seq_len_q, embed_dim_q)  # (B_q, L_q, E_q)
        
        # 最终线性变换 
        output = self.fc(attn_output)
        return output, attn_weights

# 示例使用
# 将序列B转换为嵌入向量（示例中简单使用随机初始化）
# B_embedded = torch.randn(len(B), embed_dim)  # (L_kv, E)
# B_embedded = B_embedded.unsqueeze(0)  # 添加batch维度 (B=1, L_kv, E)

# cross_attn = CrossAttention(embed_dim, embed_dim, num_heads=1)
# output, attn_weights = cross_attn(A_embedded, B_embedded)

# print("Cross-Attention Output Shape:", output.shape)  # (1, 6, 8)
# print("Attention Weights Shape:", attn_weights.shape)  # (1, 1, 6, 4)