import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super().__init__()

        # 初始化位置编码矩阵，大小为：(max_seq_len, embed_dim)
        self.encoding = torch.zeros(max_seq_len, embed_dim)

        # 创建一个表示位置得向量，大小为：(max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) 

        # 计算div_term，用于生成正弦和余弦函数得周期性变化，大小为：(embed_dim // 2)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -math.log(10000) / embed_dim)

        # 对偶数索引位置应用正弦函数，奇数索引位置应用余弦函数
        # 这里position与div_term相乘得结果会被广播以匹配encoding矩阵得尺寸
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

        # 将位置编码扩展到(batch_size, max_seq_len, embed_dim)，但实际使用时只选取相应序列长度得部分
        self.encoding = self.encoding.unsqueeze(0) # (1, max_seq_len, embed_dim)

    def forward(self, x):
        # x是输入张量，形状通常是(batch_size, seq_len, embed_dim)
        # 将位置编码加到输入张量上，并且detach()确保位置编码不参与梯度计算
        return x + self.encoding[:, :x.size(1)].detach() # 输出形状：(batch_size, seq_len, embed_dim)


if  __name__ == '__main__':
    embed_dim = 128
    max_seq_len = 100
    positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
    input_tensor = torch.randn(1, max_seq_len, embed_dim)
    output = positional_encoding(input_tensor)
    print(output.shape)
    print(output)