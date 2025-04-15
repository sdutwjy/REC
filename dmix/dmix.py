import os
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

from mmcv.cnn.bricks import ConvModule



class Attention(nn.Module):  ### 修改为适应 1D 卷积和注意力
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1,):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        
        # 使用 1D 卷积替代 2D 卷积
        self.q = nn.Conv1d(dim, dim, kernel_size=1)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=sr_ratio + 3, stride=sr_ratio, padding=(sr_ratio + 3) // 2),
                nn.Conv1d(dim, dim, kernel_size=1)
            )
        else:
            self.sr = nn.Identity()
        
        # 修改 local_conv 为适应 1D 卷积
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, L, C = x.shape  # 输入形状: [B, L, C] (例如 [1, 14257, 256])
        
        # 转换输入格式
        x_t = x.transpose(1, 2)  # [B, C, L]
        
        # 计算查询向量
        q = self.q(x_t)  # [B, C, L]
        q = q.reshape(B, self.num_heads, C // self.num_heads, L).transpose(-1, -2)  # [B, num_heads, L, C//num_heads]
        
        # 如果序列长度太长，分块处理
        CHUNK_SIZE = 1000  # 调整块大小以适应GPU内存
        if L > CHUNK_SIZE:
            # 分块处理 sr 和 local_conv
            kv_chunks = []
            for i in range(0, L, CHUNK_SIZE):
                end = min(i + CHUNK_SIZE, L)
                chunk = x_t[:, :, i:end]  # [B, C, chunk_size]
                
                # 处理当前块
                kv_chunk = self.sr(chunk)  # 应用 sr
                kv_chunk = self.local_conv(kv_chunk) + kv_chunk  # 添加 local_conv
                kv_chunks.append(kv_chunk)
                
                # 主动释放内存
                torch.cuda.empty_cache()
            
            # 将所有块连接起来
            kv = torch.cat(kv_chunks, dim=2)  # [B, C, L]
        else:
            # 无需分块，直接处理
            kv = self.sr(x_t)  # [B, C, L]
            kv = self.local_conv(kv) + kv  # [B, C, L]
        
        # 计算键值向量
        kv = self.kv(kv)  # [B, 2*C, L]
        k, v = torch.chunk(kv, chunks=2, dim=1)  # 各自 [B, C, L]
        
        # 重新塑形 K 和 V
        k = k.reshape(B, self.num_heads, C // self.num_heads, L)  # [B, num_heads, C//num_heads, L]
        v = v.reshape(B, self.num_heads, C // self.num_heads, L).transpose(-1, -2)  # [B, num_heads, L, C//num_heads]
        
        # 分块计算注意力以减少内存使用
        out = []
        for i in range(0, L, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, L)
            # 提取查询的当前块
            q_chunk = q[:, :, i:end, :]  # [B, num_heads, chunk_size, C//num_heads]
            
            # 计算注意力分数
            attn = (q_chunk @ k) * self.scale  # [B, num_heads, chunk_size, L]
            
            # 添加相对位置编码（如果有）
            if relative_pos_enc is not None:
                if attn.shape[2:] != relative_pos_enc.shape[2:]:
                    relative_pos_enc = F.interpolate(
                        relative_pos_enc, 
                        size=attn.shape[2:], 
                        mode='bicubic', 
                        align_corners=False
                    )
                attn = attn + relative_pos_enc[:, :, i:end, :]
            
            # 应用 softmax 和 dropout
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            
            # 计算输出
            out_chunk = (attn @ v)  # [B, num_heads, chunk_size, C//num_heads]
            out.append(out_chunk)
            
            # 主动释放内存
            torch.cuda.empty_cache()
        
        # 连接所有输出块
        x = torch.cat(out, dim=2).transpose(-1, -2)  # [B, num_heads, C//num_heads, L]
        x = x.reshape(B, L, C)  # [B, L, C]
        
        return x

class DynamicConv1d(nn.Module):  ### 修改为适应 1D 卷积
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        
        # 修改 weight 为 1D 卷积核
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size), requires_grad=True)
        
        # 适应 1D 输入的池化操作
        self.pool = nn.AdaptiveAvgPool1d(kernel_size)
        
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim // reduction_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim // reduction_ratio, dim * num_groups, kernel_size=1)
        )

        # 处理 bias
        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, L, C = x.shape  # 输入形状: [B, L, C] (例如 [1, 14257, 256])
        
        # 转换输入格式，为后续操作做准备
        x_t = x.transpose(1, 2)  # [B, C, L]
        
        # 内存优化：分块计算投影
        CHUNK_SIZE = 1000  # 调整块大小
        if L > CHUNK_SIZE:
            # 使用自适应池化来减少计算量
            pooled = self.pool(x_t)  # [B, C, K]
            scale = self.proj(pooled)  # [B, C*num_groups, K]
        else:
            pooled = self.pool(x_t)  # [B, C, K]
            scale = self.proj(pooled)  # [B, C*num_groups, K]
        
        # 重新塑形并应用 softmax
        scale = scale.reshape(B, self.num_groups, C, self.K)
        scale = torch.softmax(scale, dim=1)  # softmax 归一化
        
        # 计算动态权重
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1)  # 按维度 1 求和
        weight = weight.reshape(-1, 1, self.K)

        # 处理偏置项
        if self.bias is not None:
            # 计算均值，减少内存使用
            x_mean = torch.mean(x_t, dim=2, keepdim=True)  # [B, C, 1]
            scale_bias = self.proj(x_mean)  # [B, C*num_groups, 1]
            scale_bias = torch.softmax(scale_bias.reshape(B, self.num_groups, C), dim=1)
            bias = scale_bias * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        # 分块卷积，减少内存使用
        if L > CHUNK_SIZE:
            x_flat = x_t.reshape(1, -1, L)  # [1, B*C, L]
            result_chunks = []
            
            for i in range(0, L, CHUNK_SIZE):
                end = min(i + CHUNK_SIZE, L)
                chunk_len = end - i
                
                # 对当前块应用卷积
                chunk = x_flat[:, :, i:end]  # [1, B*C, chunk_len]
                
                # 使用分组卷积处理
                result_chunk = F.conv1d(
                    chunk,
                    weight=weight,
                    padding=self.K // 2,
                    groups=B * C,
                    bias=bias if i == 0 else None  # 只在第一个块应用偏置
                )
                
                # 如果不是第一个块，需要处理边界重叠
                if i > 0:
                    # 只保留非重叠部分
                    pad_size = self.K // 2
                    if pad_size > 0:
                        result_chunk = result_chunk[:, :, pad_size:]
                
                result_chunks.append(result_chunk)
                
                # 主动释放内存
                torch.cuda.empty_cache()
            
            # 连接所有结果块
            result = torch.cat(result_chunks, dim=2)
            result = result.reshape(B, L, C)
        else:
            # 无需分块，直接应用卷积
            result = F.conv1d(
                x_t.reshape(1, -1, L),
                weight=weight,
                padding=self.K // 2,
                groups=B * C,
                bias=bias
            )
            result = result.reshape(B, L, C)
        
        return result

class HybridTokenMixer(nn.Module): ### D-Mixer
    def __init__(self, 
                 dim,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv1d(
            dim=dim//2, kernel_size=kernel_size, num_groups=num_groups)
        self.global_unit = Attention(
            dim=dim//2, num_heads=num_heads, sr_ratio=sr_ratio)
        
        inner_dim = max(16, dim//reduction_ratio)
        # 简化投影层，使用1D卷积替代2D卷积以节省内存
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(inner_dim),
            nn.Conv1d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),)

    def forward(self, x, relative_pos_enc=None):
        B, L, C = x.shape
        
        # 分块拆分并处理
        CHUNK_SIZE = 1000  # 调整块大小
        
        # 将输入分为两半
        x1, x2 = torch.chunk(x, chunks=2, dim=2)  # 各自 [B, L, C/2]
        
        # 处理第一部分：本地处理
        x1 = self.local_unit(x1)  # [B, L, C/2]
        
        # 处理第二部分：全局注意力
        x2 = self.global_unit(x2, relative_pos_enc)  # [B, L, C/2]
        
        # 合并结果
        x = torch.cat([x1, x2], dim=2)  # [B, L, C]
        
        # 应用投影层
        if L > CHUNK_SIZE:
            # 分块处理投影
            x_t = x.transpose(1, 2)  # [B, C, L]
            result_chunks = []
            
            for i in range(0, L, CHUNK_SIZE):
                end = min(i + CHUNK_SIZE, L)
                # 处理当前块
                chunk = x_t[:, :, i:end]  # [B, C, chunk_size]
                result_chunk = self.proj(chunk)  # [B, C, chunk_size]
                result_chunks.append(result_chunk)
                
                # 主动释放内存
                torch.cuda.empty_cache()
            
            # 连接所有结果块
            result = torch.cat(result_chunks, dim=2)  # [B, C, L]
            result = result + x_t  # 残差连接
            result = result.transpose(1, 2)  # [B, L, C]
        else:
            # 直接应用投影层
            x_t = x.transpose(1, 2)  # [B, C, L]
            result = self.proj(x_t) + x_t  # 残差连接
            result = result.transpose(1, 2)  # [B, L, C]
        
        return result