"""
DSC-CBAM-LSTM: 轻量化深度学习模型 for 卫星网络威胁检测

架构:
1. 输入层: (Batch, 10, 2) - 时序数据
2. 空间特征提取: 深度可分离卷积 (DSC)
3. 特征增强: CBAM (通道注意力 + 空间注意力)
4. 时序分析: LSTM
5. 分类头: 全连接层 -> Softmax (4类)

优化目标: 降低FLOPs和参数量 (SWaP约束)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    """CBAM: 卷积块注意力模块 (通道 + 空间注意力)"""
    def __init__(self, channels, reduction=8):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class DepthwiseSeparableConv1D(nn.Module):
    """深度可分离卷积: 减少参数量和计算量"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class DSC_CBAM_LSTM(nn.Module):
    """
    DSC-CBAM-LSTM 轻量化模型
    
    输入: (Batch, Seq_Len, 2) - timestamp, packet_size
    输出: 4类分类概率
    """
    def __init__(self, seq_len=10, input_dim=2, hidden_dim=32, num_classes=4, dropout=0.2):
        super(DSC_CBAM_LSTM, self).__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # 输入投影: (B, 10, 8) -> (B, 10, 16)
        self.input_proj = nn.Linear(input_dim, 16)
        
        # DSC特征提取器 (轻量化)
        self.dsc1 = DepthwiseSeparableConv1D(16, 32, kernel_size=3, padding=1)
        self.cbam1 = CBAM(32, reduction=8)
        self.dsc2 = DepthwiseSeparableConv1D(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        
        # 第二层特征提取
        self.dsc3 = DepthwiseSeparableConv1D(32, 64, kernel_size=3, padding=1)
        self.cbam2 = CBAM(64, reduction=8)
        self.dsc4 = DepthwiseSeparableConv1D(64, 64, kernel_size=3, padding=1)
        
        # 全局池化 + Dropout
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM时序建模
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 64, 32),  # 拼接LSTM输出和全局特征
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 输入投影: (B, 10, 8) -> (B, 10, 16)
        x = self.input_proj(x)
        
        # 转换维度: (B, 10, 16) -> (B, 16, 10)
        x = x.permute(0, 2, 1)
        
        # DSC + CBAM 特征提取
        x = self.dsc1(x)
        x = self.cbam1(x)
        x = self.dsc2(x)
        x = self.pool1(x)  # (B, 32, 5)
        
        x = self.dsc3(x)
        x = self.cbam2(x)
        x = self.dsc4(x)  # (B, 64, 5)
        
        # 全局特征
        global_feat = self.global_pool(x).squeeze(-1)  # (B, 64)
        
        # LSTM时序建模
        x = x.permute(0, 2, 1)  # (B, 5, 64)
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_feat = lstm_out[:, -1, :]  # (B, hidden_dim)
        
        # 拼接全局特征和LSTM特征
        combined = torch.cat([lstm_feat, global_feat], dim=1)
        combined = self.dropout(combined)
        
        # 分类
        output = self.classifier(combined)
        return output
    
    def count_parameters(self):
        """计算可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size(self):
        """获取模型大小 (MB)"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024
    
    def flops(self, input_size=None):
        """估算FLOPs (简化版)"""
        if input_size is None:
            input_size = (1, self.seq_len, self.input_dim)
        
        batch_size = input_size[0]
        
        # 简化FLOPs估算
        flops = 0
        
        # Input projection: 10 * 2 * 16
        flops += batch_size * self.seq_len * self.input_dim * 16
        
        # DSC layers
        flops += batch_size * 16 * 3 * 32  # dsc1
        flops += batch_size * 32 * 3 * 32  # dsc2
        flops += batch_size * 32 * 3 * 64  # dsc3
        flops += batch_size * 64 * 3 * 64  # dsc4
        
        # LSTM
        flops += batch_size * 5 * 64 * self.hidden_dim * 4  # gates
        
        # Classifier
        flops += batch_size * (self.hidden_dim + 64) * 32
        flops += batch_size * 32 * 4
        
        return flops


def create_model(config=None):
    """模型工厂函数"""
    if config is None:
        config = {
            'seq_len': 10,
            'input_dim': 2,
            'hidden_dim': 32,
            'num_classes': 4,
            'dropout': 0.2
        }
    
    return DSC_CBAM_LSTM(
        seq_len=config.get('seq_len', 10),
        input_dim=config.get('input_dim', 2),
        hidden_dim=config.get('hidden_dim', 32),
        num_classes=config.get('num_classes', 4),
        dropout=config.get('dropout', 0.2)
    )


if __name__ == "__main__":
    # 测试模型
    model = DSC_CBAM_LSTM(seq_len=10, input_dim=2, hidden_dim=32, num_classes=4)
    
    # 打印模型结构
    print("=" * 60)
    print("DSC-CBAM-LSTM Model Architecture")
    print("=" * 60)
    print(model)
    print("=" * 60)
    
    # 测试前向传播
    x = torch.randn(4, 10, 2)  # (Batch, Seq_Len, 2)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 参数量统计
    num_params = model.count_parameters()
    model_size = model.get_model_size()
    print(f"\n{'='*60}")
    print(f"Model Statistics:")
    print(f"{'='*60}")
    print(f"Parameters: {num_params:,}")
    print(f"Model Size: {model_size:.4f} MB")
    print(f"Target: < 100K parameters")
    
    if num_params < 100000:
        print("✓ Parameter constraint met!")
    else:
        print("✗ Parameter constraint exceeded!")
