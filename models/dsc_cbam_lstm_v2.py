"""
Enhanced DSC-CBAM-LSTM v2 模型 - 数学优化（参数量优化版）

优化点：
1. 双向LSTM - 增强时序特征提取能力
2. 增强型CBAM - 更好的通道和空间注意力
3. 残差连接 - 改善梯度流动
4. 更深的特征提取网络

目标：参数量 < 100K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块 - 优化版"""
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
    """CBAM: 卷积块注意力模块"""
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


class ResidualBlock(nn.Module):
    """轻量残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv1D(channels, channels, kernel_size=3, padding=1)
        self.conv2 = DepthwiseSeparableConv1D(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual  # 残差连接


class DSC_CBAM_LSTM_V2(nn.Module):
    """
    增强版 DSC-CBAM-LSTM V2

    数学优化：
    1. 双向LSTM: 增强时序特征提取
    2. 增强型CBAM: 更好的通道和空间注意力
    3. 残差连接: 改善梯度流动
    4. 更深的特征提取网络

    输入: (Batch, Seq_Len, 8) - 10个时间步，8维特征
    输出: 4类分类概率
    """
    def __init__(self, seq_len=10, input_dim=8, hidden_dim=32, num_classes=4, dropout=0.2):
        super(DSC_CBAM_LSTM_V2, self).__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim

        # 输入投影: (B, 10, 8) -> (B, 10, 16)
        self.input_proj = nn.Linear(input_dim, 16)

        # 第一层特征提取
        self.dsc1 = DepthwiseSeparableConv1D(16, 32, kernel_size=3, padding=1)
        self.cbam1 = CBAM(32, reduction=4)
        self.pool1 = nn.MaxPool1d(2)  # (B, 32, 5)

        # 残差块
        self.residual_block = ResidualBlock(32)

        # 第二层特征提取
        self.dsc2 = DepthwiseSeparableConv1D(32, 64, kernel_size=3, padding=1)
        self.cbam2 = CBAM(64, reduction=8)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # 双向LSTM - 关键优化
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,  # 双向
            dropout=0
        )

        # 分类头
        # 双向LSTM输出是hidden_dim*2 = 64
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 32),  # 拼接LSTM输出和全局特征
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 输入投影
        x = self.input_proj(x)

        # 转换维度: (B, 10, 16) -> (B, 16, 10)
        x = x.permute(0, 2, 1)

        # 第一层特征提取
        x = self.dsc1(x)
        x = self.cbam1(x)
        x = self.pool1(x)  # (B, 32, 5)

        # 残差块
        x = self.residual_block(x)

        # 第二层特征提取
        x = self.dsc2(x)
        x = self.cbam2(x)  # (B, 64, 5)

        # 全局特征
        global_feat = self.global_pool(x).squeeze(-1)  # (B, 64)

        # LSTM时序建模
        x = x.permute(0, 2, 1)  # (B, 5, 64)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 拼接双向LSTM输出
        lstm_feat = lstm_out[:, -1, :]  # (B, hidden_dim*2)

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


def create_model_v2(config=None):
    """模型工厂函数"""
    if config is None:
        config = {
            'seq_len': 10,
            'input_dim': 8,
            'hidden_dim': 32,
            'num_classes': 4,
            'dropout': 0.2
        }

    return DSC_CBAM_LSTM_V2(
        seq_len=config.get('seq_len', 10),
        input_dim=config.get('input_dim', 8),
        hidden_dim=config.get('hidden_dim', 32),
        num_classes=config.get('num_classes', 4),
        dropout=config.get('dropout', 0.2)
    )


if __name__ == "__main__":
    # 测试模型
    model = DSC_CBAM_LSTM_V2(seq_len=10, input_dim=8, hidden_dim=32, num_classes=4)

    # 打印模型结构
    print("=" * 60)
    print("DSC-CBAM-LSTM V2 Model Architecture")
    print("=" * 60)
    print(model)
    print("=" * 60)

    # 测试前向传播
    x = torch.randn(4, 10, 8)  # (Batch, Seq_Len, 8)
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
