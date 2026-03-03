import torch
import torch.nn as nn

class CBAM(nn.Module):
    """
    卷积块注意力模块 (CBAM)
    包含通道注意力和空间注意力，用于加强关键特征提取
    """
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # 1. 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        # 2. 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力加权
        x = x * self.channel_attn(x)
        # 空间注意力加权 (Avg + Max)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        x = x * self.spatial_attn(spatial)
        return x

class DSC_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DSC_Block, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, groups=in_channels, padding=1)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.bn(self.act(self.pointwise(self.depthwise(x))))

class SatThreatModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=3):
        super(SatThreatModel, self).__init__()
        
        # --- Encoder: 特征提取 (DS-Conv + CBAM) ---
        self.input_proj = nn.Linear(input_dim, 32)
        self.encoder = nn.Sequential(
            DSC_Block(32, 64),
            CBAM(64), # 在卷积后引入注意力机制，识别关键通道(如SYN标志位)
            DSC_Block(64, 64)
        )
        
        # --- Predictor: 时序预测 (LSTM) ---
        # 根据规划，使用 LSTM 捕捉流量节奏 (IAT/BPS)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, 
                            num_layers=2, batch_first=True, dropout=0.2)
        
        # --- Classifier: 分类头 ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, 17)
        x = self.input_proj(x) 
        x = x.permute(0, 2, 1) # -> (Batch, 32, Seq_Len)
        
        x = self.encoder(x)    # -> (Batch, 64, Seq_Len)
        
        x = x.permute(0, 2, 1) # -> (Batch, Seq_Len, 64)
        out, (hn, cn) = self.lstm(x) # LSTM 输出包含隐状态和细胞状态
        
        # 取最后一个时间步
        logits = self.classifier(out[:, -1, :])
        return logits