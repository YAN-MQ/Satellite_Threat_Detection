"""
DSC-CBAM-GRU Model Components
Main model using GRU instead of LSTM for lighter weight
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution - Lightweight conv for resource-constrained environments"""
    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.dw = nn.Conv1d(c_in, c_in, 3, padding=1, groups=c_in)
        self.pw = nn.Conv1d(c_in, c_out, 1)
        self.bn = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    """Channel Attention Module - Learn importance of each channel"""
    
    def __init__(self, c, r=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(),
            nn.Linear(c // r, c, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module - Focus on important positions"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module
    
    Combines channel and spatial attention to focus on relevant features.
    
    Args:
        c: Number of input channels
        r: Reduction ratio for channel attention
    """
    
    def __init__(self, c, r=8):
        super().__init__()
        self.channel_attention = ChannelAttention(c, r)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DSC_CBAM_GRU(nn.Module):
    """DSC-CBAM-GRU Model for Network Threat Detection
    
    Architecture:
        Input(18) → Conv1D(conv_dim) → DSC(dsc_dim) → CBAM → GRU(hidden_dim) → FC(num_classes)

    Args:
        input_dim: Number of input features (default: 18)
        num_classes: Number of output classes (default: 4)
    """
    
    def __init__(
        self,
        input_dim=18,
        num_classes=4,
        hidden_dim=64,
        bidirectional=True,
        dropout=0.3,
        conv_dim=32,
        dsc_dim=64,
    ):
        super().__init__()

        self.conv = nn.Conv1d(input_dim, conv_dim, kernel_size=1)
        self.dsc = DepthwiseSeparableConv(conv_dim, dsc_dim)
        self.cbam = CBAM(dsc_dim)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.gru = nn.GRU(
            dsc_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, window, features) -> (batch, features, window)
        x = x.permute(0, 2, 1)
        
        # Initial feature mapping
        x = self.conv(x)
        
        # Depthwise separable convolution
        x = self.dsc(x)
        
        # Attention mechanism
        x = self.cbam(x)
        
        # GRU expects (batch, seq, features)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        
        # Use last hidden state
        x = x[:, -1, :]
        
        # Classification
        x = self.fc(x)
        return x
    
    def get_num_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelFactory:
    """Factory for creating models"""
    
    @staticmethod
    def create(model_name, **kwargs):
        """Create model by name
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional arguments for model
            
        Returns:
            nn.Module: Created model
        """
        models = {
            'dsc_cbam_gru': DSC_CBAM_GRU,
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        
        return models[model_name](**kwargs)
