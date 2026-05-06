"""
Structured DSC-CBAM-GRU model variants for formal compression candidates.
"""

import torch
import torch.nn as nn

from .dsc_cbam_gru import CBAM, DepthwiseSeparableConv


class StructuredDSC_CBAM_GRU(nn.Module):
    """Structured DSC-CBAM-GRU with explicit width knobs.

    Architecture:
        Input -> Conv1D -> DSC -> CBAM -> GRU -> FC -> logits
    """

    def __init__(
        self,
        input_dim=18,
        num_classes=3,
        conv_dim=16,
        dsc_dim=32,
        hidden_dim=24,
        fc_hidden=32,
        dropout=0.4,
        bidirectional=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.conv_dim = conv_dim
        self.dsc_dim = dsc_dim
        self.hidden_dim = hidden_dim
        self.fc_hidden = fc_hidden
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.conv = nn.Conv1d(input_dim, conv_dim, kernel_size=1)
        self.dsc = DepthwiseSeparableConv(conv_dim, dsc_dim)
        self.cbam = CBAM(dsc_dim, r=max(1, min(8, dsc_dim)))

        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.gru = nn.GRU(
            dsc_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Linear(gru_out_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.dsc(x)
        x = self.cbam(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
