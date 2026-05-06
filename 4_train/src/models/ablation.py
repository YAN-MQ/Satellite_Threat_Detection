"""
Ablation Study Models (GRU-based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardConv1D(nn.Module):
    """Standard 1D Convolution - replaces Depthwise Separable Convolution"""
    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(c_out)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class NoDSC_GRU(nn.Module):
    """GRU model with Standard Conv replacing DSC"""

    def __init__(
        self,
        input_dim=18,
        num_classes=4,
        hidden_dim=64,
        bidirectional=False,
        dropout=0.3,
        conv_dim=32,
        dsc_dim=64,
    ):
        super().__init__()
        from .dsc_cbam_gru import CBAM

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.conv1 = nn.Conv1d(input_dim, conv_dim, kernel_size=1)
        self.conv2 = StandardConv1D(conv_dim, dsc_dim)
        self.cbam = CBAM(dsc_dim)
        self.feature_dropout = nn.Dropout1d(dropout)
        self.gru = nn.GRU(dsc_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam(x)
        x = self.feature_dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return self.fc(x)


class NoCBAM_GRU(nn.Module):
    """GRU model without CBAM"""

    def __init__(
        self,
        input_dim=18,
        num_classes=4,
        hidden_dim=64,
        bidirectional=False,
        dropout=0.3,
        conv_dim=32,
        dsc_dim=64,
    ):
        super().__init__()
        from .dsc_cbam_gru import DepthwiseSeparableConv

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.conv1 = nn.Conv1d(input_dim, conv_dim, kernel_size=1)
        self.dsc = DepthwiseSeparableConv(conv_dim, dsc_dim)
        self.feature_dropout = nn.Dropout1d(dropout)
        self.gru = nn.GRU(dsc_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.dsc(x)
        x = self.feature_dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return self.fc(x)


class NoBiGRU(nn.Module):
    """GRU model with Unidirectional GRU"""

    def __init__(
        self,
        input_dim=18,
        num_classes=4,
        hidden_dim=64,
        dropout=0.3,
        conv_dim=32,
        dsc_dim=64,
    ):
        super().__init__()
        from .dsc_cbam_gru import DepthwiseSeparableConv, CBAM

        self.conv1 = nn.Conv1d(input_dim, conv_dim, kernel_size=1)
        self.dsc = DepthwiseSeparableConv(conv_dim, dsc_dim)
        self.cbam = CBAM(dsc_dim)
        self.feature_dropout = nn.Dropout1d(dropout)
        self.gru = nn.GRU(dsc_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.dsc(x)
        x = self.cbam(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return self.fc(x)


class NoGRU(nn.Module):
    """Model without GRU - replace with lightweight temporal pooling"""

    def __init__(
        self,
        input_dim=18,
        num_classes=4,
        dropout=0.3,
        conv_dim=32,
        dsc_dim=64,
    ):
        super().__init__()
        from .dsc_cbam_gru import DepthwiseSeparableConv, CBAM

        feature_dropout = min(dropout * 1.5, 0.6)
        self.conv1 = nn.Conv1d(input_dim, conv_dim, kernel_size=1)
        self.dsc = DepthwiseSeparableConv(conv_dim, dsc_dim)
        self.cbam = CBAM(dsc_dim)
        self.feature_dropout = nn.Dropout1d(feature_dropout)
        self.pool = nn.AdaptiveAvgPool1d(5)
        self.fc = nn.Sequential(
            nn.Linear(dsc_dim * 5, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.dsc(x)
        x = self.cbam(x)
        x = self.feature_dropout(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_size=(1, 10, 18)):
    """Estimate forward-pass FLOPs from actual tensor shapes.

    Notes:
    - Uses a dummy forward pass plus hooks, so Conv1d/Pooling sequence lengths are
      measured from real intermediate tensors instead of a fixed heuristic.
    - Reports FLOPs with multiply-add counted as 2 floating-point ops.
    """
    total_flops = 0
    hooks = []
    was_training = model.training

    def conv1d_out_len(length, module):
        kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
        padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
        dilation = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
        return ((length + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1

    def pool1d_out_len(length, module):
        kernel = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
        stride = module.stride if module.stride is not None else kernel
        stride = stride if isinstance(stride, int) else stride[0]
        padding = module.padding if isinstance(module.padding, int) else module.padding[0]
        dilation = module.dilation if isinstance(module.dilation, int) else module.dilation[0]
        return ((length + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1

    def hook_fn(module, inputs, output):
        nonlocal total_flops
        x = inputs[0]

        if isinstance(module, nn.Conv1d):
            batch_size = x.shape[0]
            cin = module.in_channels
            cout = module.out_channels
            kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            groups = module.groups
            out_len = output.shape[-1] if hasattr(output, 'shape') else conv1d_out_len(x.shape[-1], module)
            total_flops += int(batch_size * 2 * cin * cout * kernel * out_len / groups)

        elif isinstance(module, nn.Linear):
            batch_size = x.shape[0] if x.ndim > 1 else 1
            total_flops += int(batch_size * 2 * module.in_features * module.out_features)

        elif isinstance(module, nn.GRU):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            h = module.hidden_size
            i = module.input_size
            directions = 2 if module.bidirectional else 1
            gates = 3
            total_flops += int(batch_size * directions * seq_len * gates * 2 * (i * h + h * h + h))

        elif isinstance(module, nn.LSTM):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            h = module.hidden_size
            i = module.input_size
            directions = 2 if module.bidirectional else 1
            gates = 4
            total_flops += int(batch_size * directions * seq_len * gates * 2 * (i * h + h * h + h))

        elif isinstance(module, nn.BatchNorm1d):
            total_flops += int(x.numel() * 2)

        elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
            total_flops += int(output.numel())

        elif isinstance(module, (nn.MaxPool1d, nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool1d)):
            out_len = output.shape[-1] if hasattr(output, 'shape') else pool1d_out_len(x.shape[-1], module)
            total_flops += int(x.shape[0] * x.shape[1] * out_len)

    for module in model.modules():
        if len(list(module.children())) == 0 and not isinstance(module, (nn.Dropout, nn.Flatten)):
            hooks.append(module.register_forward_hook(hook_fn))

    device = next(model.parameters(), torch.empty(0)).device
    dummy = torch.randn(*input_size, device=device)

    model.eval()
    with torch.no_grad():
        model(dummy)

    for hook in hooks:
        hook.remove()
    model.train(was_training)

    return int(total_flops)


class AblationFactory:
    """Factory for ablation study models"""
    
    @staticmethod
    def create(model_name, input_dim=18, num_classes=4, **kwargs):
        from .dsc_cbam_gru import DSC_CBAM_GRU

        if model_name in {'full', 'dsc_cbam_gru'}:
            return DSC_CBAM_GRU(input_dim, num_classes, **kwargs)
        elif model_name == 'ablation_no_dsc':
            return NoDSC_GRU(input_dim, num_classes, **kwargs)
        elif model_name == 'ablation_no_cbam':
            return NoCBAM_GRU(input_dim, num_classes, **kwargs)
        elif model_name == 'ablation_no_bigru':
            return NoBiGRU(input_dim, num_classes, **kwargs)
        elif model_name == 'ablation_no_gru':
            return NoGRU(
                input_dim,
                num_classes,
                dropout=kwargs.get('dropout', 0.3),
                conv_dim=kwargs.get('conv_dim', 32),
                dsc_dim=kwargs.get('dsc_dim', 64),
            )
        else:
            raise ValueError(f"Unknown ablation model: {model_name}")
