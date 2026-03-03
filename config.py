"""
配置管理模块 - 统一管理所有超参数
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "DSC-CBAM-LSTM"
    seq_len: int = 10
    input_dim: int = 8
    hidden_dim: int = 32
    num_classes: int = 4
    dropout: float = 0.2


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "04_datasets"
    feature_cols: List[str] = field(default_factory=lambda: [
        'IAT_Mean_norm', 'IAT_Std_norm', 'Duration_norm',
        'PktLen_Mean_norm', 'PktLen_Std_norm',
        'Packets_s_norm', 'Bytes_s_norm', 'Idle_Mean_norm'
    ])
    scaler_type: str = "standard"
    seq_len: int = 10
    stride: int = 1
    use_augment: bool = False


@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 128
    epochs: int = 80
    learning_rate: float = 0.002
    weight_decay: float = 1e-5
    patience: int = 15
    gradient_clip: float = 1.0
    use_early_stopping: bool = True
    use_lr_scheduler: bool = True
    use_class_weight: bool = True
    loss_type: str = "label_smoothing"  # 损失函数类型: cross_entropy, focal, weighted, label_smoothing
    label_smoothing: float = 0.1  # 标签平滑因子


@dataclass
class Config:
    """全局配置"""
    # 模型
    model = ModelConfig()
    
    # 数据
    data = DataConfig()
    
    # 训练
    train = TrainConfig()
    
    # 系统
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    @property
    def device_obj(self):
        import torch
        return torch.device(self.device if torch.cuda.is_available() else "cpu")


# 全局配置实例
config = Config()


def get_config() -> Config:
    """获取配置实例"""
    return config


def update_config(**kwargs):
    """更新配置"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if hasattr(getattr(config, key), sub_key):
                        setattr(getattr(config, key), sub_key, sub_value)
            else:
                setattr(config, key, value)


if __name__ == "__main__":
    c = get_config()
    print("Model Config:")
    print(f"  model_name: {c.model.model_name}")
    print(f"  seq_len: {c.model.seq_len}")
    print(f"  hidden_dim: {c.model.hidden_dim}")
    print(f"  num_classes: {c.model.num_classes}")
    print("\nData Config:")
    print(f"  data_dir: {c.data.data_dir}")
    print(f"  feature_cols: {c.data.feature_cols}")
    print("\nTrain Config:")
    print(f"  batch_size: {c.train.batch_size}")
    print(f"  epochs: {c.train.epochs}")
    print(f"  learning_rate: {c.train.learning_rate}")
    print(f"  patience: {c.train.patience}")
