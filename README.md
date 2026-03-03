# Satellite Network Threat Detection System
## 基于深度学习与联邦学习的卫星网络威胁检测系统

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 项目概述

本项目针对卫星网络环境下的威胁检测问题，设计并实现了轻量级深度学习模型（DSC-CBAM-LSTM），同时支持集中式训练和联邦学习两种部署模式。

### 核心特性

- **轻量级模型设计**：参数量 < 100K，满足卫星节点的计算和存储限制
- **DSC-CBAM-LSTM 架构**：融合深度可分离卷积、注意力机制和LSTM时序建模
- **联邦学习支持**：基于 FedAvg/FedProx 算法的多卫星协同训练
- **高性能推理**：推理时间 < 10ms，满足实时检测需求

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Satellite Threat Detection                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Satellite  │      │   Satellite  │      │   Satellite  │  │
│  │     Node 1   │      │     Node 2   │      │     Node N   │  │
│  │  (Client)    │      │  (Client)    │      │  (Client)    │  │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘  │
│         │                     │                     │          │
│         └─────────────────────┼─────────────────────┘          │
│                               │                                  │
│                    ┌──────────▼──────────┐                      │
│                    │   Server (Aggregator) │                     │
│                    │   - FedAvg          │                     │
│                    │   - FedProx         │                     │
│                    │   - Model Update    │                     │
│                    └─────────────────────┘                      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Model: DSC-CBAM-LSTM                     │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │ │
│  │  │   DSC   │→ │  CBAM   │→ │  LSTM   │→ │ Classifier  │  │ │
│  │  │ (Conv)  │  │(Attention│  │(Sequence)│  │  (FC Layer) │  │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 技术规格

| 指标 | 要求 | 实际达成 |
|------|------|----------|
| 参数量 | < 100K | 25,728 ~ 36,320 |
| 推理时间 | < 10ms | 0.04 ms |
| 测试准确率 (集中式) | > 85% | 85.74% ~ 86.11% |
| 测试准确率 (联邦学习) | > 80% | 80.26% |

## 项目结构

```
Satellite_Threat_Detection/
├── config.py                 # 配置管理
├── train_model.py            # 集中式训练入口 (V1)
├── train_model_v2.py         # 集中式训练入口 (V2)
├── fedavg_train.py           # 联邦学习训练入口
├── evaluate_model.py         # 模型评估 (V1)
├── evaluate_model_v2.py      # 模型评估 (V2)
├── requirements.txt          # 依赖包
│
├── models/                   # 模型定义
│   ├── __init__.py
│   ├── dsc_cbam_lstm.py      # DSC-CBAM-LSTM V1 模型
│   ├── dsc_cbam_lstm_v2.py   # DSC-CBAM-LSTM V2 模型 (增强版)
│   ├── lightweight_cnn_lstm.py
│   └── lightweight_model.py
│
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── data_processor.py    # 数据预处理
│   └── data_loader.py       # 数据加载
│
├── core/                    # 核心训练模块
│   ├── __init__.py
│   ├── trainer.py          # 训练器基类
│   ├── losses.py           # 损失函数
│   ├── metrics.py          # 评估指标
│   └── logger.py           # 日志记录
│
├── federated/              # 联邦学习模块
│   ├── __init__.py
│   ├── server.py          # 服务端
│   ├── client.py          # 客户端
│   ├── aggregation.py     # 参数聚合
│   └── strategies.py      # 聚合策略
│
└── dataset/               # 数据集 (需要自行准备)
    ├── train_dataset_final.csv
    ├── val_dataset_final.csv
    └── test_dataset_final.csv
```

## 快速开始

### 环境配置

```bash
# 创建虚拟环境
conda create -n satellite_threat python=3.10
conda activate satellite_threat

# 安装依赖
pip install -r requirements.txt

# 安装 PyTorch (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 数据准备

将预处理后的数据集放入 `dataset/` 目录：

```
dataset/
├── train_dataset_final.csv
├── val_dataset_final.csv
└── test_dataset_final.csv
```

### 集中式训练 (V1 - 基础版)

```bash
python train_model.py
```

### 集中式训练 (V2 - 增强版)

```bash
python train_model_v2.py
```

### 联邦学习训练

```bash
python fedavg_train.py
```

### 模型评估

```bash
# 评估V1模型
python evaluate_model.py

# 评估V2模型
python evaluate_model_v2.py
```

## 模型架构详解

### DSC-CBAM-LSTM (V1)

基础版本，采用单向LSTM进行时序建模。

### DSC-CBAM-LSTM V2 (增强版)

优化版本，采用双向LSTM和残差连接：

```
Input: (Batch, 10, 8)

├── Input Projection: Linear(8→16)

├── DSC Block 1 + CBAM + Pooling
│
├── Residual Block (残差连接)

├── DSC Block 2 + CBAM

├── Bidirectional LSTM: LSTM(64→32×2)

└── Classifier
    └── Linear(32×2 + 64 → 4)
```

### DSC-CBAM-LSTM (V1)

```
Input: (Batch, 10, 8)  - 10个时间步，8维特征

├── Input Projection: Linear(8→16)
│
├── DSC Block 1: Depthwise Separable Conv + CBAM
│   ├── DepthwiseConv1D(16, 32, kernel=3)
│   ├── CBAM(32)
│   ├── DepthwiseConv1D(32, 32, kernel=3)
│   └── MaxPool1d(2) → (B, 32, 5)
│
├── DSC Block 2
│   ├── DepthwiseConv1D(32, 64, kernel=3)
│   ├── CBAM(64)
│   └── DepthwiseConv1D(64, 64, kernel=3) → (B, 64, 5)
│
├── Global Pooling: AdaptiveAvgPool1d(1) → (B, 64)
│
├── LSTM: LSTM(64→32) → (B, 32)
│
└── Classifier
    └── Linear(32+64 → 4) → (B, 4)
```

### 核心技术

1. **深度可分离卷积 (DSC)**：减少参数量和计算量
2. **CBAM 注意力**：通道注意力 + 空间注意力
3. **LSTM 时序建模**：捕获网络流量的时序特征

## 联邦学习详解

### FedAvg 算法

```
Server:
1. 初始化全局模型参数 W₀
2. for round t = 1, 2, ...:
3.     将 Wₜ 分发给所有客户端
4.     接收各客户端更新 Wₜ⁽ᵏ⁾
5.     Wₜ₊₁ = Σ(nₖ/n)Wₜ⁽ᵏ⁾  // 加权平均
6. return W_T
```

### FedProx 正则化

在本地损失函数中加入近端项：

```
L_prox = L_task + (μ/2) ||W - W_global||²
```

- 限制本地模型偏离全局模型
- 提高聚合稳定性

## 实验结果

### 集中式训练

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 10    | 0.6523    | 72.23%    | 0.5891   | 74.56%  |
| 20    | 0.3215    | 85.41%    | 0.3124   | 87.23%  |
| 40    | 0.1892    | 91.56%    | 0.2034   | 89.45%  |
| 60    | 0.1423    | 93.78%    | 0.1789   | 90.12%  |
| 80    | 0.1023    | 95.23%    | 0.1523   | 91.45%  |

### 联邦学习

| Round | Val Acc | Client Avg Acc |
|-------|---------|----------------|
| 10    | 74.91%  | 78.71%         |
| 30    | 77.22%  | 86.23%         |
| 50    | 77.53%  | 87.49%         |
| 80    | 77.65%  | 87.57%         |
| 100   | 77.89%  | 87.70%         |

## 性能指标

### V1 模型 (基础版)

```
============================================================
Model: DSC-CBAM-LSTM V1
Parameters: 25,728 (< 100K: ✓)
Test Accuracy: 85.74%
Inference Time: 0.148 ms (< 10ms: ✓)
============================================================
```

### V2 模型 (增强版)

```
============================================================
Model: DSC-CBAM-LSTM V2
Parameters: 36,320 (< 100K: ✓)
Test Accuracy: 86.11%
Inference Time: 0.037 ms (< 10ms: ✓)
============================================================

优化点:
- 双向LSTM增强时序特征提取
- 残差连接改善梯度流动
- 增强型CBAM注意力机制
```

### 联邦学习

```
Federated Learning:
Clients: 5
Local Epochs: 5
Communication Rounds: 100
FedProx: True, μ=0.001
Final Test Accuracy: 80.26%
```

## 配置说明

### 训练配置 (config.py)

```python
# 模型配置
model = ModelConfig(
    model_name='DSC-CBAM-LSTM',
    seq_len=10,
    input_dim=8,
    hidden_dim=32,        # V2可设为48
    num_classes=4,
    dropout=0.2
)

# 训练配置
train = TrainConfig(
    batch_size=128,
    epochs=80,
    learning_rate=0.002,   # 推荐使用0.002
    weight_decay=1e-5,
    patience=15,
    use_class_weight=True,
    loss_type='label_smoothing',  # focal, cross_entropy, label_smoothing
    label_smoothing=0.1
)

# 联邦学习配置
federated = FederatedConfig(
    num_clients=5,
    local_epochs=5,
    num_rounds=100,
    use_fedprox=True,
    mu=0.001
)
```

## 依赖版本

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tensorboard>=2.13.0
tqdm>=4.65.0
```

## 许可证

MIT License

## 参考文献

1. Li, T., et al. "FedProx: Federated Optimization with Proximal Terms"
2. Howard, A., et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
3. Woo, S., et al. "CBAM: Convolutional Block Attention Module"
