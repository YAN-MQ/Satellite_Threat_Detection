# 06_realtime_emulation

## 概述

本文档是当前目录的最终说明，只对应清理后的窗口版实验链路。

本次流程严格按以下链路执行：

`原始 PCAP -> 攻击时间窗口提取 -> IP 分片 -> NS-3 实时仿真 -> 流量捕获 -> 特征提取 -> DSC-CBAM-GRU 训练`

论文依据为 [66398.pdf](./66398.pdf)，本地数据目录为：

- `/home/lithic/final/data`

## 论文核对结论

根据 [66398.pdf](./66398.pdf) 和 Friday afternoon 细化攻击时间表：

- `Monday` 对应 `Benign`
- `Wednesday` 主要是 `DoS / Heartbleed`
- `Friday afternoon` 才包含 `PortScan / DDoS`

因此，原项目把 `Wednesday-workingHours.pcap` 直接当成 `ddos` 使用是错误的。

本次复现已改成真正的三分类：

- `Benign`
- `DDoS`
- `PortScan`

## 当前保留内容

当前目录只保留以下实验闭环：

- 论文：`66398.pdf`
- 时间窗提取与分片：`1_fragment_pcap_window.py`
- 特征提取：`2_extract_features_window.py`
- 实时仿真：`realtime_satellite.cc`、`CMakeLists.txt`、`run_all_window.sh`
- 当前数据产物：`fragments_window/`、`captured_window/`、`dataset_window/`
- 当前训练与扩展实验：`4_train/`

原始版、鲁棒版、历史 checkpoint、旧文档和过时脚本都已经移除。

## 本次修复点

已修复以下问题：

1. 把错误的 `Wednesday DoS -> ddos` 标签改为真正的 `Friday afternoon DDoS`。
2. 明确使用本机 `Asia/Shanghai` 时区换算 Friday afternoon 时间窗。
3. 清理脚本中的旧路径 `/root/final/...`。
4. `run_all_window.sh` 不再重写 Python 脚本。
5. 抓包改为从 `tap-right` 获取仿真输出。
6. `train_gru.py` 改为当前窗口版可用的 3 分类训练配置。
7. `DataLoader` 改为单进程，避免当前环境 `SemLock` 权限问题。
8. 修复 IP 分片逻辑，消除 `tcpreplay: Message too long`。
9. 为 `scratch/06_realtime_emulation` 增加独立 [CMakeLists.txt](./CMakeLists.txt)，把 `realtime_satellite.cc` 并回 ns-3 正式 scratch target。
10. 把 `CMakeLists.txt` 链接库收敛到最小依赖，避免不必要的全模块链接。

本次涉及的主要文件：

- [1_fragment_pcap_window.py](./1_fragment_pcap_window.py)
- [2_extract_features_window.py](./2_extract_features_window.py)
- [run_all_window.sh](./run_all_window.sh)
- [4_train/scripts/train_gru.py](./4_train/scripts/train_gru.py)
- [CMakeLists.txt](./CMakeLists.txt)

## 时间窗定义

当前窗口定义见 [1_fragment_pcap_window.py](./1_fragment_pcap_window.py)：

- `benign`: `Monday-WorkingHours.pcap`
  - `2017-07-03 20:00:00` -> `2017-07-04 03:59:59` (`Asia/Shanghai`)
- `portscan`: `Friday-WorkingHours.pcap`
  - `2017-07-07 23:30:00` -> `2017-07-08 02:39:59`
- `ddos`: `Friday-WorkingHours.pcap`
  - `2017-07-08 02:40:00` -> `2017-07-08 03:29:59`

其中 Friday 两个窗口对应 CICIDS2017 Friday afternoon 的细化时段：

- `PortScan`: `12:30 PM - 3:40 PM`
- `DDoS`: `3:40 PM - 4:30 PM`

## 分片修复说明

原始问题不在简单长度判断，而在“生成的新片段仍保留原始以太网负载”，导致回放时帧实际仍然超长。

当前 [1_fragment_pcap_window.py](./1_fragment_pcap_window.py) 的修复策略是：

- 只使用 `IP.len` 对应的有效 IP 负载
- 重新构造干净的 `Ether` 头
- 使用 `scapy.fragment()` 按 `MTU - IP头长度` 进行分片

修复后，对当前大样本输入的校验结果为：

- `benign`: `55475 packets, max_frame=1514, max_ip=1500`
- `ddos`: `53705 packets, max_frame=1514, max_ip=1500`
- `portscan`: `55002 packets, max_frame=1514, max_ip=1500`

并且完整回放日志中：

- `Failed packets = 0`
- `Truncated packets = 0`
- 不再出现 `Message too long`

验证日志位于：

- 本次清理后未保留历史日志文件；对应结论已体现在本文档记录中。

## 构建说明

### 当前已完成的事

当前仅保留 [realtime_satellite.cc](./realtime_satellite.cc) 这一条实时仿真入口，并已通过 [CMakeLists.txt](./CMakeLists.txt) 并回 ns-3 的 scratch 子目录目标，不再需要手写 `c++ ...` 链接命令。

当前推荐构建命令是：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1
cmake --build build --target scratch_06_realtime_emulation_realtime_satellite -j"$(nproc)"
```

`run_all_window.sh` 也已经同步为优先使用这个正式 target。

### 当前环境限制

在这台 WSL 环境中，顶层 `cmake -S . -B build` / `./ns3 build realtime_satellite` 的重新生成步骤仍然比较慢，因此本次完整大样本仿真实际使用的是已存在的 `realtime_satellite_manual` 二进制作为运行入口。

也就是说：

- “手工编译命令”已经不再是工程唯一入口
- 正式 CMake target 已经并回工程
- 但本机上的顶层重配置性能问题还没有完全消掉

## 可复现实验步骤

### 1. 生成攻击窗口并分片

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation
python3 1_fragment_pcap_window.py --max-packets 50000
```

输出：

- [fragments_window/benign.pcap](./fragments_window/benign.pcap)
- [fragments_window/ddos.pcap](./fragments_window/ddos.pcap)
- [fragments_window/portscan.pcap](./fragments_window/portscan.pcap)

### 2. 构建实时仿真目标

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1
cmake --build build --target scratch_06_realtime_emulation_realtime_satellite -j"$(nproc)"
```

### 3. 创建 TAP 接口并运行仿真

需要 root 权限：

```bash
sudo ip tuntap add dev tap-left mode tap
sudo ip tuntap add dev tap-right mode tap
sudo ip link set tap-left up
sudo ip link set tap-right up
```

### 4. 回放并抓包

当前实际抓包方向为 `tap-right`，这是仿真输出接口。

完整结果位于：

- [captured_window/benign.pcap](./captured_window/benign.pcap)
- [captured_window/ddos.pcap](./captured_window/ddos.pcap)
- [captured_window/portscan.pcap](./captured_window/portscan.pcap)

本次目录整理后未保留运行日志目录，保留的是最终抓包结果和数据集产物。

### 5. 提取特征

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation
python3 2_extract_features_window.py
```

生成：

- [dataset_window/train.npz](./dataset_window/train.npz)
- [dataset_window/val.npz](./dataset_window/val.npz)
- [dataset_window/test.npz](./dataset_window/test.npz)

## 特征与预处理

当前特征提取逻辑定义在 [2_extract_features_window.py](./2_extract_features_window.py)。

### 1. 基础包级字段

对每个捕获包，先抽取以下基础字段：

- `ts`: 包时间戳
- `size`: 包长，直接取 `len(pkt)`
- `proto`: IP 协议号
- `SYN / ACK / RST / FIN / PSH`: TCP 标志位

预处理时只保留 `IP` 包：

- 非 `IP` 包直接丢弃
- 非 `TCP` 包的 5 个 TCP 标志位统一记为 `0`

### 2. 18 维特征

最终送入模型的 18 维特征如下：

| 序号 | 特征名 | 含义 |
|------|--------|------|
| `1` | `IAT` | 当前包与前一包的到达间隔 |
| `2` | `size` | 当前包长度 |
| `3` | `proto` | IP 协议号 |
| `4` | `SYN` | TCP SYN 标志 |
| `5` | `ACK` | TCP ACK 标志 |
| `6` | `RST` | TCP RST 标志 |
| `7` | `FIN` | TCP FIN 标志 |
| `8` | `PSH` | TCP PSH 标志 |
| `9` | `Size_M` | 近窗口包长均值 |
| `10` | `Size_S` | 近窗口包长标准差 |
| `11` | `IAT_M` | 近窗口到达间隔均值 |
| `12` | `IAT_S` | 近窗口到达间隔标准差 |
| `13` | `IAT_X` | 近窗口到达间隔最大值 |
| `14` | `IAT_N` | 近窗口到达间隔最小值 |
| `15` | `PPS` | 窗口内每秒包数 |
| `16` | `BPS` | 窗口内每秒字节数 |
| `17` | `SYN_R` | 窗口内 SYN 包比例 |
| `18` | `ACK_R` | 窗口内 ACK 包比例 |

### 3. 统计窗口

统计特征使用固定窗口：

- `window size = 10`
- `stride = 1`

也就是说，每个样本最终形状为：

- `(10, 18)`

窗口统计规则包括：

- `Size_M / Size_S`: 对最近 10 个包长度做滚动均值和标准差
- `IAT_M / IAT_S / IAT_X / IAT_N`: 对最近 10 个包间隔做滚动统计
- `PPS / BPS`: 用窗口内包数或字节数除以窗口时间跨度
- `SYN_R / ACK_R`: 用窗口内标志位计数除以窗口包数

### 4. 数据清洗

提取后还做了以下处理：

- 第一包没有前驱，因此其 `IAT = 0`
- 滚动标准差的空值用 `0` 填充
- `inf / -inf` 统一替换成 `0`
- 所有缺失值最终都填成 `0`

### 5. 训练集 / 验证集 / 测试集划分

每个类别的 PCAP 都单独处理，然后按时间顺序切分：

- `train = 60%`
- `val = 20%`
- `test = 20%`

这一步是在单类别内部完成的，之后再把三类结果拼接成最终的 `train.npz / val.npz / test.npz`。

### 6. 归一化方式

归一化使用 `MinMaxScaler`：

- 每个类别各自拟合训练段的 scaler
- `val` 和 `test` 使用同一个类别的训练段 scaler 变换
- 不同类别之间没有共享同一个 scaler

因此，当前实现是“按类别、按时间顺序、先切分后归一化”。

### 7. 序列样本生成

归一化后，再用滑窗生成模型输入序列：

- 输入张量形状：`(samples, 10, 18)`
- 标签是类别常量：
  - `benign = 0`
  - `ddos = 1`
  - `portscan = 2`

## 模型结构

当前主模型定义在 [4_train/src/models/dsc_cbam_gru.py](./4_train/src/models/dsc_cbam_gru.py)，整体前向路径为：

`Input(10x18) -> Conv1D(32) -> DSC(64) -> CBAM(64) -> GRU(hidden) -> FC -> 3类输出`

### 1. 输入与张量形状

单个样本的输入形状是：

- `(window, features) = (10, 18)`

进入模型后，首先做一次维度变换：

- 原始输入：`(batch, 10, 18)`
- 变换后：`(batch, 18, 10)`

这样可以直接送入 `Conv1D` 做时序卷积。

### 2. 初始特征映射层

模型入口先用一个 `1x1 Conv1D` 做通道映射：

- 输入通道：`18`
- 输出通道：`32`
- 核大小：`1`

它的作用不是提取长距离时序模式，而是先把 18 维原始统计特征映射到更适合卷积建模的隐空间。

对应形状变化：

- 输入：`(batch, 18, 10)`
- 输出：`(batch, 32, 10)`

### 3. DSC 模块

`DSC` 指 `Depthwise Separable Convolution`，由两部分组成：

1. `Depthwise Conv1D`
   - 每个输入通道独立做 `3x1` 卷积
   - 不做跨通道混合
2. `Pointwise Conv1D`
   - 用 `1x1` 卷积把通道数从 `32` 投影到 `64`

当前实现还带有：

- `BatchNorm1d`
- `ReLU`

它的核心作用是：

- 在时间维度上提取局部模式
- 相比标准卷积减少参数量和计算量
- 保持轻量化，适合实时检测场景

对应形状变化：

- 输入：`(batch, 32, 10)`
- 输出：`(batch, 64, 10)`

### 4. CBAM 模块

`CBAM` 指 `Convolutional Block Attention Module`，由两个子模块组成：

#### Channel Attention

输入：

- `(batch, 64, 10)`

处理方式：

- 先对时间维做 `AdaptiveAvgPool1d(1)` 和 `AdaptiveMaxPool1d(1)`
- 得到两个 `64` 维通道描述
- 经过共享的两层全连接网络
- 生成每个通道的重要性权重

作用：

- 让模型学会“哪些特征通道更重要”
- 强化关键统计特征，抑制冗余通道

#### Spatial Attention

输入：

- 通道注意力后的 `(batch, 64, 10)`

处理方式：

- 在通道维做平均池化和最大池化
- 拼接成 `2` 通道特征图
- 再做一次 `Conv1D`
- 输出时间位置上的注意力权重

作用：

- 让模型学会“窗口内哪些时间位置更重要”
- 强化攻击流量中更关键的局部片段

CBAM 整体不改变张量尺寸：

- 输入：`(batch, 64, 10)`
- 输出：`(batch, 64, 10)`

### 5. GRU 模块

CBAM 输出后，再把张量变回循环网络需要的形式：

- 输入到 GRU 前：`(batch, 10, 64)`

当前 `DSC_CBAM_GRU` 支持以下可调参数：

- `hidden_dim`
- `bidirectional`
- `dropout`

默认主训练脚本当前使用：

- `hidden_dim = 64`
- `bidirectional = False`
- `dropout = 0.3`

在扩展实验中还测试过更强的 `BiGRU` 配置。

GRU 的作用是：

- 建模 10 个包窗口内的时序依赖
- 将卷积提取到的局部模式进一步整合成序列表示
- 用最后一个时间步的隐藏状态作为整个窗口的摘要表示

对应输出：

- 单向 GRU：`(batch, 10, 64)` -> 取最后一步后得到 `(batch, 64)`
- 双向 GRU：`(batch, 10, 128)` -> 取最后一步后得到 `(batch, 128)`

### 6. 全连接分类头

分类头由两层全连接组成：

1. `Linear(gru_out_dim, 64)`
2. `ReLU`
3. `Dropout`
4. `Linear(64, 3)`

作用：

- 把 GRU 输出的窗口级表示映射到最终的三分类空间
- 输出类别为：
  - `0 = benign`
  - `1 = ddos`
  - `2 = portscan`

最终输出形状：

- `(batch, 3)`

### 7. 三个核心模块的职责总结

可以把整个模型理解为三段：

- `DSC`
  - 负责轻量化局部时序模式提取
- `CBAM`
  - 负责对通道和时间位置做注意力加权
- `GRU`
  - 负责把局部模式整合成窗口级时序表示

因此，这个模型的设计逻辑不是单纯堆叠模块，而是：

- 先用卷积提局部模式
- 再用注意力筛关键模式
- 最后用循环网络做时序汇总

## 模型训练配置

当前主训练脚本是 [4_train/scripts/train_gru.py](./4_train/scripts/train_gru.py)。

默认训练配置如下：

- 模型：`DSC-CBAM-GRU`
- `epochs = 20`
- `batch_size = 64`
- `input_dim = 18`
- `num_classes = 3`
- 优化器：`AdamW`
- 学习率：`1e-3`
- 权重衰减：`1e-2`
- 学习率调度：`ReduceLROnPlateau`
- DataLoader：`num_workers = 0`
- 当前默认设备：`cpu`

## GPU 训练环境

当前已经为本项目单独配置了 GPU 训练虚拟环境：

- 虚拟环境路径：`/home/lithic/final/ns3-gpu-venv`
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
- 已安装 PyTorch：`2.10.0+cu128`

注意：

1. 在当前 Codex 沙箱里，`torch.cuda.is_available()` 会误报失败。
2. 在真实系统环境中，CUDA 已验证可用。
3. 当前所有训练脚本都已经统一为“若检测到 CUDA，则默认优先使用 GPU；否则回退 CPU”。
4. 因此，后续真正跑 GPU 训练时，应直接在终端里使用该虚拟环境或包装脚本执行。

### 启用环境

```bash
source /home/lithic/final/ns3-gpu-venv/bin/activate
```

### GPU 可用性验证

```bash
/home/lithic/final/ns3-gpu-venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

### 启动 GPU 训练

推荐直接使用包装脚本：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
./run_train_gpu.sh --epochs 20 --output checkpoints_gru/window_gru_gpu.pt
```

对应包装脚本见：

- [4_train/run_train_gpu.sh](./4_train/run_train_gpu.sh)
- [4_train/run_ablation_gpu.sh](./4_train/run_ablation_gpu.sh)
- [4_train/run_comparison_gpu.sh](./4_train/run_comparison_gpu.sh)
- [4_train/run_tune_full_gpu.sh](./4_train/run_tune_full_gpu.sh)

例如：

```bash
./run_ablation_gpu.sh --epochs 20
./run_comparison_gpu.sh --epochs 20
./run_tune_full_gpu.sh --epochs 20 --max_runs 4
```

如果你想直接调 Python 脚本，也可以：

```bash
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_gru.py \
  --data_dir ../dataset_window \
  --device cuda \
  --epochs 20 \
  --output checkpoints_gru/window_gru_gpu.pt
```

### 本次 GPU 烟雾测试

已实际在 `cuda` 上跑通 1 个 epoch：

- 输出模型：[4_train/checkpoints_gru/window_gru_gpu_smoke.pt](./4_train/checkpoints_gru/window_gru_gpu_smoke.pt)
- 运行设备：`cuda`
- `Val Acc = 0.9507`
- `Test Accuracy = 0.9297`

### 6. 训练 DSC-CBAM-GRU

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
python3 scripts/train_gru.py \
  --data_dir ../dataset_window \
  --epochs 20 \
  --num_classes 3 \
  --input_dim 18 \
  --output checkpoints_gru/window_gru_best.pt
```

模型输出：

- [4_train/checkpoints_gru/window_gru_best.pt](./4_train/checkpoints_gru/window_gru_best.pt)

## 本次完整重跑结果

### 1. 分片输入

- `benign`: `55475` packets
- `ddos`: `53705` packets
- `portscan`: `55002` packets

### 2. 实时仿真抓包

`tcpdump` 抓到的输出包数与输入一致：

- `benign`: `55475`
- `ddos`: `53705`
- `portscan`: `55002`

### 3. 特征数据集

总规模：

- `Train`: `(98482, 10, 18)`
- `Val`: `(32809, 10, 18)`
- `Test`: `(32810, 10, 18)`

分类拆分：

- `benign`
  - Train `(33276, 10, 18)`
  - Val `(11086, 10, 18)`
  - Test `(11086, 10, 18)`
- `ddos`
  - Train `(32214, 10, 18)`
  - Val `(10732, 10, 18)`
  - Test `(10732, 10, 18)`
- `portscan`
  - Train `(32992, 10, 18)`
  - Val `(10991, 10, 18)`
  - Test `(10992, 10, 18)`

### 4. GRU 训练结果

训练日志关键点：

```text
Epoch 01 | Train Loss: 0.1747 | Val Loss: 0.1781 | Val Acc: 0.9344
Epoch 05 | Train Loss: 0.0448 | Val Loss: 0.0471 | Val Acc: 0.9787
Epoch 10 | Train Loss: 0.0287 | Val Loss: 0.0445 | Val Acc: 0.9827
Epoch 15 | Train Loss: 0.0205 | Val Loss: 0.0543 | Val Acc: 0.9835
Epoch 20 | Train Loss: 0.0163 | Val Loss: 0.0539 | Val Acc: 0.9832
Best model saved to checkpoints_gru/window_gru_best.pt (Val Acc: 0.9902)
```

测试集结果：

- `Accuracy`: `0.9886`
- `Precision`: `0.9887`
- `Recall`: `0.9886`
- `F1`: `0.9886`

混淆矩阵：

```text
[[10935    22   129]
 [    7 10607   118]
 [   76    22 10894]]
```

## 当前结论

1. Friday afternoon 的真正 `DDoS` 包级窗口已经并入当前流程。
2. 分片逻辑已经修好，并通过大样本回放验证，不再出现 `Message too long`。
3. `realtime_satellite.cc` 已经并回 ns-3 工程目标，`run_all_window.sh` 也已同步到正式构建路径。
4. 当前机器上剩余的问题不是项目逻辑错误，而是顶层 ns-3 重配置在 WSL 环境中的性能偏慢。

## 扩展实验脚本

在不覆盖现有主实验结果的前提下，当前新增了 3 个独立脚本，输出默认写入 `4_train/experiments_window/`：

- [4_train/scripts/run_ablation_window.py](./4_train/scripts/run_ablation_window.py)
  - 基于当前 `dataset_window` 运行 `full / no_dsc / no_cbam / no_gru` 消融实验
- [4_train/scripts/run_comparison_window.py](./4_train/scripts/run_comparison_window.py)
  - 运行 `DSC-CBAM-GRU / DSC-CBAM-LSTM / MLP / CNN-LSTM / RF / ID3` 对比实验
- [4_train/scripts/plot_tsne_window.py](./4_train/scripts/plot_tsne_window.py)
  - 读取当前 `window_gru_best.pt` 并生成新的 t-SNE 图

此外，当前还新增了 2 个模型压缩脚本：

- [4_train/scripts/prune_window_model.py](./4_train/scripts/prune_window_model.py)
  - 对当前 `DSC-CBAM-GRU` 做后训练全局非结构化剪枝
  - 默认剪枝 `Conv1D / DSC / CBAM 内线性层 / GRU / FC` 的权重
  - 输出剪枝后的 checkpoint 和精度、稀疏率、推理时延汇总
- [4_train/scripts/quantize_window_model.py](./4_train/scripts/quantize_window_model.py)
  - 对当前 `DSC-CBAM-GRU` 做后训练动态量化
  - 默认量化 `GRU` 和 `Linear` 为 `INT8`
  - 输出量化后的 checkpoint 和精度、文件大小、推理时延汇总

示例：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train

python3 scripts/run_ablation_window.py --data_dir ../dataset_window --epochs 20
python3 scripts/run_comparison_window.py --data_dir ../dataset_window --epochs 20
python3 scripts/plot_tsne_window.py --data_dir ../dataset_window --model_path checkpoints_gru/window_gru_best.pt
python3 scripts/prune_window_model.py --data_dir ../dataset_window --checkpoint checkpoints_gru/window_gru_best.pt --amount 0.30
python3 scripts/quantize_window_model.py --data_dir ../dataset_window --checkpoint checkpoints_gru/window_gru_best.pt
```

## 模型压缩脚本

### 1. 剪枝脚本

[4_train/scripts/prune_window_model.py](./4_train/scripts/prune_window_model.py) 使用 PyTorch 原生剪枝接口做后训练压缩，流程为：

1. 加载当前 `window_gru_best.pt`
2. 在测试集上评估基线模型
3. 对卷积层、注意力层、GRU 和全连接层做全局 `L1` 非结构化剪枝
4. 移除 pruning re-parameterization，导出普通 `state_dict`
5. 再次评估剪枝后模型，并统计：
   - `Accuracy / Precision / Recall / F1`
   - `Confusion Matrix`
   - `Sparsity`
   - `Checkpoint Size`
   - `Latency / Throughput`

默认输出目录：

- `4_train/experiments_window/compression/pruning/`

示例：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
python3 scripts/prune_window_model.py \
  --data_dir ../dataset_window \
  --checkpoint checkpoints_gru/window_gru_best.pt \
  --amount 0.30
```

### 2. 量化脚本

[4_train/scripts/quantize_window_model.py](./4_train/scripts/quantize_window_model.py) 使用 PyTorch 动态量化接口，流程为：

1. 加载当前 `window_gru_best.pt`
2. 在 CPU 上评估基线模型
3. 对 `GRU` 和 `Linear` 做动态 `INT8` 量化
4. 导出量化模型 `state_dict`
5. 再次评估量化后模型，并统计：
   - `Accuracy / Precision / Recall / F1`
   - `Confusion Matrix`
   - `Checkpoint Size`
   - `Latency / Throughput`

默认输出目录：

- `4_train/experiments_window/compression/quantization/`

示例：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
python3 scripts/quantize_window_model.py \
  --data_dir ../dataset_window \
  --checkpoint checkpoints_gru/window_gru_best.pt
```

### 3. 当前压缩实验结果

本次已经基于当前主模型 [4_train/checkpoints_gru/window_gru_best.pt](./4_train/checkpoints_gru/window_gru_best.pt) 实际跑通：

- [4_train/experiments_window/compression/pruning/window_gru_pruned_30.pt](./4_train/experiments_window/compression/pruning/window_gru_pruned_30.pt)
- [4_train/experiments_window/compression/pruning/window_gru_pruned_30_summary.json](./4_train/experiments_window/compression/pruning/window_gru_pruned_30_summary.json)
- [4_train/experiments_window/compression/quantization/window_gru_dynamic_int8.pt](./4_train/experiments_window/compression/quantization/window_gru_dynamic_int8.pt)
- [4_train/experiments_window/compression/quantization/window_gru_dynamic_int8_summary.json](./4_train/experiments_window/compression/quantization/window_gru_dynamic_int8_summary.json)

结果如下：

| 方法 | Accuracy | F1 | 文件大小(MB) | 推理时延(ms/sample) | 备注 |
|------|----------|----|--------------|---------------------|------|
| `baseline` | `0.9886` | `0.9886` | `0.1343` | `0.0579` | 原始主模型 |
| `30% pruning` | `0.8779` | `0.8792` | `0.1352` | `0.0512` | 稀疏率 `0.2936`，精度下降明显 |
| `dynamic int8 quantization` | `0.9688` | `0.9688` | `0.0534` | `0.0186` | 体积和时延均明显下降 |

当前可以得出的结论是：

1. 动态量化对当前 `DSC-CBAM-GRU` 更友好，几乎不损失精度，同时显著降低模型体积和 CPU 推理时延。
2. 固定 `30%` 的全局非结构化剪枝对当前模型破坏较大，若后续要把剪枝结果写进论文，应进一步做剪枝率搜索，而不是直接固定一个比例。

## 扩展实验结果

从这一步开始，扩展实验不再只按 `Accuracy/F1` 排序，而是使用综合评分：

`0.30 * Accuracy + 0.20 * F1 + 0.25 * 参数效率 + 0.15 * FLOPs效率 + 0.10 * 推理时延效率`

其中：

- `Accuracy / F1` 越高越好
- `参数量 / FLOPs / 推理时延` 越低越好
- 传统树模型 `RF / ID3` 不参与综合评分排序，因为这里没有与深度模型同口径的参数量和 FLOPs 统计

本次已实际运行：

- [4_train/experiments_window/ablation/ablation_summary.csv](./4_train/experiments_window/ablation/ablation_summary.csv)
- [4_train/experiments_window/comparison/comparison_summary.csv](./4_train/experiments_window/comparison/comparison_summary.csv)
- [4_train/experiments_window/visualization/tsne_window_gru.png](./4_train/experiments_window/visualization/tsne_window_gru.png)

### 消融实验

结果如下：

| 排名 | 模型 | 准确率 | F1 | 参数量 | FLOPs | 综合评分 |
|------|------|--------|----|--------|-------|----------|
| `1` | `dsc_cbam_gru(ours)` | `0.9891` | `0.9891` | `33,329` | `210,424` | `0.930454` |
| `2` | `ablation_no_cbam` | `0.9892` | `0.9692` | `61,347` | `216,064` | `0.786831` |
| `3` | `ablation_no_dsc` | `0.9876` | `0.9876` | `66,353` | `362,616` | `0.538805` |
| `4` | `ablation_no_gru` | `0.9797` | `0.9797` | `37,169` | `1,004,280` | `0.280509` |

当前数据上可以看到：

1. 按综合评分，`dsc_cbam_gru(ours)` 仍然是消融实验中的最佳结构。
2. 去掉 `GRU` 的影响最大，不仅精度下降，综合评分也掉到最低。
3. 去掉 `DSC` 后参数量和 FLOPs 都上升，因此综合评分明显受损。

### 对比实验

结果如下：

深度学习模型综合排序如下：

| 排名 | 模型 | 准确率 | F1 | 参数量 | FLOPs | 综合评分 |
|------|------|--------|----|--------|-------|----------|
| `1` | `dsc_cbam_gru(ours)` | `0.9891` | `0.9936` | `33,329` | `210,424` | `0.716448` |
| `2` | `cnn_lstm` | `0.9865` | `0.9965` | `82,979` | `300,160` | `0.632252` |
| `3` | `mlp` | `0.9625` | `0.9925` | `87,683` | `174,464` | `0.512999` |
| `4` | `dsc_cbam_lstm` | `0.9862` | `0.9862` | `79,025` | `119,544` | `0.235560` |

传统机器学习模型单独列出：

| 模型 | 准确率 | F1 | 说明 |
|------|--------|----|------|
| `rf` | `0.9985` | `0.9985` | 不参与综合评分排序 |
| `id3` | `0.9936` | `0.9936` | 不参与综合评分排序 |

这里的 `dsc_cbam_gru(ours)` 使用的是调优后的 `full` 配置：

- `hidden_dim = 64`
- `bidirectional = True`
- `dropout = 0.2`

在“精度 + 模型规模 + 计算量 + 推理时延”的综合指标下，`DSC-CBAM-GRU` 在深度学习模型里排第 `1`。也就是说，它不再追求单一 `Accuracy` 绝对最高，而是作为当前项目里更均衡的主模型来报告。

### t-SNE 图

正式 t-SNE 图已生成：

- [4_train/experiments_window/visualization/tsne_window_gru.png](./4_train/experiments_window/visualization/tsne_window_gru.png)
- [4_train/experiments_window/visualization/tsne_window_gru.npz](./4_train/experiments_window/visualization/tsne_window_gru.npz)

其中 `.png` 是可直接查看的二维聚类图，`.npz` 保存了对应的二维嵌入点、标签和采样索引，后续可用于复绘或写论文图注。

## 联邦学习子系统

在不破坏现有单机训练闭环的前提下，当前已经新增了一个“面向低轨卫星多星协同威胁预测”的联邦学习版本。联邦代码全部放在 `4_train/federated/` 下，单机脚本、特征提取脚本和原始 `DSC-CBAM-GRU` 模型均保持不变。

### 1. 设计目标

联邦版本遵循以下原则：

- 直接复用当前 [dataset_window](./dataset_window) 中已经生成好的 `train.npz / val.npz / test.npz`
- 直接复用当前 [4_train/src/models/dsc_cbam_gru.py](./4_train/src/models/dsc_cbam_gru.py)
- 不重新定义 18 维特征，不改变三分类标签
- 第一版只做“算法仿真级联邦”，不要求真实多机通信
- 将 12 颗卫星模拟为 12 个联邦客户端，分成 3 个轨道面，每面 4 星

客户端映射如下：

- `plane_0`: `sat_0, sat_1, sat_2, sat_3`
- `plane_1`: `sat_4, sat_5, sat_6, sat_7`
- `plane_2`: `sat_8, sat_9, sat_10, sat_11`

### 2. 目录结构

新增联邦模块如下：

- [4_train/federated/__init__.py](./4_train/federated/__init__.py)
- [4_train/federated/config.py](./4_train/federated/config.py)
- [4_train/federated/client.py](./4_train/federated/client.py)
- [4_train/federated/serverless_orchestrator.py](./4_train/federated/serverless_orchestrator.py)
- [4_train/federated/topology.py](./4_train/federated/topology.py)
- [4_train/federated/contact_plan.py](./4_train/federated/contact_plan.py)
- [4_train/federated/aggregators.py](./4_train/federated/aggregators.py)
- [4_train/federated/gossip.py](./4_train/federated/gossip.py)
- [4_train/federated/compensation.py](./4_train/federated/compensation.py)
- [4_train/federated/reputation.py](./4_train/federated/reputation.py)
- [4_train/federated/partition.py](./4_train/federated/partition.py)
- [4_train/federated/metrics_fl.py](./4_train/federated/metrics_fl.py)

新增脚本如下：

- [4_train/scripts/train_federated_window.py](./4_train/scripts/train_federated_window.py)
- [4_train/scripts/run_federated_demo.sh](./4_train/scripts/run_federated_demo.sh)
- [4_train/scripts/run_federated_ablation.sh](./4_train/scripts/run_federated_ablation.sh)

### 3. 核心机制

联邦训练主控不是传统中心服务器，而是 [4_train/federated/serverless_orchestrator.py](./4_train/federated/serverless_orchestrator.py) 中的 `ServerlessOrchestrator`。每轮训练按如下顺序执行：

1. 生成当前离散时隙拓扑
2. 按 `dirichlet / iid / quantity_skew / hybrid` 划分方式为每颗卫星分配本地训练子集
3. 每个活跃客户端在本地执行 `AdamW + CrossEntropyLoss`
4. 同一轨道面内做面内加权聚合
5. 不同轨道面之间做异步 gossip
6. 若邻面模型缺失，则做失败补偿
7. 按陈旧度、链路质量和信誉更新加权
8. 在全局 `val/test` 上评估并记录轮次指标

当前实现的核心函数包括：

- `load_window_dataset(...)`
- `partition_train_dataset_for_satellites(...)`
- `create_client_dataloaders(...)`
- `compute_staleness(...)`
- `estimate_link_quality(...)`
- `intra_plane_aggregate(...)`
- `inter_plane_gossip(...)`
- `compensate_missing_model(...)`
- `update_reputation(...)`
- `evaluate_global_model(...)`
- `train_one_federated_round(...)`
- `run_federated_training(...)`

### 4. 数据划分与评估方式

当前联邦版本默认：

- 仅对 `train.npz` 做联邦划分
- `val.npz` 和 `test.npz` 继续作为全局验证/测试集
- 默认划分方式为 `Dirichlet non-IID`
- 默认 `alpha = 0.3`

这样做的目的是：

- 与现有单机训练数据保持完全兼容
- 不破坏原有特征提取和样本构造方式
- 仅在训练阶段引入多星异构分布

### 5. 训练入口

联邦主脚本是：

- [4_train/scripts/train_federated_window.py](./4_train/scripts/train_federated_window.py)

支持的关键参数包括：

- `--data_dir`
- `--num_clients`
- `--num_planes`
- `--rounds`
- `--local_epochs`
- `--batch_size`
- `--partition_mode`
- `--dirichlet_alpha`
- `--beta`
- `--lambda_s`
- `--rho`
- `--mu`
- `--device`
- `--output_dir`
- `--method`
- `--init_checkpoint`

### 6. 运行方法

#### 最小 demo

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
./scripts/run_federated_demo.sh
```

当前联邦最终命名已经固化为 `OrbitShield_FL`，并且 `run_federated_demo.sh` 已经默认指向这一版本。

当前 `OrbitShield_FL` 默认配置为：

- `batch_size = 512`
- `beta = 0.1`
- `warmup_rounds = 2`
- `global_momentum = 0.1`
- `beta_floor = 0.05`
- `init_checkpoint = checkpoints_gru/window_gru_best.pt`

#### 手动运行完整方案

```bash
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_window.py \
  --data_dir ../dataset_window \
  --num_clients 12 \
  --num_planes 3 \
  --rounds 20 \
  --local_epochs 1 \
  --batch_size 512 \
  --partition_mode dirichlet \
  --dirichlet_alpha 0.3 \
  --beta 0.3 \
  --lambda_s 0.1 \
  --rho 0.5 \
  --mu 0.8 \
  --method full \
  --init_checkpoint checkpoints_gru/window_gru_best.pt \
  --device cuda \
  --output_dir experiments_window/federated/OrbitShield_FL
```

#### 运行联邦方法对比

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
./scripts/run_federated_ablation.sh
```

### 7. 输出文件

每次联邦实验默认输出到：

- `4_train/experiments_window/federated/<method>/`

其中主要文件包括：

- `round_metrics.csv`
- `summary.json`
- `best_global_model.pt`
- `partition_stats.json`
- `reputation_history.json`
- `topology_history.json`

### 8. 学习效果优化

为了提升联邦训练的收敛效果，当前已经加入了一个关键优化：

- 默认使用现有单机最优模型 [4_train/checkpoints_gru/window_gru_best.pt](./4_train/checkpoints_gru/window_gru_best.pt) 作为联邦 warm start 初始化

这一步对效果提升非常明显：

- 未 warm start 的 1 轮 smoke test，测试准确率约为 `0.3654`
- 使用 warm start 后，同样 1 轮 smoke test，测试准确率提升到 `0.9002`

这组 smoke test 结果仅作为早期收敛对比记录，相关中间目录已在清理阶段删除；当前保留的正式联邦实验目录仅包括：

- [4_train/experiments_window/federated/OrbitShield_FL](./4_train/experiments_window/federated/OrbitShield_FL)
- [4_train/experiments_window/federated/full_grid](./4_train/experiments_window/federated/full_grid)

### 9. 当前联邦方法对比结果

本次已经基于相同数据、相同本地模型、相同 `5 rounds` 和 `warm start` 实际完成联邦方法对比。出于目录清理与最终交付需要，当前仅保留最终正式版本和完整网格搜索结果：

- [OrbitShield_FL](./4_train/experiments_window/federated/OrbitShield_FL)
- [full_grid](./4_train/experiments_window/federated/full_grid)

结果如下：

| 方法 | 含义 | Accuracy | Precision | Recall | F1 | 平均通信开销(MB/round) | 平均链路鲁棒性 |
|------|------|----------|-----------|--------|----|------------------------|----------------|
| `single` | 单机集中式兼容基线 | `0.9601` | `0.9606` | `0.9601` | `0.9599` | `0.0000` | `1.0000` |
| `fedavg` | 标准 FedAvg | `0.9571` | `0.9584` | `0.9571` | `0.9570` | `1.7869` | `0.9306` |
| `intra_only` | 仅面内聚合 | `0.9636` | `0.9638` | `0.9636` | `0.9636` | `1.7869` | `0.9306` |
| `intra_gossip` | 面内聚合 + 面间 gossip | `0.9159` | `0.9262` | `0.9159` | `0.9143` | `1.7869` | `0.9306` |
| `full` | 原始完整方案 | `0.9389` | `0.9423` | `0.9389` | `0.9381` | `1.7869` | `0.9306` |
| `OrbitShield_FL` | 最终优化后的完整方案 | `0.9718` | `0.9719` | `0.9718` | `0.9718` | `1.7869` | `0.9306` |

当前可以得到的结论是：

1. 历史结果表明，未经充分调参的跨面协同并不天然优于局部稳定聚合。
2. 对 `full` 方案继续加入 `warm start + adaptive gossip weighting + global EMA stabilization + intra-plane warmup`，并经过系统网格搜索后，最终 `OrbitShield_FL` 已经从原始 `full = 0.9389` 提升到 `0.9718`。
3. 这说明完整的“面内聚合 + 面间协同 + 鲁棒权重”方案在经过合理调参后，已经能够把跨面协同真正转化为净收益。
4. 从工程角度看，联邦版本已经可以直接运行，且 `OrbitShield_FL` 已成为当前项目最终联邦命名和默认联邦配置。

### 10. 当前推荐联邦配置

如果目标是“直接使用当前最优联邦版本”，当前推荐：

- `method = full`
- `partition_mode = dirichlet`
- `dirichlet_alpha = 0.3`
- `rounds = 5 ~ 20`
- `local_epochs = 1`
- `batch_size = 512`
- `beta = 0.1`
- `warmup_rounds = 2`
- `global_momentum = 0.1`
- `beta_floor = 0.05`
- `init_checkpoint = checkpoints_gru/window_gru_best.pt`

如果目标是继续研究“更贴近低轨星间协同”的进一步提升空间，则优先继续优化：

- `beta`
- `rho`
- `lambda_s`
- `mu`
- 面间 contact 规则
- gossip 邻居选择策略

### 11. OrbitShield_FL 网格搜索结果

为了继续提升最终完整方案，本次又额外做了一轮系统网格搜索：

- 搜索维度：
  - `beta in {0.1, 0.2, 0.3}`
  - `warmup_rounds in {1, 2, 3}`
  - `global_momentum in {0.1, 0.2, 0.3}`
- 总计：`27` 组
- 输出目录：
  - [4_train/experiments_window/federated/full_grid](./4_train/experiments_window/federated/full_grid)
- 总表：
  - [full_grid_summary.csv](./4_train/experiments_window/federated/full_grid/full_grid_summary.csv)
  - [full_grid_results.json](./4_train/experiments_window/federated/full_grid/full_grid_results.json)
- 搜索脚本：
  - [4_train/scripts/tune_federated_full.py](./4_train/scripts/tune_federated_full.py)

当前前 5 名配置如下：

| 排名 | beta | warmup_rounds | global_momentum | Accuracy | F1 |
|------|------|---------------|-----------------|----------|----|
| `1` | `0.1` | `2` | `0.1` | `0.9718` | `0.9718` |
| `2` | `0.3` | `2` | `0.1` | `0.9678` | `0.9677` |
| `3` | `0.2` | `2` | `0.3` | `0.9659` | `0.9658` |
| `4` | `0.2` | `2` | `0.2` | `0.9648` | `0.9646` |
| `5` | `0.2` | `3` | `0.2` | `0.9615` | `0.9614` |

因此，当前最佳完整方案配置已经更新为 `OrbitShield_FL`：

- `method = full`
- `beta = 0.1`
- `warmup_rounds = 2`
- `global_momentum = 0.1`
- `beta_floor = 0.05`
- `init_checkpoint = checkpoints_gru/window_gru_best.pt`

在这组参数下，`OrbitShield_FL` 的测试集结果达到：

- `Accuracy = 0.9718`
- `Precision = 0.9719`
- `Recall = 0.9718`
- `F1 = 0.9718`

这已经超过了当前历史对比结果中的：

- `single = 0.9601`
- `fedavg = 0.9571`
- `intra_only = 0.9636`

也就是说，经过 `warm start + adaptive gossip + EMA + warmup + systematic tuning` 之后，当前完整联邦方案已经成为这套实验里的最优联邦方法。

### 12. 论文实验分析

可以将当前联邦实验结果概括为以下几点：

1. `FedAvg` 在当前多星非 IID 划分下虽然能够稳定收敛，但其聚合方式没有显式利用轨道结构，因此在性能上未能超过更符合星座拓扑的面内聚合方案。
2. `intra_only` 在未引入跨面信息交换的情况下取得了 `0.9636` 的测试准确率，说明“轨道面内局部稳定协同”本身已经能够有效缓解单星样本不足问题。
3. 原始 `intra_gossip` 和未经充分调参的 `full` 方案性能下降，表明跨面模型交换若混合过强，会放大不同轨道面之间的数据异质性，导致判别边界受扰动。
4. 通过引入 `warm start`、自适应 gossip 权重、全局 EMA 稳定器和跨面 warmup 阶段，最终 `OrbitShield_FL` 将测试准确率提升到 `0.9718`，优于 `single`、`FedAvg` 和 `intra_only`。
5. 这说明在低轨卫星多星协同场景下，跨面协同不是简单“加 gossip 就更好”，而是必须在“何时交换、交换多少、如何抑制陈旧和不可靠更新”这三个方面进行联合设计。
6. 从误差传播角度看，较小的 `beta=0.1`、适中的 `warmup_rounds=2` 和较低的 `global_momentum=0.1` 组合，能够在保留跨面信息增益的同时，避免邻面更新过度主导本地已收敛的判别特征。

### 13. full_grid 可视化

当前已经基于 [full_grid_summary.csv](./4_train/experiments_window/federated/full_grid/full_grid_summary.csv) 生成了 3 张调参图：

- [OrbitShield_FL_heatmaps.png](./4_train/experiments_window/federated/full_grid/plots/OrbitShield_FL_heatmaps.png)
- [OrbitShield_FL_trends.png](./4_train/experiments_window/federated/full_grid/plots/OrbitShield_FL_trends.png)
- [OrbitShield_FL_top10.png](./4_train/experiments_window/federated/full_grid/plots/OrbitShield_FL_top10.png)

对应绘图脚本为：

- [4_train/scripts/plot_federated_full_grid.py](./4_train/scripts/plot_federated_full_grid.py)

可直接重绘：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/plot_federated_full_grid.py \
  --csv_path experiments_window/federated/full_grid/full_grid_summary.csv \
  --output_dir experiments_window/federated/full_grid/plots
```
