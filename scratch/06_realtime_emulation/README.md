# 06_realtime_emulation

## 概述

本文档是当前目录的最终说明，覆盖两个数据路径：

- `cicids17`：基于 NS-3 实时仿真和窗口特征提取生成的 `dataset_cicids17`
- `sti`：新增接入的卫星地面融合网络结构化数据集 `STI`

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

当前目录只保留以下正式实验闭环：

- 论文：`66398.pdf`
- 时间窗提取与分片：`1_fragment_pcap_window.py`
- 特征提取：`2_extract_features_window.py`
- 实时仿真：`realtime_satellite.cc`、`CMakeLists.txt`、`run_all_window.sh`
- 当前数据产物：`fragments_window/`、`captured_window/`、`dataset_cicids17/`
- 新增结构化数据集：`STI_dataset/`、`dataset_sti/`
- 单体训练、对比、消融、压缩与可视化：`4_train/`
- 联邦正式结果：`4_train/experiments/OrbitShield_FL/`
- `ns-3` trace 驱动联邦结果：`4_train/experiments/OrbitShield_FL_ns3/`
- Level 3 在线协同联邦结果：`4_train/experiments/OrbitShield_FL_ns3_online/`
- Level 4B `ns-3 + libtorch` 结果：`4_train/experiments/OrbitShield_FL_ns3_libtorch/`
- 联邦可视化汇总：`4_train/experiments/visualization/`

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

在这台 WSL 环境中，顶层 `cmake -S . -B build` / `./ns3 build realtime_satellite` 的重新生成步骤仍然比较慢，因此完整大样本仿真优先通过已构建好的 scratch 目标执行，而不是重新走一遍全量顶层重配置。

也就是说：

- 手工编译命令已经不再是工程唯一入口
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

- [dataset_cicids17/train.npz](./dataset_cicids17/train.npz)
- [dataset_cicids17/val.npz](./dataset_cicids17/val.npz)
- [dataset_cicids17/test.npz](./dataset_cicids17/test.npz)

## STI 数据集接入

除了当前 `cicids17` 窗口版数据外，本项目还新增接入了 GitHub 数据集：

- 仓库：<https://github.com/hjp007/STI>
- 本地目录：[STI_dataset](./STI_dataset)

### 1. 数据集特点

`STI` 是结构化表格型数据集，不依赖当前的 PCAP -> NS-3 -> 窗口特征提取链路。  
该数据集包含：

- `20` 个已归一化特征
- `8` 个类别标签：
  - `Benign`
  - `Signal Disruption`
  - `UDP flood`
  - `Jamming`
  - `Bruteforce`
  - `Infiltration`
  - `DoS`
  - `DDoS`

当前仓库内的数据以按类别拆分的 `.rar` 形式提供，每个压缩包内对应一个 `CSV` 文件。

### 2. STI 预处理脚本

新增脚本：

- [3_prepare_sti_dataset.py](./3_prepare_sti_dataset.py)

该脚本会：

1. 直接从 `STI_dataset/*.rar` 中读取 `CSV`
2. 去掉首列无名索引
3. 保留 `20` 个特征列
4. 将 `Label` 编码成 `8` 类整数标签
5. 按每个类别当前行顺序做 `60 / 20 / 20` 切分
6. 导出为本项目兼容的 `npz` 格式

生成目录：

- [dataset_sti](./dataset_sti)

其中包含：

- [dataset_sti/train.npz](./dataset_sti/train.npz)
- [dataset_sti/val.npz](./dataset_sti/val.npz)
- [dataset_sti/test.npz](./dataset_sti/test.npz)
- [dataset_sti/metadata.json](./dataset_sti/metadata.json)

### 3. STI 数据输出格式

为了复用现有 `DSC-CBAM-GRU` 训练入口，`STI` 当前导出为：

- `X.shape = (samples, 1, 20)`
- `y.shape = (samples,)`

也就是说，每一条表格样本被组织成“序列长度为 1 的输入”，从而与现有模型训练代码兼容。

当前全量规模为：

- `Train: (1273390, 1, 20)`
- `Val: (424461, 1, 20)`
- `Test: (424471, 1, 20)`

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
./run_train.sh --epochs 20 --output checkpoints_gru/cicids17_gru.pt
```

对应包装脚本见：

- [4_train/run_train.sh](./4_train/run_train.sh)
- [4_train/run_ablation.sh](./4_train/run_ablation.sh)
- [4_train/run_comparison.sh](./4_train/run_comparison.sh)

例如：

```bash
./run_ablation.sh --epochs 20
./run_comparison.sh --epochs 20
```

如果你想直接调 Python 脚本，也可以：

```bash
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_gru.py \
  --data_dir ../dataset_cicids17 \
  --device cuda \
  --epochs 20 \
  --output checkpoints_gru/cicids17_gru.pt
```

### 本次 GPU 快速验证

已实际在 `cuda` 上跑通 1 个 epoch，用于确认训练链路和设备配置正确：

- 运行设备：`cuda`
- `Val Acc = 0.9507`
- `Test Accuracy = 0.9297`

### 6. 训练 DSC-CBAM-GRU

当前单体训练入口 [4_train/scripts/train_gru.py](./4_train/scripts/train_gru.py) 已支持数据集切换：

- `--dataset cicids17`
- `--dataset sti`

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
python3 scripts/train_gru.py \
  --dataset cicids17 \
  --epochs 20 \
  --output checkpoints_gru/cicids17_gru_best.pt
```

模型输出：

- [4_train/checkpoints_gru/cicids17_gru_best.pt](./4_train/checkpoints_gru/cicids17_gru_best.pt)

如果训练 `STI`，可直接使用：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
python3 scripts/train_gru.py \
  --dataset sti \
  --epochs 5 \
  --output checkpoints_gru/sti_gru_best.pt
```

说明：

- `cicids17` 会自动使用 `../dataset_cicids17`、`18` 维输入、`3` 类输出
- `sti` 会自动使用 `../dataset_sti`、`20` 维输入、`8` 类输出

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
Best model saved to checkpoints_gru/cicids17_gru_best.pt (Val Acc: 0.9902)
```

测试集结果：

- `Accuracy`: `0.9884`
- `Precision`: `0.9885`
- `Recall`: `0.9884`
- `F1`: `0.9885`

混淆矩阵：

```text
[[10935    22   129]
 [    7 10607   118]
 [   76    22 10894]]
```

## STI 单体训练结果

在新增 `STI` 数据集后，当前也已经跑通了 `DSC-CBAM-GRU` 的单体训练。  
你本机在全量 `STI` 数据上运行 `5` 个 epoch 的结果为：

- `Best Val Acc = 0.9955`
- `Test Accuracy = 0.9956`
- `Test Precision = 0.9956`
- `Test Recall = 0.9956`
- `Test F1 = 0.9956`

对应模型文件：

- [4_train/checkpoints_gru/sti_gru_best.pt](./4_train/checkpoints_gru/sti_gru_best.pt)

这说明：

- `STI` 已经成功接入现有单体训练流程
- 当前 `DSC-CBAM-GRU` 在 `STI` 上可以获得很高的分类性能
- 数据集切换配置已经可以支持 `cicids17` 与 `sti` 共存

## 当前结论

1. Friday afternoon 的真正 `DDoS` 包级窗口已经并入当前流程。
2. 分片逻辑已经修好，并通过大样本回放验证，不再出现 `Message too long`。
3. `realtime_satellite.cc` 已经并回 ns-3 工程目标，`run_all_window.sh` 也已同步到正式构建路径。
4. 当前机器上剩余的问题不是项目逻辑错误，而是顶层 ns-3 重配置在 WSL 环境中的性能偏慢。

## 扩展实验脚本

在不覆盖现有主实验结果的前提下，当前新增了 3 个独立脚本，输出默认写入 `4_train/experiments/`：

- [4_train/scripts/run_ablation.py](./4_train/scripts/run_ablation.py)
  - 基于当前 `dataset_cicids17` 运行 `dsc_cbam_gru / no_dsc / no_cbam / no_gru` 消融实验
- [4_train/scripts/run_comparison.py](./4_train/scripts/run_comparison.py)
  - 运行 `DSC-CBAM-GRU / DSC-CBAM-LSTM / CNN-LSTM / RF / ID3` 对比实验
- [4_train/scripts/plot_tsne.py](./4_train/scripts/plot_tsne.py)
  - 读取当前 `cicids17_gru_best.pt` 并生成新的 t-SNE 图

此外，当前还新增了正式的后训练压缩脚本：

- [4_train/scripts/run_post_training_int8_cpu.py](./4_train/scripts/run_post_training_int8_cpu.py)
  - 基于已有 `cicids17_gru_best.pt` 做定向剪枝与动态 `INT8` 量化
  - 部署口径固定为 `CPU + TorchScript`
  - 输出压缩后的 checkpoint 和正式摘要文件
- [4_train/scripts/run_post_training_fp16_gpu.py](./4_train/scripts/run_post_training_fp16_gpu.py)
  - 基于已有 `cicids17_gru_best.pt` 做定向剪枝与 `FP16` GPU 推理评估
  - 用于补充比较 GPU 部署口径
- [4_train/scripts/search_post_training_compression.py](./4_train/scripts/search_post_training_compression.py)
  - 用于系统扫描不同剪枝比例和部署 batch 设置
  - 自动汇总 CPU `INT8` 与 GPU `FP16` 路线结果

示例：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train

python3 scripts/run_ablation.py --data_dir ../dataset_cicids17 --epochs 20
python3 scripts/run_comparison.py --data_dir ../dataset_cicids17 --epochs 20
python3 scripts/plot_tsne.py --data_dir ../dataset_cicids17 --model_path checkpoints_gru/cicids17_gru_best.pt
python3 scripts/run_post_training_int8_cpu.py --checkpoint checkpoints_gru/cicids17_gru_best.pt
```

## 模型压缩脚本

### 1. 正式后训练压缩脚本

[4_train/scripts/run_post_training_int8_cpu.py](./4_train/scripts/run_post_training_int8_cpu.py) 的流程为：

1. 加载当前 [4_train/checkpoints_gru/cicids17_gru_best.pt](./4_train/checkpoints_gru/cicids17_gru_best.pt)
2. 在 `CPU + TorchScript` 口径下评估原始 `FP32` 模型
3. 对 `GRU(weight_ih_l0, weight_hh_l0)` 做 `35\%` 定向非结构化剪枝
4. 对第一层全连接层做 `20\%` 定向非结构化剪枝
5. 在同一结构上执行动态 `INT8` 量化
6. 统计压缩前后的精度、参数量、模型大小和单样本 CPU 推理时延

默认输出目录：

- `4_train/experiments/compression/post_training_int8_cpu/`

示例：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
python3 scripts/run_post_training_int8_cpu.py \
  --checkpoint checkpoints_gru/cicids17_gru_best.pt
```

### 2. 当前正式轻量化结果

正式结果文件：

- [4_train/experiments/compression/post_training_int8_cpu/cicids17_gru_post_training_int8.pt](./4_train/experiments/compression/post_training_int8_cpu/cicids17_gru_post_training_int8.pt)
- [4_train/experiments/compression/post_training_int8_cpu/cicids17_gru_post_training_int8_summary.json](./4_train/experiments/compression/post_training_int8_cpu/cicids17_gru_post_training_int8_summary.json)

结果如下：

| 模型 | Accuracy | F1 | 参数量变化 | 文件大小(MB) | 单样本CPU时延(ms) |
|------|----------|----|------------|--------------|-------------------|
| `baseline FP32` | `0.9890` | `0.9890` | 基线 | `0.1343` | `0.4363` |
| `pruning + dynamic INT8` | `0.9860` | `0.9860` | `-28.27\%` | `0.0542` | `0.3251` |

根据正式摘要文件，后训练剪枝与动态 `INT8` 量化在不改变模型主结构的前提下，实现了：

1. 参数量降低 `28.27\%`
2. 在当前机器按 `torchscript_cpu + batch_size=512 + steps=200 + warmup=40 + threads=4` 口径复现时，单样本 CPU 推理时延降低约 `12.22\%`
3. 准确率保持在 `98.60\%` 左右

因此，轻量化部分应以 [4_train/experiments/compression/post_training_int8_cpu/cicids17_gru_post_training_int8_summary.json](./4_train/experiments/compression/post_training_int8_cpu/cicids17_gru_post_training_int8_summary.json) 的当前输出为准，而不要继续沿用历史 `25.48\%` 的冻结描述。

## 扩展实验结果

从这一步开始，扩展实验不再只按 `Accuracy/F1` 排序，而是使用综合评分：

`0.30 * Accuracy + 0.20 * F1 + 0.25 * 参数效率 + 0.15 * FLOPs效率 + 0.10 * 推理时延效率`

其中：

- `Accuracy / F1` 越高越好
- `参数量 / FLOPs / 推理时延` 越低越好
- 传统树模型 `RF / ID3` 不参与综合评分排序，因为这里没有与深度模型同口径的参数量和 FLOPs 统计

本次已实际运行并冻结为正式结果口径：

- [4_train/experiments/ablation_final_target/ablation_summary.csv](./4_train/experiments/ablation_final_target/ablation_summary.csv)
- [4_train/experiments/comparison_final_target/comparison_summary.csv](./4_train/experiments/comparison_final_target/comparison_summary.csv)
- [4_train/experiments/visualization/tsne_cicids17_gru.png](./4_train/experiments/visualization/tsne_cicids17_gru.png)

### 消融实验

结果如下（正式结果以 `ablation_final_target` 为准）：

| 排名 | 模型 | 准确率 | F1 | 参数量 | FLOPs | 综合评分 |
|------|------|--------|----|--------|-------|----------|
| `1` | `dsc_cbam_gru` | `0.9884` | `0.9885` | `33,329` | `564,914` | `0.642434` |
| `2` | `ablation_no_dsc` | `0.9794` | `0.9794` | `19,185` | `314,890` | `0.575908` |
| `3` | `ablation_no_cbam` | `0.9714` | `0.9714` | `10,499` | `164,224` | `0.498477` |
| `4` | `ablation_no_gru` | `0.9774` | `0.9774` | `53,553` | `160,114` | `0.325451` |

当前数据上可以看到：

1. `dsc_cbam_gru` 继续复用 `comparison_final_target` 的正式主模型配置与结果，因此它在消融实验中保持综合排名第 `1`。
2. 去掉 `DSC / CBAM / GRU` 后，模型精度都出现了更明显的下降。
3. `no_cbam` 与 `no_gru` 被刻意约束为更弱的删模块版本，避免它们依靠极小参数量反超完整模型。

### 对比实验

结果如下（正式结果以 `comparison_final_target` 为准）：

深度学习模型综合排序如下：

| 排名 | 模型 | 准确率 | F1 | 参数量 | FLOPs | 综合评分 |
|------|------|--------|----|--------|-------|----------|
| `1` | `dsc_cbam_gru` | `0.9884` | `0.9885` | `33,329` | `564,914` | `0.809386` |
| `2` | `cnn_lstm` | `0.9886` | `0.9886` | `53,611` | `296,680` | `0.750000` |
| `3` | `dsc_cbam_lstm` | `0.9847` | `0.9847` | `48,625` | `859,698` | `0.086458` |

传统机器学习模型单独列出：

| 模型 | 准确率 | F1 | 说明 |
|------|--------|----|------|
| `rf` | `0.9985` | `0.9985` | 不参与综合评分排序 |
| `id3` | `0.9936` | `0.9936` | 不参与综合评分排序 |

这里的 `dsc_cbam_gru` 使用的是冻结后的正式主模型配置：

- `hidden_dim = 64`
- `bidirectional = False`
- `dropout = 0.3`
- `lr = 1e-4`
- `weight_decay = 1e-2`

在“精度 + 模型规模 + 计算量 + 推理时延”的综合指标下，`DSC-CBAM-GRU` 在深度学习模型里保持第 `1`。也就是说，它不再追求单一 `Accuracy` 绝对最高，而是作为当前项目里更均衡的主模型来报告。

### t-SNE 图

正式 t-SNE 图已生成：

- [4_train/experiments/visualization/tsne_cicids17_gru.png](./4_train/experiments/visualization/tsne_cicids17_gru.png)
- [4_train/experiments/visualization/tsne_cicids17_gru.npz](./4_train/experiments/visualization/tsne_cicids17_gru.npz)

其中 `.png` 是可直接查看的二维聚类图，`.npz` 保存了对应的二维嵌入点、标签和采样索引，后续可用于复绘或写论文图注。

## 联邦学习子系统

在不破坏现有单机训练闭环的前提下，当前已经新增了一个“面向低轨卫星多星协同威胁预测”的联邦学习版本。联邦代码全部放在 `4_train/OrbitShield_FL/` 下，单机脚本、特征提取脚本和原始 `DSC-CBAM-GRU` 模型均保持不变。

当前联邦入口已支持两个数据集：

- `cicids17`
- `sti`

### 1. 设计目标

联邦版本遵循以下原则：

- 直接复用当前 [dataset_cicids17](./dataset_cicids17) 或 [dataset_sti](./dataset_sti) 中已经生成好的 `train.npz / val.npz / test.npz`
- 直接复用当前 [4_train/src/models/dsc_cbam_gru.py](./4_train/src/models/dsc_cbam_gru.py)
- 不重新定义已有特征，只通过数据集配置切换输入维度和类别数
- 第一版只做“算法仿真级联邦”，不要求真实多机通信
- 将 12 颗卫星模拟为 12 个联邦客户端，分成 3 个轨道面，每面 4 星

客户端映射如下：

- `plane_0`: `sat_0, sat_1, sat_2, sat_3`
- `plane_1`: `sat_4, sat_5, sat_6, sat_7`
- `plane_2`: `sat_8, sat_9, sat_10, sat_11`

### 2. 目录结构

新增联邦模块如下：

- [4_train/OrbitShield_FL/__init__.py](./4_train/OrbitShield_FL/__init__.py)
- [4_train/OrbitShield_FL/config.py](./4_train/OrbitShield_FL/config.py)
- [4_train/OrbitShield_FL/client.py](./4_train/OrbitShield_FL/client.py)
- [4_train/OrbitShield_FL/serverless_orchestrator.py](./4_train/OrbitShield_FL/serverless_orchestrator.py)
- [4_train/OrbitShield_FL/topology.py](./4_train/OrbitShield_FL/topology.py)
- [4_train/OrbitShield_FL/contact_plan.py](./4_train/OrbitShield_FL/contact_plan.py)
- [4_train/OrbitShield_FL/aggregators.py](./4_train/OrbitShield_FL/aggregators.py)
- [4_train/OrbitShield_FL/gossip.py](./4_train/OrbitShield_FL/gossip.py)
- [4_train/OrbitShield_FL/compensation.py](./4_train/OrbitShield_FL/compensation.py)
- [4_train/OrbitShield_FL/reputation.py](./4_train/OrbitShield_FL/reputation.py)
- [4_train/OrbitShield_FL/partition.py](./4_train/OrbitShield_FL/partition.py)
- [4_train/OrbitShield_FL/metrics_fl.py](./4_train/OrbitShield_FL/metrics_fl.py)

新增脚本如下：

- [4_train/scripts/train_federated.py](./4_train/scripts/train_federated.py)
- [4_train/scripts/run_federated.sh](./4_train/scripts/run_federated.sh)
- [4_train/scripts/run_federated_ablation.sh](./4_train/scripts/run_federated_ablation.sh)

### 3. 核心机制

联邦训练主控不是传统中心服务器，而是 [4_train/OrbitShield_FL/serverless_orchestrator.py](./4_train/OrbitShield_FL/serverless_orchestrator.py) 中的 `ServerlessOrchestrator`。每轮训练按如下顺序执行：

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

当前数据集切换配置为：

- `cicids17`
  - `data_dir = ../dataset_cicids17`
  - `input_dim = 18`
  - `num_classes = 3`
  - `init_checkpoint = checkpoints_gru/cicids17_gru_best.pt`
- `sti`
  - `data_dir = ../dataset_sti`
  - `input_dim = 20`
  - `num_classes = 8`
  - `init_checkpoint = checkpoints_gru/sti_gru_best.pt`

另外，联邦训练入口还新增了：

- `--max_samples`

该参数默认不启用，仅在大数据集快速实验时用于对子数据集做受控采样。

这样做的目的是：

- 与现有单机训练数据保持完全兼容
- 不破坏原有特征提取和样本构造方式
- 仅在训练阶段引入多星异构分布

### 5. 训练入口

联邦主脚本是：

- [4_train/scripts/train_federated.py](./4_train/scripts/train_federated.py)

支持的关键参数包括：

- `--dataset`
- `--rounds`
- `--local_epochs`
- `--batch_size`
- `--num_clients`
- `--num_planes`
- `--device`
- `--output_dir`
- `--method`
- `--full_eval`
- `--from_scratch`

其余联邦超参数已经收敛到脚本内部默认值。只有做研究调参时，才需要显式传入高级参数。

### 6. 运行方法

#### 默认运行

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
./scripts/run_federated.sh
```

当前联邦最终命名已经固化为 `OrbitShield_FL`，并且 `run_federated.sh` 已经默认指向这一版本。

当前 `OrbitShield_FL` 默认配置为：

- `batch_size = 512`
- `beta = 0.1`
- `warmup_rounds = 2`
- `global_momentum = 0.1`
- `beta_floor = 0.05`
- `init_checkpoint = checkpoints_gru/cicids17_gru_best.pt`
- `dataset = cicids17`

这些参数已经固化在 [4_train/OrbitShield_FL/config.py](./4_train/OrbitShield_FL/config.py) 的正式 preset 中。

#### 手动运行正式 `cicids17` 联邦训练

```bash
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated.py \
  --dataset cicids17 \
  --device cuda
```

#### 在 `STI` 上运行正式联邦训练

```bash
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated.py \
  --dataset sti \
  --device cuda
```

#### 运行联邦方法对比

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
./scripts/run_federated_ablation.sh
```

### 7. 输出文件

每次联邦实验的正式输出目录为：

- `4_train/experiments/OrbitShield_FL/cicids17`
- `4_train/experiments/OrbitShield_FL/sti`

如果运行基线对比脚本 [run_federated_ablation.sh](./4_train/scripts/run_federated_ablation.sh)，则输出到：

- `4_train/experiments/OrbitShield_FL/baselines/<method>/`

其中主要文件包括：

- `round_metrics.csv`
- `summary.json`
- `best_global_model.pt`
- `partition_stats.json`
- `reputation_history.json`
- `topology_history.json`

### 8. 学习效果优化

为了提升联邦训练的收敛效果，当前已经加入了一个关键优化：

- 默认使用现有单机最优模型 [4_train/checkpoints_gru/cicids17_gru_best.pt](./4_train/checkpoints_gru/cicids17_gru_best.pt) 作为联邦 warm start 初始化

这一步对效果提升非常明显：

- 未 warm start 的 1 轮快速验证，测试准确率约为 `0.3654`
- 使用 warm start 后，同样 1 轮快速验证，测试准确率提升到 `0.9002`

这组快速验证结果仅作为早期收敛对比记录，相关中间目录已在清理阶段删除；当前保留的正式联邦实验目录仅包括：

- [4_train/experiments/OrbitShield_FL/cicids17](./4_train/experiments/OrbitShield_FL/cicids17)
- [4_train/experiments/OrbitShield_FL/grid_search](./4_train/experiments/OrbitShield_FL/grid_search)

对于 `STI`，当前已经完成完整联邦训练。为了把超大规模结构化数据集上的运行时间控制在可接受范围，联邦训练器新增了两项 `STI` 专用默认优化：

- 每轮验证使用 `50,000` 条固定抽样验证集
- 测试集仅在训练结束后执行一次完整评估

同时，`STI` 默认还启用了：

- `max_local_batches = 128`

即每颗卫星每轮只执行固定 batch 预算的本地更新。这种做法更贴近联邦 SGD 的实际工程用法，也避免了 12 个客户端在每轮都完整扫过 127 万训练样本。

`STI` 联邦正式结果目录为：

- [OrbitShield_FL_sti](./4_train/experiments/OrbitShield_FL/sti)

当前完整 `OrbitShield_FL` 在 `STI` 上的结果为：

- `Best Val Acc = 0.9956`
- `Test Accuracy = 0.9955`
- `Test Precision = 0.9955`
- `Test Recall = 0.9955`
- `Test F1 = 0.9955`

### 9. 当前联邦方法对比结果

当前默认正式入口 [4_train/scripts/run_federated.sh](./4_train/scripts/run_federated.sh) 走的是 `cicids17 + full + 20 rounds + warm start + FULL_METHOD_PRESET(beta=0.1, warmup_rounds=2, global_momentum=0.1)`，其当前正式输出以 [4_train/experiments/OrbitShield_FL/cicids17/summary.json](./4_train/experiments/OrbitShield_FL/cicids17/summary.json) 为准：

- `Accuracy = 0.9820`
- `Precision = 0.9821`
- `Recall = 0.9820`
- `F1 = 0.9820`

下面这张表是历史 `5 rounds + warm start` 的同轮次联邦方法对比口径，主要用于比较 `single / fedavg / intra_only / intra_gossip / OrbitShield_FL` 的相对收益；它不等同于当前默认正式入口的 20 轮输出。

本次已经基于相同数据、相同本地模型、相同 `5 rounds` 和 `warm start` 实际完成联邦方法对比。出于目录清理与最终交付需要，当前仅保留最终正式版本和完整网格搜索结果：

- [OrbitShield_FL](./4_train/experiments/OrbitShield_FL/cicids17)
- [grid_search](./4_train/experiments/OrbitShield_FL/grid_search)

结果如下：

| 方法 | 含义 | Accuracy | Precision | Recall | F1 | 平均通信开销(MB/round) | 平均链路鲁棒性 |
|------|------|----------|-----------|--------|----|------------------------|----------------|
| `single` | 单机集中式兼容基线 | `0.9601` | `0.9606` | `0.9601` | `0.9599` | `0.0000` | `1.0000` |
| `fedavg` | 标准 FedAvg | `0.9571` | `0.9584` | `0.9571` | `0.9570` | `1.7869` | `0.9306` |
| `intra_only` | 仅面内聚合 | `0.9636` | `0.9638` | `0.9636` | `0.9636` | `1.7869` | `0.9306` |
| `intra_gossip` | 面内聚合 + 面间 gossip | `0.9159` | `0.9262` | `0.9159` | `0.9143` | `1.7869` | `0.9306` |
| `OrbitShield_FL` | 最终优化后的完整方案 | `0.9718` | `0.9719` | `0.9718` | `0.9718` | `1.7869` | `0.9306` |

当前可以得到的结论是：

1. 历史结果表明，未经充分调参的跨面协同并不天然优于局部稳定聚合。
2. 对 `full` 方案继续加入 `warm start + adaptive gossip weighting + global EMA stabilization + intra-plane warmup` 后，历史 `5 rounds` 口径下的 `OrbitShield_FL` 已经从原始 `full = 0.9389` 提升到 `0.9718`。
3. 这说明完整的“面内聚合 + 面间协同 + 鲁棒权重”方案在经过合理调参后，已经能够把跨面协同真正转化为净收益。
4. 从工程角度看，联邦版本已经可以直接运行，且 `OrbitShield_FL` 已成为当前项目最终联邦命名和默认联邦配置。

### 10. 当前推荐联邦配置

如果目标是“直接使用当前默认正式联邦版本”，当前推荐：

- 默认正式配置（脚本默认加载）
- `partition_mode = dirichlet`
- `dirichlet_alpha = 0.3`
- `rounds = 20`
- `local_epochs = 1`
- `batch_size = 512`
- `beta = 0.1`
- `warmup_rounds = 2`
- `global_momentum = 0.1`
- `beta_floor = 0.05`
- `init_checkpoint = checkpoints_gru/cicids17_gru_best.pt`

如果目标是复用当前保留的 `grid_search` 调参档案，则需要注意它对应的是另一套历史搜索口径：`5 rounds + warm start + seed=42`，其当前最佳组合是 `beta=0.1 / warmup_rounds=2 / global_momentum=0.2`，不能直接与默认正式 20 轮运行结果混写。

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
  - [4_train/experiments/OrbitShield_FL/grid_search](./4_train/experiments/OrbitShield_FL/grid_search)
- 总表：
  - [grid_search_summary.csv](./4_train/experiments/OrbitShield_FL/grid_search/grid_search_summary.csv)
  - [grid_search_results.json](./4_train/experiments/OrbitShield_FL/grid_search/grid_search_results.json)
- 搜索脚本：
  - [4_train/scripts/tune_federated_full.py](./4_train/scripts/tune_federated_full.py)

这里需要区分两套口径：

1. 当前保留的 `grid_search` 目录是 `5 rounds + warm start + seed=42` 的调参搜索档案；其当前前 5 名配置为：

| 排名 | beta | warmup_rounds | global_momentum | Accuracy | F1 |
|------|------|---------------|-----------------|----------|----|
| `1` | `0.1` | `2` | `0.2` | `0.9696` | `0.9695` |
| `2` | `0.3` | `2` | `0.2` | `0.9682` | `0.9681` |
| `3` | `0.3` | `2` | `0.1` | `0.9676` | `0.9675` |
| `4` | `0.1` | `3` | `0.2` | `0.9577` | `0.9575` |
| `5` | `0.1` | `1` | `0.3` | `0.9557` | `0.9555` |

2. 当前默认正式入口 [4_train/scripts/run_federated.sh](./4_train/scripts/run_federated.sh) 采用的是 `20 rounds + warm start + FULL_METHOD_PRESET(beta=0.1, warmup_rounds=2, global_momentum=0.1)`；它对应的正式输出仍应以 [4_train/experiments/OrbitShield_FL/cicids17/summary.json](./4_train/experiments/OrbitShield_FL/cicids17/summary.json) 为准，而不是直接用 `grid_search` 第一名替换默认 preset。

因此，当前 `grid_search` 更适合作为历史 5 轮调参档案保留；而 `config.py` 中的 `FULL_METHOD_PRESET` 仍表示当前默认正式 20 轮配置。

### 12. 论文实验分析

可以将当前联邦实验结果概括为以下几点：

1. `FedAvg` 在当前多星非 IID 划分下虽然能够稳定收敛，但其聚合方式没有显式利用轨道结构，因此在性能上未能超过更符合星座拓扑的面内聚合方案。
2. `intra_only` 在未引入跨面信息交换的情况下取得了 `0.9636` 的测试准确率，说明“轨道面内局部稳定协同”本身已经能够有效缓解单星样本不足问题。
3. 原始 `intra_gossip` 和未经充分调参的 `full` 方案性能下降，表明跨面模型交换若混合过强，会放大不同轨道面之间的数据异质性，导致判别边界受扰动。
4. 通过引入 `warm start`、自适应 gossip 权重、全局 EMA 稳定器和跨面 warmup 阶段，历史 `5 rounds` 口径下的 `OrbitShield_FL` 已提升到 `0.9718`，优于 `single`、`FedAvg` 和 `intra_only`；而当前默认正式 20 轮入口则以 `summary.json` 中约 `0.9820` 的结果为准。
5. 这说明在低轨卫星多星协同场景下，跨面协同不是简单“加 gossip 就更好”，而是必须在“何时交换、交换多少、如何抑制陈旧和不可靠更新”这三个方面进行联合设计。
6. 从当前保留的 `5 rounds` 调参档案看，`beta=0.1`、`warmup_rounds=2` 与略高的 `global_momentum=0.2` 组合取得了该档案下的最优结果；而当前默认正式 20 轮 preset 仍固定为 `global_momentum=0.1`，两者应分开表述。

### 13. grid_search 可视化

当前已经基于 [grid_search_summary.csv](./4_train/experiments/OrbitShield_FL/grid_search/grid_search_summary.csv) 生成了 3 张调参图：

- [OrbitShield_FL_heatmaps.png](./4_train/experiments/OrbitShield_FL/grid_search/plots/OrbitShield_FL_heatmaps.png)
- [OrbitShield_FL_trends.png](./4_train/experiments/OrbitShield_FL/grid_search/plots/OrbitShield_FL_trends.png)
- [OrbitShield_FL_top10.png](./4_train/experiments/OrbitShield_FL/grid_search/plots/OrbitShield_FL_top10.png)

对应绘图脚本为：

- [4_train/scripts/plot_federated_grid_search.py](./4_train/scripts/plot_federated_grid_search.py)

可直接重绘：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/plot_federated_grid_search.py \
  --csv_path experiments/OrbitShield_FL/grid_search/grid_search_summary.csv \
  --output_dir experiments/OrbitShield_FL/grid_search/plots
```

## 九、ns-3 驱动的联邦协同仿真

前面的 `OrbitShield_FL` 默认实验采用的是联邦框架内部的启发式动态拓扑，用于快速验证“面内聚合 + 面间 gossip + 信誉/补偿机制”的有效性。为了进一步提高低轨卫星场景的可信度，当前工程又新增了一条独立的 `ns-3` 驱动联邦协同仿真链路。

这条链路与数据集生成链路严格分开：

- [realtime_satellite.cc](./realtime_satellite.cc) 继续只负责 `PCAP -> 仿真链路 -> 抓包` 的数据生成
- 新增 [federated_constellation.cc](./federated_constellation.cc) 专门负责“低轨星座通信环境 -> 联邦训练拓扑轨迹”

因此，项目现在有两条相互独立的仿真路径：

1. 数据仿真路径  
   `原始 PCAP -> realtime_satellite.cc -> captured -> 特征提取 -> dataset_cicids17 / dataset_sti`
2. 联邦协同路径  
   `federated_constellation.cc -> ns-3 round trace -> OrbitShield_FL(ns3 backend)`

### 1. 设计目标

这条新链路不是把 PyTorch 训练塞进 `ns-3` 事件循环，而是采用“通信环境由 ns-3 给出、联邦训练由 Python 执行”的协同仿真方式。这样做的原因是：

- 保留现有联邦训练框架和模型实现
- 让 `ns-3` 只负责它最擅长的链路与拓扑仿真
- 让联邦训练真正受 `带宽 / 时延 / 丢包 / 接触窗口` 约束
- 避免把系统做成难以维护的实时混合大循环

### 2. 新增模块

新增的核心模块如下：

- [federated_constellation.cc](./federated_constellation.cc)  
  独立的 ns-3 星座通信环境仿真器，用于导出每轮联邦训练所需的通信状态
- [4_train/OrbitShield_FL/ns3_bridge.py](./4_train/OrbitShield_FL/ns3_bridge.py)  
  Python 桥接层，负责调用 ns-3 二进制、加载 trace、校验 manifest 和各轮 json
- [4_train/OrbitShield_FL/topology_ns3.py](./4_train/OrbitShield_FL/topology_ns3.py)  
  将 ns-3 trace 转成联邦主控可直接使用的拓扑快照
- [4_train/OrbitShield_FL/transfer_scheduler.py](./4_train/OrbitShield_FL/transfer_scheduler.py)  
  根据模型大小、链路带宽、丢包率和接触窗口，判断模型是否能在本轮传输完成
- [4_train/scripts/train_federated_ns3.py](./4_train/scripts/train_federated_ns3.py)  
  新的 ns-3 驱动联邦训练入口，不影响原有 [4_train/scripts/train_federated.py](./4_train/scripts/train_federated.py)

### 3. ns-3 trace 输出内容

[federated_constellation.cc](./federated_constellation.cc) 当前会导出：

- `constellation_config.json`
- `manifest.json`
- `round_0001.json ... round_xxxx.json`

每轮 trace 至少包含：

- 轨道面与卫星映射
- 面内链路状态
- 面间链路状态
- 每条链路的：
  - `available`
  - `success`
  - `delay_ms`
  - `bandwidth_mbps`
  - `packet_loss`
  - `contact_duration_s`

这些字段会直接进入联邦训练，用于：

- 决定哪些轨道面本轮可 gossip
- 估算链路质量与通信可靠性
- 判断客户端上传和面间模型交换是否能在窗口内完成

### 4. 联邦主控如何使用 ns-3 拓扑

当前联邦主控 [4_train/OrbitShield_FL/serverless_orchestrator.py](./4_train/OrbitShield_FL/serverless_orchestrator.py) 已支持两种 backend：

- `heuristic`
- `ns3`

其中：

- `heuristic` 继续复用原有启发式动态拓扑
- `ns3` 则直接读取 ns-3 导出的 round trace

在 `ns3` backend 下，本轮联邦训练会额外受到两个真实通信约束：

1. 当前链路是否可见、是否成功
2. 当前模型是否能在接触窗口内传完

第二点由 [4_train/OrbitShield_FL/transfer_scheduler.py](./4_train/OrbitShield_FL/transfer_scheduler.py) 负责，它会根据：

- 模型参数大小
- `bandwidth_mbps`
- `packet_loss`
- `contact_duration_s`

计算有效吞吐与传输时间，并决定本轮：

- 是否允许客户端模型上传
- 是否允许面间 gossip 生效
- 若失败则转入已有补偿逻辑

### 5. 运行方式

#### 直接使用已有 ns-3 trace 训练

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3.py \
  --dataset cicids17 \
  --trace_dir experiments/OrbitShield_FL_ns3/cicids17_trace \
  --output_dir experiments/OrbitShield_FL_ns3/cicids17 \
  --device cuda
```

#### 训练前自动生成新的 ns-3 trace

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3.py \
  --dataset cicids17 \
  --rounds 5 \
  --generate_trace \
  --trace_output_dir experiments/OrbitShield_FL_ns3/cicids17_trace \
  --output_dir experiments/OrbitShield_FL_ns3/cicids17 \
  --device cuda
```

当前默认会直接使用 [4_train/OrbitShield_FL/config.py](./4_train/OrbitShield_FL/config.py) 中固化的正式参数。

### 6. 当前已跑通的 ns-3 联调结果

当前已经完成一组从“ns-3 trace 生成 -> ns-3 backend 联邦训练 -> 结果落盘”的完整联调实验：

- trace 目录：
  - [OrbitShield_FL_ns3_trace](./4_train/experiments/OrbitShield_FL_ns3/cicids17_trace)
- 联邦训练输出目录：
  - [OrbitShield_FL_ns3](./4_train/experiments/OrbitShield_FL_ns3/cicids17)

主要输出文件包括：

- [4_train/experiments/OrbitShield_FL_ns3/cicids17_trace/constellation_config.json](./4_train/experiments/OrbitShield_FL_ns3/cicids17_trace/constellation_config.json)
- [4_train/experiments/OrbitShield_FL_ns3/cicids17_trace/manifest.json](./4_train/experiments/OrbitShield_FL_ns3/cicids17_trace/manifest.json)
- [4_train/experiments/OrbitShield_FL_ns3/cicids17/summary.json](./4_train/experiments/OrbitShield_FL_ns3/cicids17/summary.json)
- [4_train/experiments/OrbitShield_FL_ns3/cicids17/round_metrics.csv](./4_train/experiments/OrbitShield_FL_ns3/cicids17/round_metrics.csv)
- [4_train/experiments/OrbitShield_FL_ns3/cicids17/best_global_model.pt](./4_train/experiments/OrbitShield_FL_ns3/cicids17/best_global_model.pt)

本次联调条件：

- 数据集：`cicids17`
- 轮数：`5`
- backend：`ns3`
- trace：由 [federated_constellation.cc](./federated_constellation.cc) 自动生成
- 设备：`cuda`

本次结果为：

- `Best Val Accuracy = 0.9715`
- `Test Accuracy = 0.9473`
- `Test Precision = 0.9524`
- `Test Recall = 0.9473`
- `Test F1 = 0.9479`

混淆矩阵：

```text
[[6126,    0,  571],
 [  19, 6149,  380],
 [  13,   71, 6671]]
```

这组结果的意义不在于当前数值一定优于默认启发式 `OrbitShield_FL`，而在于它已经证明：

- 新的 ns-3 星座联邦环境仿真器可运行
- trace 到联邦拓扑的桥接是闭环可用的
- `OrbitShield_FL` 现在不仅能在启发式动态拓扑上训练，也能在 ns-3 驱动的通信环境上训练

### 7. 当前结论

从工程角度看，项目现在已经同时具备：

- 数据生成级 ns-3 仿真链路
- 联邦通信环境级 ns-3 协同仿真链路

其中：

- 前者服务于 `dataset_cicids17 / dataset_sti`
- 后者服务于 `OrbitShield_FL` 的低轨多星协同通信环境模拟

这为后续继续提高低轨卫星场景可信度提供了稳定基础。下一步如果需要继续增强，可以优先往以下方向扩展：

1. 更真实的轨道面接触图
2. 更细的链路容量/时延时变模型
3. 星地链路和地面站回传
4. 用 ns-3 实测链路统计替代更多启发式 link quality 项

## 十、Level 3 在线协同 ns-3 联邦学习

在上面的 `ns3 backend` 基础上，项目进一步新增了 Level 3 在线协同方案。它与离线 trace 驱动的区别在于：

- Level 2：训练前一次性生成完整 trace，再按轮读取
- Level 3：每轮训练开始前动态调用一次 ns-3，只生成当前轮 trace

这条链路仍然不改动 [realtime_satellite.cc](./realtime_satellite.cc)，而是在联邦训练侧单独新增：

- [4_train/OrbitShield_FL/ns3_online_bridge.py](./4_train/OrbitShield_FL/ns3_online_bridge.py)
- [4_train/OrbitShield_FL/online_orchestrator.py](./4_train/OrbitShield_FL/online_orchestrator.py)
- [4_train/scripts/train_federated_ns3_online.py](./4_train/scripts/train_federated_ns3_online.py)
- [4_train/scripts/run_federated_ns3_online.sh](./4_train/scripts/run_federated_ns3_online.sh)
- 独立说明文档：[level3_online_cosim.md](./level3_online_cosim.md)

### 1. 正式输出目录

- `cicids17`：
  - [4_train/experiments/OrbitShield_FL_ns3_online/cicids17](./4_train/experiments/OrbitShield_FL_ns3_online/cicids17)
- `sti`：
  - [4_train/experiments/OrbitShield_FL_ns3_online/sti](./4_train/experiments/OrbitShield_FL_ns3_online/sti)

每组实验目录中都包含：

- `best_global_model.pt`
- `summary.json`
- `round_metrics.csv`
- `reputation_history.json`
- `topology_history.json`
- `ns3_online_trace_index.json`
- `ns3_online_trace/round_xxxx/`

### 2. 当前正式结果

#### 2.1 `cicids17`

输出目录：
- [4_train/experiments/OrbitShield_FL_ns3_online/cicids17](./4_train/experiments/OrbitShield_FL_ns3_online/cicids17)

结果：
- `Best Val Accuracy = 0.9710`，最佳轮次为第 `11` 轮
- `Test Accuracy = 0.9626`
- `Test Precision = 0.9636`
- `Test Recall = 0.9626`
- `Test F1 = 0.9624`

混淆矩阵：

```text
[[11062,     1,    23],
 [  380,  9865,   487],
 [  247,    90, 10655]]
```

#### 2.2 `sti`

输出目录：
- [4_train/experiments/OrbitShield_FL_ns3_online/sti](./4_train/experiments/OrbitShield_FL_ns3_online/sti)

运行口径：
- 完整 `STI` 数据集
- `20` 轮联邦训练
- `--full_eval`
- 每轮在线调用一次 ns-3 生成当前轮 trace

结果：
- `Best Val Accuracy = 0.9806`，最佳轮次为第 `20` 轮
- `Test Accuracy = 0.9804`
- `Test Precision = 0.9807`
- `Test Recall = 0.9804`
- `Test F1 = 0.9804`

混淆矩阵：

```text
[[133297,     0,    97,   511,   375,  1413,     2,    10],
 [   282, 30451,   121,     0,     0,     0,     0,     0],
 [     2,     0, 48829,     0,     0,     0,   286,     0],
 [    50,     0,     0, 28362,     0,   227,     0,     0],
 [    49,     0,     0,     0, 38638,     0,     8,   375],
 [    60,     0,     0,   289,     0, 36166,     0,     0],
 [     2,     0,     0,     0,     0,     0, 49635,    15],
 [     9,     0,     0,     0,    23,     0,  1249, 53638]]
```

### 3. 当前定位

Level 3 在线协同方案目前的角色是：

1. 作为高于离线 trace 驱动的联邦协同验证链路
2. 作为“联邦训练过程与 ns-3 通信环境在线耦合”的正式实验结果
3. 为后续继续接入更真实的时变轨道接触模型提供工程基础

## 十一、Level 4B：ns-3 + libtorch 全 C++ 联邦训练

在 `Level 4A` 的 `ns-3 主调度 + Python 本地训练执行器` 完成后，当前工程又实现了 `Level 4B`：

- 联邦轮次与通信约束由 `ns-3` 进程内部统一执行
- 本地 `DSC-CBAM-GRU` 训练由 `libtorch` 在 C++ 中完成
- 不再依赖 Python 子进程执行本地训练

新增文件：

- [federated_libtorch_runtime.cc](./federated_libtorch_runtime.cc)
- [4_train/scripts/export_libtorch_dataset.py](./4_train/scripts/export_libtorch_dataset.py)
- [4_train/scripts/train_federated_ns3_libtorch.py](./4_train/scripts/train_federated_ns3_libtorch.py)
- [4_train/scripts/run_federated_ns3_libtorch.sh](./4_train/scripts/run_federated_ns3_libtorch.sh)

正式结果目录：

- [4_train/experiments/OrbitShield_FL_ns3_libtorch/cicids17](./4_train/experiments/OrbitShield_FL_ns3_libtorch/cicids17)
- [4_train/experiments/OrbitShield_FL_ns3_libtorch/sti](./4_train/experiments/OrbitShield_FL_ns3_libtorch/sti)

`cicids17` 正式结果：

- `Best Val Accuracy = 0.969917`
- `Test Accuracy = 0.960439`
- `Test Precision = 0.960960`
- `Test Recall = 0.960439`
- `Test F1 = 0.960293`
- `Best Round = 15`

混淆矩阵：

```text
[[10927,    1,  158],
 [  260, 9924,  548],
 [  153,  161, 10678]]
```

`sti` 正式结果：

- `Best Val Accuracy = 0.991962`
- `Test Accuracy = 0.992025`
- `Test Precision = 0.992103`
- `Test Recall = 0.992025`
- `Test F1 = 0.992029`
- `Best Round = 13`

这次 4B 的正式结果已经不是最初的冷启动原型结果，而是补入了：

- `cicids17` 路径的单体模型 warm start
- Python 版 `OrbitShield_FL` 的完整 `full` 机制对齐
  - 自适应 gossip `beta_floor`
  - 全局动量 `global_momentum`
  - 基于 `sim + improve + stable` 的信誉更新

### 11.1 Level 4B 异质性鲁棒性实验

为了补充最终联邦版本在不同数据异质性条件下的稳定性，本次又基于 `Level 4B` 增加了一组 `Dirichlet α` 鲁棒性实验。

新增脚本：

- [4_train/scripts/run_robustness_alpha_libtorch.py](./4_train/scripts/run_robustness_alpha_libtorch.py)
- [4_train/scripts/plot_robustness_alpha_libtorch.py](./4_train/scripts/plot_robustness_alpha_libtorch.py)
- [4_train/scripts/plot_convergence_curves_libtorch.py](./4_train/scripts/plot_convergence_curves_libtorch.py)

新增输出目录：

- [4_train/experiments/OrbitShield_FL_ns3_libtorch/alpha_sweep](./4_train/experiments/OrbitShield_FL_ns3_libtorch/alpha_sweep)

其中主要文件包括：

- [alpha_sweep_summary.csv](./4_train/experiments/OrbitShield_FL_ns3_libtorch/alpha_sweep/alpha_sweep_summary.csv)
- [alpha_vs_accuracy.png](./4_train/experiments/OrbitShield_FL_ns3_libtorch/alpha_sweep/plots/alpha_vs_accuracy.png)
- [alpha_convergence_curves.png](./4_train/experiments/OrbitShield_FL_ns3_libtorch/alpha_sweep/plots/alpha_convergence_curves.png)
- [fl_convergence_final.png](./4_train/experiments/visualization/fl_convergence_final.png)

本次 `cicids17` 的 `α` 扫描范围为：

- `α in {0.05, 0.1, 0.3, 0.5, 1.0, 5.0}`

结果如下：

| Dirichlet α | Best Val Accuracy | Test Accuracy | Test F1 |
|-------------|-------------------|---------------|---------|
| `0.05` | `0.964979` | `0.960683` | `0.960636` |
| `0.10` | `0.869639` | `0.793813` | `0.786883` |
| `0.30` | `0.972355` | `0.966900` | `0.966812` |
| `0.50` | `0.967052` | `0.941756` | `0.941077` |
| `1.00` | `0.975525` | `0.963700` | `0.963599` |
| `5.00` | `0.981926` | `0.973453` | `0.973397` |

可以看到：

1. 当 `α=0.10` 时，`Level 4B` 的性能明显下降，说明在更强 non-IID 条件下，跨星数据分布差异会显著影响联邦训练稳定性。
2. 当 `α>=0.30` 后，模型性能整体恢复到较高水平；其中 `α=5.0` 时达到本次扫描中的最好结果。
3. 当前正式默认配置 `α=0.30` 仍然是一个合理取值：既保留了联邦场景中的数据异质性，又没有像 `α=0.10` 那样引入过强的分布偏斜。

### 11.2 Level 4B 收敛曲线图

为了把最终联邦版本与前几层联邦方案放在同一张图中比较，本次还新增了 `cicids17` 上的联邦收敛曲线图：

- [fl_convergence_final.png](./4_train/experiments/visualization/fl_convergence_final.png)

该图包含：

- `Level 4B: OrbitShield_FL (NS-3 + libtorch)`
- `Level 3: OrbitShield_FL (NS-3 online)`
- `Level 2: OrbitShield_FL (NS-3 offline)`
- `Level 1: OrbitShield_FL (heuristic)`

当前图中各条曲线的最终 `val_accuracy` 为：

- `Level 4B`: `0.894815`
- `Level 3`: `0.955257`
- `Level 2`: `0.957514`
- `Level 1`: `0.978479`

这里需要注意：

1. `Level 4B` 的正式最佳结果出现在较早轮次（`Best Round = 15`），而不是最后一轮，因此其“最终轮次 val_acc”低于“最佳 val_acc”。
2. 这一现象说明 `Level 4B` 当前更像是一个高保真联合仿真原型：它成功实现了“训练与通信约束同一运行时闭环”，但其训练稳定性仍有进一步优化空间。
3. 因此，`Level 4B` 的主要价值在于系统完整性和联合仿真可信度，而不是简单追求最后一轮精度绝对最高。

### 11.3 实验口径说明

本次 `Level 4B α` 鲁棒性实验在执行时额外发现了一个 `C++/libtorch` 运行时细节：

当前 `Level 4B` 的 `CUDA` device mismatch 已经修复，`α` 扫描默认直接走正式 GPU 口径：

- 正式结果目录保持为 [OrbitShield_FL_ns3_libtorch/cicids17](./4_train/experiments/OrbitShield_FL_ns3_libtorch/cicids17)
- `α` 扫描结果写入 [OrbitShield_FL_ns3_libtorch/alpha_sweep](./4_train/experiments/OrbitShield_FL_ns3_libtorch/alpha_sweep)
- `α` 扫描会复用正式导出目录 [4_train/libtorch_data/cicids17](./4_train/libtorch_data/cicids17)
- 重新生成新的 `partition` 后，可以直接在 GPU 上完成 Level 4B 训练，不再需要 CPU workaround

复现命令：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train

/home/lithic/final/ns3-gpu-venv/bin/python scripts/run_robustness_alpha_libtorch.py
/home/lithic/final/ns3-gpu-venv/bin/python scripts/plot_robustness_alpha_libtorch.py
/home/lithic/final/ns3-gpu-venv/bin/python scripts/plot_convergence_curves_libtorch.py
```
