# 复现流程

本文档只描述当前正式实验口径的完整复现方式。所有结果均围绕同一主线展开：

- 本文模型：单向 GRU 版 `DSC-CBAM-GRU`
- 正式压缩/部署口径：结构化压缩模型（结构化压缩 + 短程微调 + `fbgemm` 动态INT8）
- 联邦学习主模型：仍使用未压缩本文模型

## 1. 环境准备

输入目录：

- `ns-3` 工程根目录：`/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1`
- Python GPU 虚拟环境：`/home/lithic/final/ns3-gpu-venv`

系统依赖：

- `cmake`
- `python3`
- `bsdtar`（用于读取 `STI` 的 `.rar`）
- `tcpdump`
- `tcpreplay`
- `ip` / `iproute2`
- `sudo` / root 权限（用于 TAP 接口与实时仿真抓包链路）

Python 依赖准备：

```bash
source /home/lithic/final/ns3-gpu-venv/bin/activate
pip install -r /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train/requirements.txt
```

输出目录：

- `ns-3` 构建目录：`/home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/build`

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1
source /home/lithic/final/ns3-gpu-venv/bin/activate

cmake -S . -B build
cmake --build build --target scratch_06_realtime_emulation_realtime_satellite -j"$(nproc)"
cmake --build build --target scratch_06_realtime_emulation_federated_constellation -j"$(nproc)"
cmake --build build --target scratch_06_realtime_emulation_federated_libtorch_runtime -j"$(nproc)"
```

说明：

- 若在 WSL 下看到 `appendWindowsPath = false` 提示，这是 `ns-3` 上游的环境建议，不影响本项目三个 scratch target 生成。
- `GTK3`、`Python development`、`eigen3` 缺失提示来自 `ns-3` 对可选组件的探测；当前正式复现不依赖这些可选组件。
- 若使用 Python wheel 自带的 libtorch，可能仍会看到 `kineto_LIBRARY-NOTFOUND`、`Failed to compute shorthash for libnvrtc.so` 等 PyTorch/CMake warning；当前不阻塞正式 target 的构建。

原始数据输入：

- `CICIDS2017` 原始 PCAP 放到 `/home/lithic/final/data`
- `STI` 的 8 个 `.rar` 放到 [STI_dataset](./STI_dataset)

项目根目录：

- [06_realtime_emulation](./)

## 2. 复现 `cicids17` 数据集与本文单体模型

输入目录：

- 原始 PCAP：`/home/lithic/final/data`
- 项目目录：[06_realtime_emulation](./)

输出目录：

- 分片 PCAP：[fragments_window](./fragments_window)
- 仿真抓包：[captured_window](./captured_window)
- 特征数据集：[dataset_cicids17](./dataset_cicids17)
- 本文单体模型：[4_train/checkpoints_gru_formal_tuned/cicids17_gru_best.pt](./4_train/checkpoints_gru_formal_tuned/cicids17_gru_best.pt)

从原始 PCAP 完整生成 `cicids17` 数据集并训练本文单体模型：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation
export PATH="/home/lithic/final/ns3-gpu-venv/bin:$PATH"
/home/lithic/final/ns3-gpu-venv/bin/python -c "import scapy"
MAX_PACKETS=50000 sudo -E bash ./run_all_window.sh
```

如果只想重跑本文单体训练：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
./run_train.sh
```

当前本文模型口径：

- `hidden_dim = 32`
- `dropout = 0.4`
- `conv_dim = 16`
- `dsc_dim = 48`
- `bidirectional = False`
- `lr = 0.0003`
- `weight_decay = 0.01`

## 3. 复现 `STI` 数据集与本文单体模型

输入目录：

- [STI_dataset](./STI_dataset)

输出目录：

- [dataset_sti](./dataset_sti)
- [4_train/checkpoints_gru_formal_tuned/sti_gru_best.pt](./4_train/checkpoints_gru_formal_tuned/sti_gru_best.pt)

先生成 `STI` 的 `npz` 数据：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation
/home/lithic/final/ns3-gpu-venv/bin/python 3_prepare_sti_dataset.py
```

训练 `STI` 本文单体模型：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_gru.py --dataset sti --device cuda --hidden_dim 32 --dropout 0.4 --conv_dim 16 --dsc_dim 48 --lr 0.0003 --weight_decay 0.01 --epochs 100 --batch_size 128 --early_stopping_patience 10 --no-bidirectional --output_dir checkpoints_gru_formal_tuned
```

## 4. 复现正式单机模型对比与消融

正式输出目录：

- 对比：[4_train/experiments/comparison_formal_tuned](./4_train/experiments/comparison_formal_tuned)
- 消融：[4_train/experiments/ablation_formal_tuned](./4_train/experiments/ablation_formal_tuned)

正式模型对比：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/run_comparison.py --output_dir experiments/comparison_formal_tuned --include_traditional --device cuda
```

正式消融：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
./run_ablation.sh --output_dir experiments/ablation_formal_tuned --comparison_config experiments/comparison_formal_tuned/comparison_results.json
```

## 5. 复现正式压缩 / 部署口径

正式压缩口径不是旧的“非结构化剪枝 + 动态INT8”，而是：

- 结构化压缩
- 短程微调
- `fbgemm` 动态INT8量化

正式结果目录：

- [4_train/experiments/compression/structured_candidates_formal_tuned](./4_train/experiments/compression/structured_candidates_formal_tuned)

其中最终正式压缩结果由：
- [structured_formal_summary.json](./4_train/experiments/compression/structured_candidates_formal_tuned/structured_formal_summary.json)

给出，当前正式压缩模型已满足参数量、CPU时延和精度三项验收条件。

## 6. 复现 `OrbitShield_FL` Level 1 正式结果

正式输出目录：

- `cicids17`：[4_train/experiments/OrbitShield_FL_formal_tuned/cicids17](./4_train/experiments/OrbitShield_FL_formal_tuned/cicids17)
- `sti`：[4_train/experiments/OrbitShield_FL_formal_tuned/sti](./4_train/experiments/OrbitShield_FL_formal_tuned/sti)

`cicids17`：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated.py --dataset cicids17 --device cuda --output_dir experiments/OrbitShield_FL_formal_tuned/cicids17 --init_checkpoint checkpoints_gru_formal_tuned/cicids17_gru_best.pt --hidden_dim 32 --dropout 0.4 --conv_dim 16 --dsc_dim 48
```

`sti`：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated.py --dataset sti --device cuda --output_dir experiments/OrbitShield_FL_formal_tuned/sti --init_checkpoint checkpoints_gru_formal_tuned/sti_gru_best.pt --hidden_dim 32 --dropout 0.4 --conv_dim 16 --dsc_dim 48 --full_eval
```

## 7. 复现 `OrbitShield_FL + ns-3` Level 2 正式结果

正式输出目录：

- `cicids17`：[4_train/experiments/OrbitShield_FL_ns3_formal_tuned/cicids17](./4_train/experiments/OrbitShield_FL_ns3_formal_tuned/cicids17)
- `sti`：[4_train/experiments/OrbitShield_FL_ns3_formal_tuned/sti](./4_train/experiments/OrbitShield_FL_ns3_formal_tuned/sti)
- trace 目录继续复用已有正式轨迹目录：[4_train/experiments/OrbitShield_FL_ns3/cicids17_trace](./4_train/experiments/OrbitShield_FL_ns3/cicids17_trace)、[4_train/experiments/OrbitShield_FL_ns3/sti_trace](./4_train/experiments/OrbitShield_FL_ns3/sti_trace)

`cicids17`：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3.py \
  --dataset cicids17 \
  --trace_dir experiments/OrbitShield_FL_ns3/cicids17_trace \
  --output_dir experiments/OrbitShield_FL_ns3_formal_tuned/cicids17 \
  --device cuda \
  --init_checkpoint checkpoints_gru_formal_tuned/cicids17_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```

`sti`：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3.py \
  --dataset sti \
  --trace_dir experiments/OrbitShield_FL_ns3/sti_trace \
  --output_dir experiments/OrbitShield_FL_ns3_formal_tuned/sti \
  --device cuda \
  --init_checkpoint checkpoints_gru_formal_tuned/sti_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```

## 8. 复现 `OrbitShield_FL + ns-3 online` Level 3 正式结果

正式输出目录：

- `cicids17`：[4_train/experiments/OrbitShield_FL_ns3_online_formal_tuned/cicids17](./4_train/experiments/OrbitShield_FL_ns3_online_formal_tuned/cicids17)
- `sti`：[4_train/experiments/OrbitShield_FL_ns3_online_formal_tuned/sti](./4_train/experiments/OrbitShield_FL_ns3_online_formal_tuned/sti)

`cicids17`：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3_online.py \
  --dataset cicids17 \
  --rounds 20 \
  --output_dir experiments/OrbitShield_FL_ns3_online_formal_tuned/cicids17 \
  --device cuda \
  --init_checkpoint checkpoints_gru_formal_tuned/cicids17_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```

`sti`：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3_online.py \
  --dataset sti \
  --rounds 20 \
  --full_eval \
  --output_dir experiments/OrbitShield_FL_ns3_online_formal_tuned/sti \
  --device cuda \
  --init_checkpoint checkpoints_gru_formal_tuned/sti_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```

## 9. 复现 `Level 4B: ns-3 + libtorch` 全 C++ 联邦训练

正式输出目录：

- `cicids17`：[4_train/experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/cicids17](./4_train/experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/cicids17)
- `sti`：[4_train/experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/sti](./4_train/experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/sti)

`cicids17`：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3_libtorch.py \
  --dataset cicids17 \
  --rounds 20 \
  --local_epochs 1 \
  --batch_size 512 \
  --device cuda \
  --output_dir experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/cicids17 \
  --init_checkpoint checkpoints_gru_formal_tuned/cicids17_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```

`sti`：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3_libtorch.py \
  --dataset sti \
  --rounds 20 \
  --local_epochs 1 \
  --batch_size 512 \
  --device cuda \
  --output_dir experiments/OrbitShield_FL_ns3_libtorch_formal_tuned/sti \
  --init_checkpoint checkpoints_gru_formal_tuned/sti_gru_best.pt \
  --hidden_dim 32 \
  --dropout 0.4 \
  --conv_dim 16 \
  --dsc_dim 48
```
