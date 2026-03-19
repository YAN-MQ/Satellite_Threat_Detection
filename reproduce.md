# 复现流程

## 1. 环境准备

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1
source /home/lithic/final/ns3-gpu-venv/bin/activate

cmake -S . -B build
cmake --build build --target scratch_06_realtime_emulation_realtime_satellite -j"$(nproc)"
cmake --build build --target scratch_06_realtime_emulation_federated_constellation -j"$(nproc)"
```

原始数据准备：

- `CICIDS2017` 原始 PCAP 放到 `/home/lithic/final/data`
- `STI` 的 8 个 `.rar` 放到 [STI_dataset](./STI_dataset)

项目根目录：

- [06_realtime_emulation](./)

## 2. 复现 `cicids17` 数据集与单体模型

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation
```

从原始 PCAP 完整生成 `cicids17` 数据集并训练单体模型：

```bash
export PATH="/home/lithic/final/ns3-gpu-venv/bin:$PATH"
MAX_PACKETS=50000 sudo -E bash ./run_all_window.sh
```

这一步会依次完成：

1. 攻击时间窗口提取与分片
2. `ns-3` 实时仿真
3. 抓包
4. 特征提取
5. 单体 `DSC-CBAM-GRU` 训练

主要输出：

- 数据集：[dataset_cicids17](./dataset_cicids17)
- 单体模型：[cicids17_gru.pt](./4_train/checkpoints_gru/cicids17_gru.pt)

如果只想重跑单体训练：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
./run_train.sh
```

## 3. 复现 `STI` 数据集与单体模型

先生成 `STI` 的 `npz` 数据：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation
python3 3_prepare_sti_dataset.py
```

输出：

- [dataset_sti](./dataset_sti)

训练 `STI` 单体模型：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
python scripts/train_gru.py --dataset sti --device cuda
```

输出模型：

- [sti_gru_best.pt](./4_train/checkpoints_gru/sti_gru_best.pt)

## 4. 复现 `OrbitShield_FL` 联邦学习

复现正式 `cicids17` 联邦结果：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
./scripts/run_federated.sh
```

这条命令默认就是正式最优 `full` 配置，不需要额外传联邦超参数。

正式输出目录：

- [cicids17](./4_train/experiments/OrbitShield_FL/cicids17)

如果要复现 `STI` 联邦结果：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
python scripts/train_federated.py --dataset sti --method full --device cuda
```

正式输出目录：

- [sti](./4_train/experiments/OrbitShield_FL/sti)

如果要复现联邦基线对比：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
./scripts/run_federated_ablation.sh
```

输出目录：

- [baselines](./4_train/experiments/OrbitShield_FL/baselines)

## 5. 复现 `OrbitShield_FL + ns-3` 联邦学习

使用已有 trace 训练：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
python scripts/train_federated_ns3.py \
  --dataset cicids17 \
  --trace_dir experiments/OrbitShield_FL_ns3/cicids17_trace \
  --output_dir experiments/OrbitShield_FL_ns3/cicids17 \
  --device cuda
```

重新生成 trace 再训练：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
python scripts/train_federated_ns3.py \
  --dataset cicids17 \
  --rounds 20 \
  --generate_trace \
  --trace_output_dir experiments/OrbitShield_FL_ns3/cicids17_trace \
  --output_dir experiments/OrbitShield_FL_ns3/cicids17 \
  --device cuda
```

正式输出：

- 结果目录：[cicids17](./4_train/experiments/OrbitShield_FL_ns3/cicids17)
- trace 目录：[cicids17_trace](./4_train/experiments/OrbitShield_FL_ns3/cicids17_trace)

## 6. 复现消融、对比、压缩与可视化

在 [4_train](./4_train) 下执行：

消融：

```bash
./run_ablation.sh
```

对比：

```bash
./run_comparison.sh
```

单体模型调参：

```bash
./run_tune_full.sh
```

剪枝：

```bash
python scripts/prune_model.py --device cuda
```

量化：

```bash
python scripts/quantize_model.py --device cuda
```

t-SNE：

```bash
python scripts/plot_tsne.py --device cuda
```

输出根目录统一在：

- [experiments](./4_train/experiments)
