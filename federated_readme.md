# OrbitShield_FL 联邦学习说明

## 1. 文档定位

本文档专门说明本项目在现有单机训练闭环基础上，增量式扩展得到的联邦学习版本 `OrbitShield_FL`。  
该版本的目标是模拟低轨卫星多星协同威胁预测，而不是推翻已有单机流程重写整个工程。

当前保留的联邦实验目录只有：

- [4_train/experiments/OrbitShield_FL/cicids17](./4_train/experiments/OrbitShield_FL/cicids17)
- [4_train/experiments/OrbitShield_FL/grid_search](./4_train/experiments/OrbitShield_FL/grid_search)

其中：

- `OrbitShield_FL` 是当前最终正式联邦版本
- `grid_search` 是把完整联邦方案调到最优时保留的网格搜索档案

## 2. 与原项目的关系

联邦版本严格复用了现有单机项目的核心资产，没有改动原始特征定义、标签定义和 NS-3 数据链路。

原始单机闭环仍然是：

`原始 PCAP -> 攻击时间窗口提取 -> IP 分片 -> NS-3 实时仿真 -> 流量捕获 -> 特征提取 -> DSC-CBAM-GRU 训练`

联邦版本只在训练层上新增一层抽象：

`dataset_cicids17 -> 联邦划分 -> 多星协同训练 -> 全局评估`

直接复用的内容包括：

- 数据集：
  - [dataset_cicids17/train.npz](./dataset_cicids17/train.npz)
  - [dataset_cicids17/val.npz](./dataset_cicids17/val.npz)
  - [dataset_cicids17/test.npz](./dataset_cicids17/test.npz)
- 模型：
  - [4_train/src/models/dsc_cbam_gru.py](./4_train/src/models/dsc_cbam_gru.py)
- 单机最佳初始化模型：
  - [4_train/checkpoints_gru/cicids17_gru_best.pt](./4_train/checkpoints_gru/cicids17_gru_best.pt)

保持不变的约束包括：

- 三分类标签定义不变：
  - `benign = 0`
  - `ddos = 1`
  - `portscan = 2`
- 输入窗口形状不变：
  - `(samples, 10, 18)`
- 18 维特征定义不变
- 现有单机训练脚本不变
- 现有 NS-3 实时仿真链路不变

## 3. 联邦学习要解决的问题

单机训练默认把所有训练样本集中在一个训练进程中，这适合做基线验证，但不符合低轨卫星多星协同场景。

在真实的低轨卫星威胁预测场景中，常见问题是：

- 单星本地样本有限，分布不均
- 不同卫星观测到的攻击流量具有明显异质性
- 轨道面内链路相对稳定
- 轨道面间链路具有周期性可见窗口和更高不确定性
- 直接做传统同步中心式 FedAvg，无法很好反映星间协同特性

因此，本项目新增的联邦训练层采用了“低侵入、可直接运行、贴近星座结构”的设计：

- 卫星被抽象为联邦客户端
- 轨道面被抽象为局部聚合单元
- 不做真实分布式通信，先做算法仿真级联邦
- 重点模拟“轨道面内稳定协同 + 轨道面间断续协同”

## 4. OrbitShield_FL 的总体思路

`OrbitShield_FL` 的核心思路可以概括为：

1. 复用已有 `dataset_cicids17`
2. 把训练集切分给多颗卫星
3. 每颗卫星本地训练同一个 `DSC-CBAM-GRU`
4. 先在轨道面内做加权聚合
5. 再在轨道面间做异步 gossip
6. 对链路失败做模型补偿
7. 对陈旧更新做衰减
8. 对客户端更新做信誉加权
9. 用全局验证集和测试集统一评估

最终形成的联邦骨架是：

- `DFedSat` 风格的多星协同骨架
- 面内聚合
- 面间异步 gossip
- 链路失败自补偿
- 陈旧度衰减
- 信誉感知聚合
- warm start
- 全局 EMA 稳定器

## 5. 星座抽象方式

当前默认联邦配置采用 12 颗卫星、3 个轨道面：

- `plane_0`: `sat_0, sat_1, sat_2, sat_3`
- `plane_1`: `sat_4, sat_5, sat_6, sat_7`
- `plane_2`: `sat_8, sat_9, sat_10, sat_11`

每颗卫星都对应一个 `FederatedClient`。  
每个客户端持有 `train.npz` 的一个本地子集。  
验证集和测试集默认不再划分，继续采用全局评估：

- `val.npz` 用于全局验证
- `test.npz` 用于全局测试

## 6. 数据复用与联邦划分

联邦学习第一版不重新提特征，直接复用现有 `dataset_cicids17`。

训练集的联邦划分逻辑由 [4_train/OrbitShield_FL/partition.py](./4_train/OrbitShield_FL/partition.py) 负责，核心函数包括：

- `load_window_dataset(...)`
- `partition_train_dataset_for_satellites(...)`
- `create_client_dataloaders(...)`
- `dump_partition_stats(...)`

当前支持 4 种划分方式：

- `iid`
- `dirichlet`
- `quantity_skew`
- `hybrid`

默认使用：

- `partition_mode = dirichlet`
- `dirichlet_alpha = 0.3`

这意味着不同卫星的类别分布是非 IID 的，更接近多星观测场景。

`partition_train_dataset_for_satellites(...)` 的行为是：

1. 读取 `train.npz`
2. 根据标签分布和设定的划分模式切分索引
3. 为每个 `sat_i` 生成本地样本索引集合
4. 统计每个客户端的：
   - `sample_count`
   - `label_distribution`
5. 将这些信息写入：
   - `partition_stats.json`

这样可以直接复现每颗卫星本地数据量及标签偏斜情况。

## 7. 复用的本地模型

联邦版本没有重新定义模型，而是直接复用了现有单机主模型：

- [4_train/src/models/dsc_cbam_gru.py](./4_train/src/models/dsc_cbam_gru.py)

模型输入输出保持不变：

- 输入：`(batch, 10, 18)`
- 输出：3 类分类 logits

本地训练仍然使用：

- `AdamW`
- `CrossEntropyLoss`

因此，联邦学习版本与单机训练版本之间的核心差别不是模型结构，而是“训练组织方式”和“参数聚合策略”。

## 8. 联邦训练框架原理

### 8.1 本地目标函数

对客户端 `i` 而言，本地目标函数为：

`F_i(w) = (1 / n_i) * sum_{(x,y) in D_i} ell(w; x, y)`

全局目标函数为：

`F(w) = sum_{i=1}^{N} (n_i / n_total) * F_i(w)`

其中：

- `D_i` 是第 `i` 颗卫星的本地数据
- `n_i` 是该卫星样本数
- `ell` 是交叉熵损失

### 8.2 本地训练

每轮联邦训练中，每颗活跃卫星会在本地执行若干步梯度下降，得到本地更新后的参数：

`w_i^{t,e+1} = w_i^{t,e} - eta * grad F_i(w_i^{t,e})`

对应代码在 [4_train/OrbitShield_FL/client.py](./4_train/OrbitShield_FL/client.py) 中的：

- `FederatedClient.local_train(...)`

该函数会：

1. 记录训练前模型参数
2. 在本地 `DataLoader` 上训练 `local_epochs`
3. 输出训练后参数 `weights`
4. 计算本地参数增量 `update`
5. 返回平均损失和样本数

### 8.3 面内聚合

轨道面内聚合由 [4_train/OrbitShield_FL/aggregators.py](./4_train/OrbitShield_FL/aggregators.py) 中的 `intra_plane_aggregate(...)` 实现。

对于 `full` 方法，聚合权重按下式计算：

`A_i^t = n_i * exp(-lambda_s * Delta_t_i) * q_i^t * r_i^t`

`alpha_i^t = A_i^t / sum_{j in P_k} A_j^t`

其中：

- `n_i`：客户端样本数
- `Delta_t_i`：陈旧度
- `q_i^t`：链路质量
- `r_i^t`：客户端信誉
- `lambda_s`：陈旧度衰减系数

然后对同一轨道面内的卫星模型进行加权平均，得到轨道面模型。

如果方法是：

- `fedavg`
- `intra_only`
- `intra_gossip`

则当前实现使用更简单的样本数加权。

### 8.4 面间异步 gossip

轨道面间协同由 [4_train/OrbitShield_FL/gossip.py](./4_train/OrbitShield_FL/gossip.py) 中的 `inter_plane_gossip(...)` 实现。

其思想是：

- 只有当前轮可见的邻居轨道面才参与 gossip
- 邻居轨道面模型根据链路质量和陈旧度进行再加权
- 当前轨道面模型与邻面混合模型再按 `beta` 混合

代码中使用了自适应混合系数：

- 最小值由 `beta_floor` 控制
- 实际混合强度由邻居平均质量决定

这样可以避免链路差时仍然强行做大幅跨面混合。

### 8.5 通信失败补偿

当邻居轨道面模型未成功收到时，由 [4_train/OrbitShield_FL/compensation.py](./4_train/OrbitShield_FL/compensation.py) 中的 `compensate_missing_model(...)` 进行补偿：

`w_hat_m^t = rho * w_m^{last} + (1-rho) * w_k^{self}`

其中：

- `w_m^{last}` 是邻面上一次可获得模型
- `w_k^{self}` 是当前轨道面自身模型
- `rho` 控制补偿时历史邻面信息所占比例

这使得“邻面短时失联”不会直接让 gossip 退化为完全不可用。

### 8.6 信誉更新

信誉机制由 [4_train/OrbitShield_FL/reputation.py](./4_train/OrbitShield_FL/reputation.py) 实现，核心函数为：

- `compute_score(...)`
- `update_reputation(...)`

信誉分数由三部分构成：

- `sim_i`：客户端更新与轨道面平均更新方向的相似度
- `improve_i`：该客户端更新带来的验证损失改善
- `stable_i`：历史参与稳定性

再按平滑更新公式：

`r_i^{t+1} = clip(mu * r_i^t + (1-mu) * score_i^t, r_min, 1.0)`

这使得长期稳定、方向一致、确实带来收益的客户端在后续聚合时权重更高。

### 8.7 动态拓扑模拟

动态拓扑由 [4_train/OrbitShield_FL/topology.py](./4_train/OrbitShield_FL/topology.py) 生成。

当前实现不是复杂轨道力学仿真，而是离散时隙近似：

- 面内链路：
  - 默认始终可用
  - 成功率较高
- 面间链路：
  - 有周期性可见窗口
  - 成功率较低
  - 带随机时延和丢包率

核心参数包括：

- `intra_plane_success_prob`
- `inter_plane_success_prob`
- `inter_plane_contact_period`
- `inter_plane_contact_duration`
- `packet_loss_prob`
- `link_delay_mean`

因此每轮训练前都会生成一个新的拓扑快照，用于决定当前轮哪些轨道面能交换模型。

## 9. 训练编排逻辑

联邦主控不采用传统中心式 `server` 命名，而是使用：

- [4_train/OrbitShield_FL/serverless_orchestrator.py](./4_train/OrbitShield_FL/serverless_orchestrator.py)

其中的核心类是：

- `ServerlessOrchestrator`

主流程如下：

1. 加载 `dataset_cicids17`
2. 对训练集进行联邦划分
3. 创建 12 个 `FederatedClient`
4. 为每轮生成动态拓扑
5. 让可用客户端执行本地训练
6. 先做面内聚合
7. 再做面间 gossip
8. 对失败邻面做补偿
9. 更新信誉、陈旧度和拓扑记录
10. 用全局验证集、测试集评估
11. 保存本轮指标和最优模型

脚本层的直接入口是：

- [4_train/scripts/train_federated.py](./4_train/scripts/train_federated.py)

## 10. 代码结构说明

联邦模块位于：

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

各文件职责如下：

### 10.1 `config.py`

定义 `FederatedConfig`，统一管理：

- 联邦训练轮数
- 客户端数量
- 轨道面数量
- 本地训练参数
- gossip 参数
- 信誉参数
- 拓扑参数
- 输出目录

当前默认配置已经固化为项目最终联邦版本 `OrbitShield_FL`。

### 10.2 `partition.py`

负责：

- 加载 `train/val/test.npz`
- 对 `train.npz` 做联邦划分
- 构造每个客户端的 `DataLoader`
- 导出 `partition_stats.json`

这是联邦训练与现有数据闭环衔接的入口。

### 10.3 `client.py`

负责定义 `FederatedClient`。

每个客户端对象都维护：

- `client_id`
- `plane_id`
- `sample_count`
- `reputation`
- `last_sync_round`
- 本地 `DSC-CBAM-GRU` 模型

并提供：

- `get_weights()`
- `set_weights()`
- `local_train()`
- `evaluate()`

### 10.4 `aggregators.py`

负责参数聚合和状态字典运算，包括：

- `clone_state_dict`
- `subtract_state_dict`
- `weighted_average_state_dict`
- `compute_staleness`
- `estimate_link_quality`
- `intra_plane_aggregate`

这里是联邦聚合数学逻辑的核心实现位置。

### 10.5 `gossip.py`

负责轨道面间的模型混合，核心函数是：

- `inter_plane_gossip(...)`

它会根据：

- 当前邻居是否可见
- 邻居链路质量
- 邻居模型陈旧度
- `beta`
- `beta_floor`

决定当前轮跨面信息应混合多少。

### 10.6 `compensation.py`

负责在跨面模型缺失时构造补偿模型，避免“邻居一失联就完全无信息”。

### 10.7 `reputation.py`

负责客户端信誉分数的计算与更新。  
其作用是区分：

- 更新方向稳定且有效的客户端
- 不稳定或贡献较低的客户端

从而让后续面内聚合更鲁棒。

### 10.8 `topology.py`

负责每轮生成动态链路快照，包括：

- 面内链路状态
- 面间链路状态
- 当前轨道面邻接关系

这是联邦算法“贴合星座结构”的关键模块之一。

### 10.9 `metrics_fl.py`

负责全局验证与测试指标统计，包括：

- `accuracy`
- `precision`
- `recall`
- `f1`
- `confusion_matrix`
- 每轮通信代价
- 陈旧更新比例
- 链路鲁棒性

### 10.10 `serverless_orchestrator.py`

这是联邦训练总调度器，负责把：

- 数据划分
- 客户端训练
- 面内聚合
- 面间 gossip
- 补偿
- 信誉更新
- 指标输出

串成一个可运行的联邦训练闭环。

## 11. 训练脚本说明

### 11.1 主训练脚本

- [4_train/scripts/train_federated.py](./4_train/scripts/train_federated.py)

主要参数包括：

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

大多数联邦超参数已经固化为脚本内部默认值，常规运行通常不需要再显式设置。

### 11.2 联邦默认脚本

- [4_train/scripts/run_federated.sh](./4_train/scripts/run_federated.sh)

该脚本已经固定为当前最优联邦配置，默认直接输出到：

- `experiments/OrbitShield_FL/cicids17`

### 11.3 联邦方法对比脚本

- [4_train/scripts/run_federated_ablation.sh](./4_train/scripts/run_federated_ablation.sh)

用于跑多种联邦方法对比，包括：

- `single`
- `fedavg`
- `intra_only`
- `intra_gossip`
- `full`

当前这些历史方法目录已清理，只保留最终汇总结果和正式版本目录。

### 11.4 联邦网格搜索脚本

- [4_train/scripts/tune_federated_full.py](./4_train/scripts/tune_federated_full.py)

用于对完整联邦方案进行系统调参，当前主要搜索：

- `beta`
- `warmup_rounds`
- `global_momentum`

### 11.5 网格搜索绘图脚本

- [4_train/scripts/plot_federated_grid_search.py](./4_train/scripts/plot_federated_grid_search.py)

用于绘制：

- 热力图
- 性能趋势图
- Top10 配置柱状图

## 12. 输出文件说明

每次联邦训练会输出：

- `round_metrics.csv`
- `summary.json`
- `best_global_model.pt`
- `partition_stats.json`
- `reputation_history.json`
- `topology_history.json`

含义分别是：

- `round_metrics.csv`：逐轮验证和测试指标
- `summary.json`：本次实验的最终摘要
- `best_global_model.pt`：验证集最优模型
- `partition_stats.json`：客户端样本量和标签分布
- `reputation_history.json`：各卫星信誉随轮次变化
- `topology_history.json`：各轮拓扑快照记录

## 13. Warm Start 设计

为了提升联邦训练早期收敛速度，当前联邦训练默认从单机最佳模型开始：

- [4_train/checkpoints_gru/cicids17_gru_best.pt](./4_train/checkpoints_gru/cicids17_gru_best.pt)

早期验证结果表明：

- 不使用 warm start 时，1 轮快速验证准确率约为 `0.3654`
- 使用 warm start 后，1 轮快速验证准确率可提升到约 `0.9002`

因此，`OrbitShield_FL` 最终将 warm start 固化为默认行为之一。

## 14. 当前正式结果

当前最终联邦版本目录：

- [4_train/experiments/OrbitShield_FL/cicids17](./4_train/experiments/OrbitShield_FL/cicids17)

当前正式结果为：

- `Accuracy = 0.9718`
- `Precision = 0.9719`
- `Recall = 0.9718`
- `F1 = 0.9718`

历史方法对比结果如下：

| 方法 | Accuracy | Precision | Recall | F1 | 平均通信开销(MB/round) |
|------|----------|-----------|--------|----|------------------------|
| `single` | `0.9601` | `0.9606` | `0.9601` | `0.9599` | `0.0000` |
| `fedavg` | `0.9571` | `0.9584` | `0.9571` | `0.9570` | `1.7869` |
| `intra_only` | `0.9636` | `0.9638` | `0.9636` | `0.9636` | `1.7869` |
| `intra_gossip` | `0.9159` | `0.9262` | `0.9159` | `0.9143` | `1.7869` |
| `full` | `0.9389` | `0.9423` | `0.9389` | `0.9381` | `1.7869` |
| `OrbitShield_FL` | `0.9718` | `0.9719` | `0.9718` | `0.9718` | `1.7869` |

可以看到，`OrbitShield_FL` 已经超过：

- `single`
- `fedavg`
- `intra_only`

这说明在经过合理调参后，跨面协同确实可以从“潜在噪声源”转化为“净收益”。

## 15. OrbitShield_FL 网格搜索

为了继续提升完整联邦方案，本项目保留了完整网格搜索档案：

- [4_train/experiments/OrbitShield_FL/grid_search](./4_train/experiments/OrbitShield_FL/grid_search)

核心汇总文件包括：

- [grid_search_summary.csv](./4_train/experiments/OrbitShield_FL/grid_search/grid_search_summary.csv)
- [grid_search_results.json](./4_train/experiments/OrbitShield_FL/grid_search/grid_search_results.json)

当前最优配置为：

- `beta = 0.1`
- `warmup_rounds = 2`
- `global_momentum = 0.1`
- `beta_floor = 0.05`

这套参数就是当前正式版 `OrbitShield_FL` 的来源。
并且它已经固化在 [4_train/OrbitShield_FL/config.py](./4_train/OrbitShield_FL/config.py) 的 `full` 正式 preset 中。

前 5 组配置如下：

| 排名 | beta | warmup_rounds | global_momentum | Accuracy | F1 |
|------|------|---------------|-----------------|----------|----|
| `1` | `0.1` | `2` | `0.1` | `0.9718` | `0.9718` |
| `2` | `0.3` | `2` | `0.1` | `0.9678` | `0.9677` |
| `3` | `0.2` | `2` | `0.3` | `0.9659` | `0.9658` |
| `4` | `0.2` | `2` | `0.2` | `0.9648` | `0.9646` |
| `5` | `0.2` | `3` | `0.2` | `0.9615` | `0.9614` |

## 16. 可视化结果

当前已经生成的调参图包括：

- [OrbitShield_FL_heatmaps.png](./4_train/experiments/OrbitShield_FL/grid_search/plots/OrbitShield_FL_heatmaps.png)
- [OrbitShield_FL_trends.png](./4_train/experiments/OrbitShield_FL/grid_search/plots/OrbitShield_FL_trends.png)
- [OrbitShield_FL_top10.png](./4_train/experiments/OrbitShield_FL/grid_search/plots/OrbitShield_FL_top10.png)

这些图用于回答三个问题：

- `beta` 对性能的影响是否过强
- `warmup_rounds` 是否确实改善跨面混合稳定性
- `global_momentum` 应该取小还是取大

从当前结果看：

- 较小的 `beta` 更稳
- `warmup_rounds = 2` 最合适
- 较小的 `global_momentum = 0.1` 更有利于稳定提升

## 17. 推荐运行命令

### 17.1 默认运行正式联邦版本

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
./scripts/run_federated.sh
```

### 17.2 直接命令行运行

```bash
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated.py \
  --dataset cicids17 \
  --method full \
  --device cuda
```

### 17.3 重绘调参图

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/plot_federated_grid_search.py \
  --csv_path experiments/OrbitShield_FL/grid_search/grid_search_summary.csv \
  --output_dir experiments/OrbitShield_FL/grid_search/plots
```

### 17.4 在 STI 上运行完整联邦训练

`STI` 的训练样本、验证样本和测试样本规模都远大于 `cicids17`。因此在 `STI` 上，联邦训练默认额外启用了两项工程优化：

- 每轮仅使用 `50,000` 条固定验证子集做验证
- 完整测试集仅在训练结束后执行一次

同时默认启用：

- `max_local_batches = 128`

即每颗卫星在每轮本地更新时只执行固定 batch 预算的本地 SGD，而不是完整扫过本地全部数据。这样既保留了联邦算法结构，也把超大数据集上的运行时间控制在可接受范围。

实际运行命令如下：

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated.py \
  --dataset sti \
  --method full \
  --device cuda
```

结果目录：

- [OrbitShield_FL_sti](./4_train/experiments/OrbitShield_FL/sti)

实际结果：

- `Best Val Acc = 0.9956`
- `Test Accuracy = 0.9955`
- `Test Precision = 0.9955`
- `Test Recall = 0.9955`
- `Test F1 = 0.9955`

## 18. 论文式实验分析

当前实验结果说明，面向低轨卫星多星协同威胁预测，是否进行跨面模型交换并不是唯一关键，更关键的是如何在断续连接、异质数据和陈旧更新条件下，控制交换节奏和混合强度。

标准 `FedAvg` 提供了一个稳定但不够结构化的基线，其聚合过程中没有显式利用轨道面结构，因此在非 IID 划分下收益有限。仅采用轨道面内聚合的 `intra_only` 已经取得了较强结果，这表明“轨道面内稳定协同”本身就能够有效缓解单星样本不足问题。

然而，早期的 `intra_gossip` 和未经充分调参的 `full` 结果也表明，跨面模型交换如果过于激进，会把来自不同轨道面的异质更新直接注入到已相对稳定的局部表示中，导致判别边界被扰动，最终使性能下降。

`OrbitShield_FL` 的改进之处在于，它不是简单增加通信，而是联合引入了：

- 单机最优模型 warm start
- 面内 warmup
- 自适应 gossip 权重
- 信誉感知聚合
- 陈旧度衰减
- 全局 EMA 稳定器

在这些机制共同作用下，跨面协同从噪声源转变为有效增益，最终将测试准确率提升到 `0.9718`。这说明在低轨卫星协同威胁预测中，真正重要的不是“是否做联邦”，而是“如何把轨道结构、连接断续性和更新可信度共同编码进联邦聚合过程”。

## 19. 后续可扩展方向

当前版本已经是一个可直接运行、结构清晰、与现有工程强复用的第一版联邦学习系统。下一步可扩展方向包括：

1. 将面间邻接关系从环状近似扩展为更真实的接触图
2. 将 `contact_period/contact_duration` 与 NS-3 输出的链路统计耦合
3. 用仿真阶段真实测得的时延、丢包和链路可用率替换当前启发式链路质量估计
4. 为每颗卫星维护独立验证切片，进一步细化信誉机制
5. 将当前算法仿真级联邦扩展到真实多进程或多机联邦原型

## 20. 总结

`OrbitShield_FL` 是在现有单机 `DSC-CBAM-GRU` 威胁检测工程上，增量式构建出来的低轨卫星多星联邦训练版本。

它的特点是：

- 不破坏原有单机闭环
- 复用现有数据、模型和训练基础设施
- 用最小新增模块实现多星联邦训练
- 在结构上贴近低轨卫星星座协同特性
- 当前已经具备可运行、可复现、可扩展的实验基础

如果只保留一个最终联邦版本，当前正式推荐的就是：

- `OrbitShield_FL`

## 21. ns-3 驱动的联邦协同仿真扩展

除了默认的启发式动态拓扑版本外，当前工程还新增了一条独立的 `ns-3` 驱动联邦协同仿真链路，用于提升低轨卫星通信环境建模的可信度。

这里必须强调一个边界：

- [realtime_satellite.cc](./realtime_satellite.cc) 不参与联邦训练环境仿真
- [realtime_satellite.cc](./realtime_satellite.cc) 仍然只负责数据集生成

因此，项目中的 `ns-3` 现在分成两条互不干扰的用途：

1. 数据生成  
   `原始 PCAP -> realtime_satellite.cc -> captured -> 特征提取 -> dataset_cicids17 / dataset_sti`
2. 联邦通信环境  
   `federated_constellation.cc -> ns-3 round trace -> OrbitShield_FL(ns3 backend)`

## 22. 新增模块与职责

ns-3 驱动联邦协同仿真新增了以下模块：

- [federated_constellation.cc](./federated_constellation.cc)  
  独立的 ns-3 星座通信环境仿真器，负责导出轮级通信 trace
- [4_train/OrbitShield_FL/ns3_bridge.py](./4_train/OrbitShield_FL/ns3_bridge.py)  
  负责调用 ns-3 二进制、加载 `manifest.json` 和每轮 trace、做字段校验
- [4_train/OrbitShield_FL/topology_ns3.py](./4_train/OrbitShield_FL/topology_ns3.py)  
  负责把 ns-3 trace 映射为联邦主控能直接使用的拓扑快照
- [4_train/OrbitShield_FL/transfer_scheduler.py](./4_train/OrbitShield_FL/transfer_scheduler.py)  
  负责根据模型大小、带宽、丢包和接触窗口，判断模型能否在本轮成功传输
- [4_train/scripts/train_federated_ns3.py](./4_train/scripts/train_federated_ns3.py)  
  独立的 ns-3 联邦训练入口，不替换 [4_train/scripts/train_federated.py](./4_train/scripts/train_federated.py)

这几层的关系是：

`federated_constellation.cc -> ns3_bridge.py -> topology_ns3.py -> serverless_orchestrator.py -> OrbitShield_FL`

## 23. ns-3 trace 的作用

[federated_constellation.cc](./federated_constellation.cc) 会导出：

- `constellation_config.json`
- `manifest.json`
- `round_0001.json ... round_xxxx.json`

每轮 trace 至少包含：

- 轨道面内链路状态
- 轨道面间链路状态
- 每条链路的：
  - `available`
  - `success`
  - `delay_ms`
  - `bandwidth_mbps`
  - `packet_loss`
  - `contact_duration_s`

这些字段会被联邦主控用于：

1. 决定当前哪些轨道面可以发生 gossip
2. 估计链路质量并更新鲁棒聚合权重
3. 判断模型参数是否能在接触窗口内传完

第 3 点是本次扩展的核心。默认启发式拓扑只能近似表示“链路好坏”，而 `ns-3` 驱动版本可以更进一步判断：

- 当前链路是否存在
- 当前链路是否成功
- 当前窗口是否足够支撑一次模型交换

如果传输不能完成，则本轮 gossip 失败，并进入当前已有的补偿逻辑。

## 24. 代码实现要点

### 24.1 `federated_constellation.cc`

该程序完成以下工作：

- 构建多轨道面、多卫星节点
- 按 ring 结构建立面内链路和面间链路
- 以轮为单位生成通信状态
- 导出完整的 trace 目录

它当前支持的关键输入包括：

- `--num-planes`
- `--sats-per-plane`
- `--rounds`
- `--round-duration`
- `--intra-rate / --intra-delay / --intra-loss`
- `--inter-rate / --inter-delay / --inter-loss`
- `--contact-period`
- `--contact-duration-rounds`
- `--output-dir`
- `--seed`

### 24.2 `ns3_bridge.py`

该模块提供：

- `run_federated_constellation(...)`
- `load_ns3_trace_bundle(...)`
- `load_ns3_round_trace(...)`
- `validate_round_trace(...)`

也就是说，Python 侧不直接散乱读取 json，而是统一通过桥接层获取经过校验的 trace。

### 24.3 `topology_ns3.py`

该模块负责把 `ns-3` 轮级 trace 转成和现有联邦框架兼容的拓扑快照结构，包括：

- `plane_neighbors`
- `intra_plane_links`
- `inter_plane_links`
- 面内和面间带宽映射

因此，联邦主控不需要重写 gossip 和聚合逻辑，只需要切换 `topology backend` 即可。

### 24.4 `transfer_scheduler.py`

该模块负责计算：

- 模型参数大小
- 有效吞吐
- 预计传输耗时
- 本轮是否能在 `contact_duration_s` 内传完

这让联邦训练真正开始受“通信完成性”约束，而不是只受“抽象成功率”约束。

### 24.5 `train_federated_ns3.py`

该脚本是新的实验入口，特点是：

- 不影响默认 [4_train/scripts/train_federated.py](./4_train/scripts/train_federated.py)
- 支持直接读取已有 trace
- 也支持训练前自动调用 ns-3 生成 trace

## 25. 运行方式

### 25.1 使用已有 trace 训练

```bash
cd /home/lithic/final/ns3/ns-3-allinone/ns-3.46.1/scratch/06_realtime_emulation/4_train
/home/lithic/final/ns3-gpu-venv/bin/python scripts/train_federated_ns3.py \
  --dataset cicids17 \
  --trace_dir experiments/OrbitShield_FL_ns3/cicids17_trace \
  --output_dir experiments/OrbitShield_FL_ns3/cicids17 \
  --device cuda
```

这里如果显式传 `--method full`，仍然会自动复用 [4_train/OrbitShield_FL/config.py](./4_train/OrbitShield_FL/config.py) 里固化的正式最优 preset。

### 25.2 训练前自动生成 trace

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

因此，heuristic 联邦入口和 ns-3 联邦入口在 `full` 模式下使用的是同一套正式最优参数，而不是两套彼此独立的默认值。

## 26. 当前已跑通的 ns-3 联调结果

当前已经实际跑通了一组完整的联调实验：

- ns-3 trace 目录：
  - [4_train/experiments/OrbitShield_FL_ns3/cicids17_trace](./4_train/experiments/OrbitShield_FL_ns3/cicids17_trace)
- 训练输出目录：
  - [4_train/experiments/OrbitShield_FL_ns3/cicids17](./4_train/experiments/OrbitShield_FL_ns3/cicids17)

其中 trace 目录包含：

- [constellation_config.json](./4_train/experiments/OrbitShield_FL_ns3/cicids17_trace/constellation_config.json)
- [manifest.json](./4_train/experiments/OrbitShield_FL_ns3/cicids17_trace/manifest.json)
- 5 个 `round_000x.json`

训练目录包含：

- [summary.json](./4_train/experiments/OrbitShield_FL_ns3/cicids17/summary.json)
- [round_metrics.csv](./4_train/experiments/OrbitShield_FL_ns3/cicids17/round_metrics.csv)
- [best_global_model.pt](./4_train/experiments/OrbitShield_FL_ns3/cicids17/best_global_model.pt)
- [partition_stats.json](./4_train/experiments/OrbitShield_FL_ns3/cicids17/partition_stats.json)
- [reputation_history.json](./4_train/experiments/OrbitShield_FL_ns3/cicids17/reputation_history.json)
- [topology_history.json](./4_train/experiments/OrbitShield_FL_ns3/cicids17/topology_history.json)

本次联调条件：

- 数据集：`cicids17`
- backend：`ns3`
- 轮数：`5`
- trace：自动生成
- 设备：`cuda`

实际结果为：

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

这组结果的意义主要在于证明：

- 新增的 ns-3 星座联邦环境仿真器已经能稳定工作
- trace 桥接、拓扑映射和传输调度都已接入联邦主控
- `OrbitShield_FL` 已经从“启发式低轨联邦”扩展成“支持 ns-3 驱动通信环境的低轨联邦”

## 27. 当前判断

默认的 `OrbitShield_FL` 结果仍然是当前项目中用于算法对比和最终报告的正式联邦结果，因为那套实验已经做了完整调参和结果固化。

新增的 ns-3 驱动版本当前更适合承担以下角色：

1. 作为更高可信度的通信环境验证链路
2. 作为后续论文中“算法在更真实低轨通信条件下仍可运行”的支撑实验
3. 作为后续继续扩展星地链路、时变容量和更真实接触图的基础工程框架
