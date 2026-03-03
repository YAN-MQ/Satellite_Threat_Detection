"""
联邦学习训练脚本 - FedAvg算法

功能:
1. 服务端初始化全局模型
2. 分发模型到5个客户端(卫星节点)
3. 客户端本地训练(5 epochs)
4. FedAvg聚合更新全局模型
5. 20轮迭代,记录验证集表现
6. 日志记录和结果保存
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime

from config import config, get_config
from models.dsc_cbam_lstm import DSC_CBAM_LSTM
from core.logger import TrainingLogger, ResultsWriter


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_and_split_data(data_dir="dataset", num_clients=5, seq_length=10, batch_size=64):
    """加载数据并按客户端划分(模拟卫星节点数据分布)"""
    print("\n" + "=" * 60)
    print("Loading and partitioning data for federated learning...")
    print("=" * 60)
    
    train_csv = os.path.join(data_dir, "train_dataset_final.csv")
    val_csv = os.path.join(data_dir, "val_dataset_final.csv")
    test_csv = os.path.join(data_dir, "test_dataset_final.csv")
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    feature_cols = [
        'IAT_Mean_norm', 'IAT_Std_norm', 'Duration_norm', 
        'PktLen_Mean_norm', 'PktLen_Std_norm', 
        'Packets_s_norm', 'Bytes_s_norm', 'Idle_Mean_norm'
    ]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    def create_sequences(X, y, seq_len, stride=1):
        X_seq, y_seq = [], []
        for i in range(0, len(X) - seq_len + 1, stride):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len - 1])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
    
    indices = np.arange(len(X_train_seq))
    np.random.shuffle(indices)
    client_indices = np.array_split(indices, num_clients)
    
    client_data = []
    for i, idx in enumerate(client_indices):
        train_ds = TensorDataset(
            torch.FloatTensor(X_train_seq[idx]),
            torch.LongTensor(y_train_seq[idx])
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        num_samples = len(idx)
        print(f"  Client {i+1}: {num_samples} samples ({100*num_samples/len(X_train_seq):.1f}%)")
        client_data.append({
            'train_loader': train_loader,
            'num_samples': num_samples
        })
    
    val_ds = TensorDataset(torch.FloatTensor(X_val_seq), torch.LongTensor(y_val_seq))
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    test_ds = TensorDataset(torch.FloatTensor(X_test_seq), torch.LongTensor(y_test_seq))
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    print(f"\nTotal training samples: {len(X_train_seq)}")
    print(f"Validation samples: {len(X_val_seq)}")
    print(f"Test samples: {len(X_test_seq)}")
    print(f"Input dimension: {len(feature_cols)}")
    
    return client_data, val_loader, test_loader, len(feature_cols)


def get_model_parameters(model):
    """获取模型参数字典"""
    return {name: param.clone() for name, param in model.state_dict().items()}


def set_model_parameters(model, parameters):
    """设置模型参数"""
    model.load_state_dict(parameters)


def train_client(model, train_loader, criterion, optimizer, epochs, device, gradient_clip=1.0):
    """客户端本地训练"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / (len(train_loader) * epochs)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate_model(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def fedavg_aggregate(global_params, client_params_list, client_weights):
    """FedAvg聚合: 加权平均更新全局模型"""
    aggregated_params = {}
    
    for param_name in global_params.keys():
        weighted_sum = None
        
        for client_params, weight in zip(client_params_list, client_weights):
            if weighted_sum is None:
                weighted_sum = weight * client_params[param_name]
            else:
                weighted_sum += weight * client_params[param_name]
        
        aggregated_params[param_name] = weighted_sum
    
    return aggregated_params


def fedprox_regularize(client_model, global_params, mu=0.01):
    """FedProx正则化项：减少参数漂移"""
    reg_loss = 0.0
    for name, param in client_model.named_parameters():
        reg_loss += torch.sum((param - global_params[name]) ** 2)
    return mu * reg_loss


class FocalLoss(nn.Module):
    """Focal Loss: 解决类别不平衡问题"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def train_client_enhanced(model, train_loader, criterion, optimizer, epochs, device, 
                          global_params=None, mu=0.01, gradient_clip=1.0):
    """增强版客户端本地训练（支持FedProx）"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if global_params is not None and mu > 0:
                prox_loss = 0.0
                for name, param in model.named_parameters():
                    prox_loss += torch.sum((param - global_params[name]) ** 2)
                loss = loss + mu * prox_loss
            
            loss.backward()
            
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / (len(train_loader) * epochs)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def knowledge_distillation_aggregate(global_model, client_models, client_weights, 
                                      temperature=4.0, alpha=0.5):
    """知识蒸馏聚合：让全局模型吸收客户端知识"""
    device = next(global_model.parameters()).device
    
    avg_logits = None
    total_weight = sum(client_weights)
    
    for client_model, weight in zip(client_models, client_weights):
        client_model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 10, 8).to(device)
            logits = client_model(dummy_input)
            
            if avg_logits is None:
                avg_logits = (weight / total_weight) * logits
            else:
                avg_logits += (weight / total_weight) * logits
    
    global_params = {name: param.clone() for name, param in global_model.state_dict().items()}
    
    return global_params


def federated_train(config, num_rounds=100, local_epochs=5, use_fedprox=True, mu=0.001, logger=None):
    """联邦学习主训练流程"""
    set_seed(config.seed)
    
    device = config.device_obj
    print(f"\nUsing device: {device}")
    
    print("\n" + "=" * 60)
    print("Initializing Federated Learning...")
    print("=" * 60)
    print(f"Number of clients: 5")
    print(f"Local epochs per round: {local_epochs}")
    print(f"Communication rounds: {num_rounds}")
    print(f"FedProx: {use_fedprox}, mu={mu}")
    
    client_data, val_loader, test_loader, input_dim = load_and_split_data(
        data_dir=config.data.data_dir,
        num_clients=5,
        seq_length=config.model.seq_len,
        batch_size=config.train.batch_size
    )
    
    print("\n" + "=" * 60)
    print("Creating global model...")
    print("=" * 60)
    
    global_model = DSC_CBAM_LSTM(
        seq_len=config.model.seq_len,
        input_dim=input_dim,
        hidden_dim=config.model.hidden_dim,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout
    ).to(device)
    
    print(f"Model: DSC-CBAM-LSTM")
    print(f"Parameters: {global_model.count_parameters():,}")
    
    total_samples = sum(c['num_samples'] for c in client_data)
    client_weights = [c['num_samples'] / total_samples for c in client_data]
    
    print("\n" + "=" * 60)
    print("Starting Federated Training...")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'round': [],
        'val_loss': [],
        'val_acc': [],
        'avg_client_loss': [],
        'avg_client_acc': []
    }
    
    best_val_acc = 0.0
    best_round = 0
    
    initial_lr = config.train.learning_rate
    
    for round_idx in range(num_rounds):
        round_start = time.time()
        
        current_lr = initial_lr * (0.5 ** (round_idx // 10))
        
        print(f"\n--- Round {round_idx + 1}/{num_rounds} (LR: {current_lr:.6f}) ---")
        
        global_params = get_model_parameters(global_model)
        
        client_params_list = []
        client_losses = []
        client_accs = []
        
        for client_idx, c_data in enumerate(client_data):
            client_model = DSC_CBAM_LSTM(
                seq_len=config.model.seq_len,
                input_dim=input_dim,
                hidden_dim=config.model.hidden_dim,
                num_classes=config.model.num_classes,
                dropout=config.model.dropout
            ).to(device)
            
            set_model_parameters(client_model, global_params)
            
            optimizer = optim.Adam(
                client_model.parameters(),
                lr=current_lr,
                weight_decay=config.train.weight_decay
            )
            
            if use_fedprox and round_idx > 0:
                loss, acc = train_client_enhanced(
                    client_model,
                    c_data['train_loader'],
                    criterion,
                    optimizer,
                    local_epochs,
                    device,
                    global_params=global_params,
                    mu=mu,
                    gradient_clip=config.train.gradient_clip
                )
            else:
                loss, acc = train_client(
                    client_model,
                    c_data['train_loader'],
                    criterion,
                    optimizer,
                    local_epochs,
                    device,
                    config.train.gradient_clip
                )
            
            client_params = get_model_parameters(client_model)
            client_params_list.append(client_params)
            client_losses.append(loss)
            client_accs.append(acc)
            
            print(f"  Client {client_idx + 1}: Loss={loss:.4f}, Acc={acc:.2f}%")
            
            del client_model
        
        aggregated_params = fedavg_aggregate(global_params, client_params_list, client_weights)
        set_model_parameters(global_model, aggregated_params)
        
        val_loss, val_acc = validate_model(global_model, val_loader, criterion, device)
        
        avg_client_loss = np.mean(client_losses)
        avg_client_acc = np.mean(client_accs)
        
        history['round'].append(round_idx + 1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['avg_client_loss'].append(avg_client_loss)
        history['avg_client_acc'].append(avg_client_acc)
        
        round_time = time.time() - round_start
        print(f"  Aggregation complete!")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Round time: {round_time:.2f}s")
        
        # 记录到日志
        if logger:
            client_metrics = [
                {'client_id': i+1, 'loss': loss, 'accuracy': acc}
                for i, (loss, acc) in enumerate(zip(client_losses, client_accs))
            ]
            logger.log_federated_round(
                round_idx + 1,
                {'loss': val_loss, 'accuracy': val_acc},
                client_metrics
            )
        
        if (round_idx + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"fedavg_round_{round_idx + 1}.pth"
            )
            torch.save({
                'round': round_idx + 1,
                'model_state_dict': global_model.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Federated Training Complete!")
    print("=" * 60)
    
    final_loss, final_acc = validate_model(global_model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"  Test Loss: {final_loss:.4f}")
    print(f"  Test Accuracy: {final_acc:.2f}%")
    
    print("\nTraining History:")
    print(f"{'Round':<8}{'Val Loss':<12}{'Val Acc':<12}{'Client Loss':<12}{'Client Acc':<12}")
    print("-" * 56)
    for i in range(len(history['round'])):
        print(f"{history['round'][i]:<8}{history['val_loss'][i]:<12.4f}{history['val_acc'][i]:<12.2f}%{history['avg_client_loss'][i]:<12.4f}{history['avg_client_acc'][i]:<12.2f}%")
    
    final_model_path = os.path.join(config.checkpoint_dir, "fedavg_final.pth")
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'val_acc': history['val_acc'][-1],
        'test_acc': final_acc,
        'history': history
    }, final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    return history


def main():
    # 创建日志记录器
    experiment_name = f"fedavg_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(log_dir='logs', experiment_name=experiment_name)
    results_writer = ResultsWriter(results_dir='results')
    
    # 记录配置
    fedprox_config = {
        'strategy': 'FedProx',
        'num_clients': 5,
        'local_epochs': 5,
        'num_rounds': 100,
        'mu': 0.001
    }
    logger.info(f"Starting federated learning experiment: {experiment_name}")
    logger.info(f"Configuration: {json.dumps(fedprox_config)}")
    
    # 训练模型
    history = federated_train(
        config, 
        num_rounds=100,
        local_epochs=5,
        use_fedprox=True,
        mu=0.001,
        logger=logger
    )
    
    # 记录最终结果
    results_writer.add_result('experiment_name', experiment_name)
    results_writer.add_result('strategy', 'FedProx')
    results_writer.add_result('num_clients', 5)
    results_writer.add_result('num_rounds', 100)
    results_writer.add_result('final_val_acc', history['val_acc'][-1])
    results_writer.add_result('training_history', history)
    
    # 保存结果
    results_path = results_writer.save(f'{experiment_name}_results.json')
    logger.info(f"Results saved to {results_path}")
    
    logger.close()
    print(f"\nTraining logs saved to logs/{experiment_name}.log")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
