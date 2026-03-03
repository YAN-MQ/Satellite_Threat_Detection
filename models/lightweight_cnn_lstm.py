"""
轻量化CNN-LSTM模型 for 卫星网络威胁检测
- 输入: (batch_size, 3) - timestamp, packet_size, iat
- 输出: 4类分类概率
- 参数量 < 100K
- 推理时间 < 10ms
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import time


class LightweightCNN_LSTM(nn.Module):
    """
    轻量化CNN-LSTM模型
    1D卷积提取空间特征 + LSTM处理时序依赖
    """
    def __init__(self, input_dim=3, hidden_dim=32, num_classes=4):
        super(LightweightCNN_LSTM, self).__init__()
        
        # 1D卷积层 - 提取空间特征
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)
        
        # LSTM层 - 处理时序依赖
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True,
            dropout=0
        )
        
        # 全连接分类器
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim) -> 需要转换为 (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # 1D卷积特征提取
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # 转换为LSTM输入格式: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM处理时序
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        
        # 分类
        out = self.fc(out)
        return out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SatelliteDataset(Dataset):
    """卫星网络威胁数据集"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_preprocess_data(data_dir, seq_length=10):
    """
    加载并预处理数据
    - 使用StandardScaler归一化
    - 构建时序样本
    """
    feature_cols = ['timestamp', 'packet_size', 'iat']
    
    # 加载数据
    train_df = pd.read_csv(f'{data_dir}/train_dataset.csv')
    val_df = pd.read_csv(f'{data_dir}/val_dataset.csv')
    test_df = pd.read_csv(f'{data_dir}/test_dataset.csv')
    
    # 合并训练和验证数据用于归一化
    all_data = pd.concat([train_df, val_df], ignore_index=True)
    
    # 归一化
    scaler = StandardScaler()
    scaler.fit(all_data[feature_cols])
    
    # 归一化训练、验证、测试数据
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # 构建时序样本
    def create_sequences(df, seq_length):
        X, y = [], []
        labels = df['label'].values
        features = df[feature_cols].values
        
        for i in range(len(df) - seq_length + 1):
            X.append(features[i:i+seq_length])
            y.append(labels[i+seq_length-1])
        
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_df, seq_length)
    X_val, y_val = create_sequences(val_df, seq_length)
    X_test, y_test = create_sequences(test_df, seq_length)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, device):
    """训练模型 with early stopping"""
    best_val_acc = 0.0
    best_model_state = None
    no_improve_count = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc


def evaluate_model(model, test_loader, device):
    """评估模型 - 返回混淆矩阵和准确率"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算准确率
    accuracy = 100 * np.sum(all_preds == all_labels) / len(all_labels)
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    
    # 分类报告
    target_names = ['Normal', 'DDoS', 'PortScan', 'Route']
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    
    return accuracy, cm, report


def measure_inference_time(model, test_loader, device, num_iterations=1000):
    """测量推理时间"""
    model.eval()
    
    # 预热
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= 10:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # 测量推理时间
    start_time = time.time()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_iterations:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    
    return avg_time


def main():
    # 配置
    DATA_DIR = 'data'
    SEQ_LENGTH = 10
    BATCH_SIZE = 128
    HIDDEN_DIM = 32
    EPOCHS = 50
    PATIENCE = 10
    LEARNING_RATE = 0.001
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_preprocess_data(
        DATA_DIR, SEQ_LENGTH
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 创建数据加载器
    train_dataset = SatelliteDataset(X_train, y_train)
    val_dataset = SatelliteDataset(X_val, y_val)
    test_dataset = SatelliteDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 创建模型
    model = LightweightCNN_LSTM(input_dim=3, hidden_dim=HIDDEN_DIM, num_classes=4)
    model = model.to(device)
    
    # 打印模型参数量
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")
    if num_params < 100000:
        print("✓ Parameter count < 100K requirement met!")
    else:
        print("✗ Parameter count exceeds 100K!")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练模型
    print("\nStarting training...")
    model, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, EPOCHS, PATIENCE, device
    )
    
    # 评估模型
    print("\nEvaluating on test set...")
    accuracy, cm, report = evaluate_model(model, test_loader, device)
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # 测量推理时间
    print("\nMeasuring inference time...")
    avg_inference_time = measure_inference_time(model, test_loader, device)
    print(f"Average inference time: {avg_inference_time:.4f} ms")
    
    if avg_inference_time < 10:
        print("✓ Inference time < 10ms requirement met!")
    else:
        print("✗ Inference time exceeds 10ms!")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'accuracy': accuracy,
        'num_params': num_params,
    }, 'checkpoints/lightweight_cnn_lstm.pth')
    print("\nModel saved to checkpoints/lightweight_cnn_lstm.pth")


if __name__ == "__main__":
    main()
