import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(csv_path, seq_length=10, batch_size=64):
    df = pd.read_csv(csv_path)
    
    # 按照规划选择 17 个专业特征
    feature_cols = [
        'pkt_len_mean', 'pkt_len_std', 'pkt_len_max', 'pkt_len_min', 
        'flag_syn', 'flag_ack', 'flag_rst', 'win_size_mean', 
        'iat_mean', 'iat_std', 'iat_max', 'duration', 
        'pps', 'bps', 'ttl_mean', 'retrans_rate', 'dst_port'
    ]
    
    # 1. 归一化 (使用 StandardScaler 对统计特征更友好)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    X, y = [], []
    
    # 2. 基于 file_id 的滑动窗口切分
    # 确保窗口内的包都属于同一次仿真，防止数据泄露
    for fid in df['file_id'].unique():
        group = df[df['file_id'] == fid]
        data = group[feature_cols].values
        labels = group['label'].values
        
        if len(data) < seq_length: continue
        
        for i in range(len(data) - seq_length + 1):
            X.append(data[i : i + seq_length])
            y.append(labels[i + seq_length - 1])
            
    X = np.array(X)
    y = np.array(y)
    
    # 3. 划分并打包
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=batch_size
    )
    
    return train_loader, test_loader, len(feature_cols)

class TensorDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]