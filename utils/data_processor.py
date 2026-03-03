"""
数据加载模块 - 支持卫星网络威胁检测数据集

功能:
1. 数据加载与归一化
2. 时序样本构建
3. 数据增强
4. PyTorch DataLoader封装
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List


class SatelliteThreatDataset(Dataset):
    """卫星网络威胁检测数据集"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 augment: bool = False):
        """
        Args:
            features: 特征数组 (N, seq_len, feature_dim)
            labels: 标签数组 (N,)
            augment: 是否启用数据增强
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.features[idx].clone()
        y = self.labels[idx]
        
        if self.augment:
            x = self._augment(x)
            
        return x, y
    
    def _augment(self, x):
        """数据增强: 添加轻微噪声"""
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        return x


class DataProcessor:
    """数据预处理器"""
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Args:
            scaler_type: 'standard', 'minmax', 或 'none'
        """
        self.scaler_type = scaler_type
        self.scaler = None
        
    def fit_transform(self, data: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """拟合并转换数据"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(data[feature_cols])
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            return self.scaler.fit_transform(data[feature_cols])
        else:
            self.scaler = None
            return data[feature_cols].values
    
    def transform(self, data) -> np.ndarray:
        """转换数据"""
        if self.scaler is None:
            return np.array(data)
        return self.scaler.transform(data)


def load_raw_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载原始CSV数据
    
    Returns:
        train_df, val_df, test_df
    """
    import glob
    
    train_files = glob.glob(os.path.join(data_dir, '*train*.csv'))
    val_files = glob.glob(os.path.join(data_dir, '*val*.csv'))
    test_files = glob.glob(os.path.join(data_dir, '*test*.csv'))
    
    if not train_files or not val_files or not test_files:
        raise FileNotFoundError(f"Cannot find dataset files in {data_dir}")
    
    train_df = pd.read_csv(train_files[0])
    val_df = pd.read_csv(val_files[0])
    test_df = pd.read_csv(test_files[0])
    
    return train_df, val_df, test_df


def create_sequences(features: np.ndarray, labels: np.ndarray, 
                    seq_len: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建时序样本
    
    Args:
        features: 特征数据 (N, feature_dim)
        labels: 标签数据 (N,)
        seq_len: 序列长度
        stride: 滑动步长
    
    Returns:
        X: (N', seq_len, feature_dim)
        y: (N',)
    """
    X, y = [], []
    
    for i in range(0, len(features) - seq_len + 1, stride):
        X.append(features[i:i + seq_len])
        y.append(labels[i + seq_len - 1])
    
    return np.array(X), np.array(y)


def prepare_data(data_dir: str, seq_len: int = 10, 
                feature_cols: Optional[List[str]] = None,
                scaler_type: str = 'standard',
                stride: int = 1,
                use_val: bool = True,
                already_normalized: bool = True) -> dict:
    """
    准备训练/验证/测试数据
    
    Args:
        data_dir: 数据目录
        seq_len: 序列长度
        feature_cols: 特征列 (默认: 8个归一化特征)
        scaler_type: 归一化类型
        stride: 滑动步长
        use_val: 是否使用验证集
        already_normalized: 数据是否已经归一化
    
    Returns:
        包含所有数据加载器的字典
    """
    if feature_cols is None:
        feature_cols = [
            'IAT_Mean_norm', 'IAT_Std_norm', 'Duration_norm', 
            'PktLen_Mean_norm', 'PktLen_Std_norm', 
            'Packets_s_norm', 'Bytes_s_norm', 'Idle_Mean_norm'
        ]
    
    # 加载数据
    train_df, val_df, test_df = load_raw_data(data_dir)
    
    if already_normalized:
        # 数据已归一化，直接使用
        train_features = train_df[feature_cols].values
        val_features = val_df[feature_cols].values
        test_features = test_df[feature_cols].values
        processor = DataProcessor(scaler_type='none')
    else:
        # 需要归一化
        all_train_df = pd.concat([train_df, val_df], ignore_index=True) if use_val else train_df
        processor = DataProcessor(scaler_type=scaler_type)
        processor.fit_transform(all_train_df, feature_cols)
        
        train_features = processor.transform(train_df[feature_cols].values)
        val_features = processor.transform(val_df[feature_cols].values)
        test_features = processor.transform(test_df[feature_cols].values)
    
    # 构建时序样本
    X_train, y_train = create_sequences(
        train_features, train_df['label'].values, seq_len, stride
    )
    X_val, y_val = create_sequences(
        val_features, val_df['label'].values, seq_len, stride
    )
    X_test, y_test = create_sequences(
        test_features, test_df['label'].values, seq_len, stride
    )
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': processor.scaler,
        'feature_cols': feature_cols
    }


def get_dataloaders(data_dir: str, seq_len: int = 10,
                   batch_size: int = 128,
                   feature_cols: Optional[List[str]] = None,
                   scaler_type: str = 'standard',
                   use_augment: bool = False,
                   stride: int = 1) -> dict:
    """
    获取数据加载器
    
    Returns:
        {
            'train_loader': DataLoader,
            'val_loader': DataLoader, 
            'test_loader': DataLoader,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'num_classes': int,
            'class_names': List[str]
        }
    """
    data = prepare_data(
        data_dir=data_dir,
        seq_len=seq_len,
        feature_cols=feature_cols,
        scaler_type=scaler_type,
        stride=stride
    )
    
    # 创建数据集
    train_dataset = SatelliteThreatDataset(data['X_train'], data['y_train'], 
                                          augment=use_augment)
    val_dataset = SatelliteThreatDataset(data['X_val'], data['y_val'])
    test_dataset = SatelliteThreatDataset(data['X_test'], data['y_test'])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # 类别信息
    class_names = ['Normal', 'DDoS', 'PortScan', 'Route']
    num_classes = len(class_names)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': data['scaler'],
        'feature_cols': data['feature_cols'],
        'num_classes': num_classes,
        'class_names': class_names,
        'data_info': {
            'train_size': len(data['X_train']),
            'val_size': len(data['X_val']),
            'test_size': len(data['X_test'])
        }
    }


def get_data_info(data_dir: str) -> dict:
    """获取数据集基本信息"""
    train_df, val_df, test_df = load_raw_data(data_dir)
    
    return {
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'features': list(train_df.columns[:-1]),
        'label_distribution': {
            'train': train_df['label'].value_counts().to_dict(),
            'val': val_df['label'].value_counts().to_dict(),
            'test': test_df['label'].value_counts().to_dict()
        }
    }


if __name__ == "__main__":
    # 测试数据加载
    data_dir = 'data'
    
    print("=" * 60)
    print("Data Loading Test")
    print("=" * 60)
    
    # 获取数据信息
    info = get_data_info(data_dir)
    print(f"Train samples: {info['train_samples']}")
    print(f"Val samples: {info['val_samples']}")
    print(f"Test samples: {info['test_samples']}")
    
    # 获取数据加载器
    dataloaders = get_dataloaders(
        data_dir=data_dir,
        seq_len=10,
        batch_size=128,
        feature_cols=['timestamp', 'packet_size']
    )
    
    print(f"\nData shapes:")
    print(f"  Train: {dataloaders['data_info']['train_size']}")
    print(f"  Val: {dataloaders['data_info']['val_size']}")
    print(f"  Test: {dataloaders['data_info']['test_size']}")
    
    # 测试一个batch
    for x, y in dataloaders['train_loader']:
        print(f"\nBatch shape: {x.shape}, Labels shape: {y.shape}")
        break
