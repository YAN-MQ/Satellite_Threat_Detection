"""
Data loading and processing utilities
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


class NetworkTrafficDataset(Dataset):
    """Dataset for network traffic classification
    
    Args:
        X: Feature array of shape (samples, window, features)
        y: Label array of shape (samples,)
    """
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataSplitter:
    """Split data into train/val/test sets
    
    Args:
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    """
    
    def __init__(self, train_ratio=0.6, val_ratio=0.2):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
    
    def split(self, df):
        """Split dataframe into train/val/test
        
        Args:
            df: DataFrame with features
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        return train_df, val_df, test_df


class DataProcessor:
    """Process network traffic data
    
    Args:
        feature_columns: List of feature column names
        window_size: Sliding window size
        stride: Sliding window stride
    """
    
    def __init__(self, feature_columns, window_size=10, stride=1):
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.stride = stride
        self.scaler = MinMaxScaler()
    
    def normalize(self, train_data, val_data, test_data):
        """Normalize data using MinMaxScaler fitted on training data
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            
        Returns:
            tuple: Normalized (train, val, test) data
        """
        train_normalized = self.scaler.fit_transform(train_data)
        val_normalized = self.scaler.transform(val_data)
        test_normalized = self.scaler.transform(test_data)
        
        return train_normalized, val_normalized, test_normalized
    
    def create_sequences(self, X):
        """Create sequences using sliding window
        
        Args:
            X: Feature array of shape (samples, features)
            
        Returns:
            np.array: Sequences of shape (n_sequences, window_size, n_features)
        """
        sequences = []
        for i in range(0, len(X) - self.window_size + 1, self.stride):
            sequences.append(X[i:i + self.window_size])
        
        return np.array(sequences, dtype=np.float32)
    
    def process(self, train_df, val_df, test_df):
        """Full data processing pipeline
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        train_norm, val_norm, test_norm = self.normalize(
            train_df[self.feature_columns].values,
            val_df[self.feature_columns].values,
            test_df[self.feature_columns].values
        )
        
        X_train = self.create_sequences(train_norm)
        X_val = self.create_sequences(val_norm)
        X_test = self.create_sequences(test_norm)
        
        # Labels: use the label of the last packet in each window
        y_train = np.full(len(X_train), train_df['label'].iloc[self.window_size - 1])
        y_val = np.full(len(X_val), val_df['label'].iloc[self.window_size - 1])
        y_test = np.full(len(X_test), test_df['label'].iloc[self.window_size - 1])
        
        # Fix: get labels from corresponding windows
        y_train = np.array([train_df['label'].iloc[i + self.window_size - 1] 
                           for i in range(len(X_train))])
        y_val = np.array([val_df['label'].iloc[i + self.window_size - 1] 
                         for i in range(len(X_val))])
        y_test = np.array([test_df['label'].iloc[i + self.window_size - 1] 
                          for i in range(len(X_test))])
        
        return X_train, y_train, X_val, y_val, X_test, y_test


def load_npz_data(data_dir):
    """Load data from npz files
    
    Args:
        data_dir: Path to directory containing train.npz, val.npz, test.npz
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    train = np.load(os.path.join(data_dir, 'train.npz'))
    val = np.load(os.path.join(data_dir, 'val.npz'))
    test = np.load(os.path.join(data_dir, 'test.npz'))
    
    return (
        train['X'], train['y'],
        val['X'], val['y'],
        test['X'], test['y']
    )


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, 
                       batch_size=64, num_workers=4, pin_memory=True):
    """Create DataLoaders for training
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = NetworkTrafficDataset(X_train, y_train)
    val_dataset = NetworkTrafficDataset(X_val, y_val)
    test_dataset = NetworkTrafficDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
