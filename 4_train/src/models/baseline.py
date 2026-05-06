"""
Baseline Comparison Models
Traditional ML (RF, ID3) and Deep Learning (MLP, CNN-LSTM)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time


class MLP(nn.Module):
    """Multi-Layer Perceptron for sequence classification"""
    
    def __init__(self, input_dim=18, num_classes=4, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        h1, h2, h3 = hidden_dims
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 10, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


class CNN_LSTM(nn.Module):
    """Standard CNN-LSTM model"""

    def __init__(self, input_dim=18, num_classes=4, conv_channels=(32, 64), hidden_dim=64, bidirectional=False, dropout=0.3):
        super().__init__()
        c1, c2 = conv_channels
        self.conv1 = nn.Conv1d(input_dim, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(c2, hidden_dim, num_layers=1, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)


class BaselineTrainer:
    """Trainer for baseline models (RF, ID3)"""
    
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.model = None
        self.training_time = 0
        
    def create_model(self):
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                n_jobs=-1,
                random_state=42
            )
        elif self.model_type == 'id3':
            self.model = DecisionTreeClassifier(
                criterion='entropy',
                max_depth=20,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """Train the model"""
        # Flatten sequences for traditional ML
        X_flat = X_train.reshape(X_train.shape[0], -1)
        
        start_time = time.time()
        self.model.fit(X_flat, y_train)
        self.training_time = time.time() - start_time
        
        return self.training_time
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        X_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Predictions
        start_time = time.time()
        y_pred = self.model.predict(X_flat)
        inference_time = time.time() - start_time
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'inference_time': inference_time,
            'training_time': self.training_time,
            'predictions': y_pred
        }
        
        return metrics
    
    def predict(self, X):
        """Predict labels"""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)


class DeepLearningBaseline:
    """Trainer for deep learning baselines (MLP, CNN-LSTM)"""
    
    def __init__(self, model_type='mlp', device='cpu'):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.training_time = 0
        
    def create_model(self, input_dim=18, num_classes=4):
        if self.model_type == 'mlp':
            self.model = MLP(input_dim, num_classes)
        elif self.model_type == 'cnn_lstm':
            self.model = CNN_LSTM(input_dim, num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.to(self.device)
        return self.model
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        start_time = time.time()
        
        best_acc = 0
        best_state = None
        
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            
            val_acc = correct / total
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        
        self.training_time = time.time() - start_time
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        return self.training_time
    
    def evaluate(self, test_loader):
        """Evaluate the model"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        start_time = time.time()
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                outputs = self.model(X)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        
        inference_time = time.time() - start_time
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'inference_time': inference_time,
            'training_time': self.training_time,
            'predictions': np.array(all_preds)
        }
        
        return metrics


def get_baseline_model(model_name):
    """Get baseline model by name"""
    models = {
        'rf': lambda: BaselineTrainer('rf'),
        'id3': lambda: BaselineTrainer('id3'),
        'mlp': lambda: DeepLearningBaseline('mlp'),
        'cnn_lstm': lambda: DeepLearningBaseline('cnn_lstm'),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown baseline model: {model_name}")
    
    return models[model_name]()
