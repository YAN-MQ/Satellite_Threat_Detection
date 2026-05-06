"""
Training utilities
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


class Trainer:
    """Trainer for DSC-CBAM-LSTM model
    
    Args:
        model: PyTorch model
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
    """
    
    def __init__(self, model, criterion, optimizer, scheduler=None, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.model.to(self.device)
    
    def train_epoch(self, train_loader):
        """Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            tuple: (val_loss, val_acc, predictions, ground_truth)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        return val_loss, val_acc, all_preds, all_labels
    
    def train(self, train_loader, val_loader, num_epochs, save_path=None, 
              early_stopping_patience=10):
        """Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_path: Path to save best model
            early_stopping_patience: Patience for early stopping
            
        Returns:
            dict: Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_acc)
            
            # History
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        # Save model
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_state, save_path)
            print(f"Best model saved to {save_path} (Val Acc: {best_acc:.4f})")
        
        return history
    
    def evaluate(self, test_loader):
        """Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                outputs = self.model(X)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        
        return metrics


def get_optimizer(model, optimizer_name='adamw', lr=0.001, weight_decay=0.01, **kwargs):
    """Create optimizer
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Optimizer instance
    """
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
    }
    
    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizers[optimizer_name.lower()](model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)


def get_scheduler(optimizer, scheduler_name='plateau', **kwargs):
    """Create learning rate scheduler
    
    Args:
        optimizer: Optimizer
        scheduler_name: Name of scheduler
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_name == 'plateau':
        return ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            patience=kwargs.get('patience', 5),
            factor=kwargs.get('factor', 0.5)
        )
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50)
        )
    else:
        return None
