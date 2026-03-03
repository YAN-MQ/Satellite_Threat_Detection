"""
Base Trainer Module
=================
提供训练器基类，封装训练和验证逻辑
"""

import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from core.metrics import MetricsCollector


class BaseTrainer:
    """
    基础训练器类
    
    封装模型训练、验证、Early Stopping等通用逻辑。
    
    Args:
        model: PyTorch模型
        optimizer: 优化器
        criterion: 损失函数
        device: 训练设备
        config: 配置对象
    """
    
    def __init__(self, model, optimizer, criterion, device, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        self.train_metrics = MetricsCollector(num_classes=config.model.num_classes)
        self.val_metrics = MetricsCollector(num_classes=config.model.num_classes)
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            if self.config.train.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.train.gradient_clip
                )
            
            self.optimizer.step()
            
            self.train_metrics.update(outputs, labels)
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': self.train_metrics.get_results()['accuracy']
            })
        
        results = self.train_metrics.get_results()
        results['loss'] = total_loss / len(train_loader)
        
        return results
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                self.val_metrics.update(outputs, labels)
                total_loss += loss.item()
        
        results = self.val_metrics.get_results()
        results['loss'] = total_loss / len(val_loader)
        
        return results
    
    def save_checkpoint(self, filepath, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir, 
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        return checkpoint
    
    def should_stop(self):
        """判断是否应该停止训练"""
        if not self.config.train.use_early_stopping:
            return False
        
        return self.epochs_without_improvement >= self.config.train.patience
    
    def on_epoch_end(self, val_results):
        """每个epoch结束后的回调"""
        val_acc = val_results['accuracy']
        
        is_best = val_acc > self.best_val_acc
        if is_best:
            self.best_val_acc = val_acc
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        return is_best
    
    def fit(self, train_loader, val_loader, epochs):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
        """
        history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            train_results = self.train_epoch(train_loader)
            val_results = self.validate(val_loader)
            
            is_best = self.on_epoch_end(val_results)
            
            history['epoch'].append(self.current_epoch)
            history['train_loss'].append(train_results['loss'])
            history['train_acc'].append(train_results['accuracy'])
            history['val_loss'].append(val_results['loss'])
            history['val_acc'].append(val_results['accuracy'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"\nEpoch {self.current_epoch}/{epochs}")
            print(f"Train Loss: {train_results['loss']:.4f}, "
                  f"Train Acc: {train_results['accuracy']:.2f}%")
            print(f"Val Loss: {val_results['loss']:.4f}, "
                  f"Val Acc: {val_results['accuracy']:.2f}%")
            
            if is_best:
                print(f"✓ New best model! Val Acc: {val_results['accuracy']:.2f}%")
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, 'best_model.pth'),
                    is_best=True
                )
            
            if self.should_stop():
                print(f"\nEarly stopping triggered after {self.current_epoch} epochs")
                break
        
        return history
