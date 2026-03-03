"""
Federated Server Module
=====================
联邦学习服务端：负责模型聚合和全局模型管理
"""

import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

from federated.aggregation import FedAvgAggregator


class FederatedServer:
    """
    联邦学习服务端
    
    负责：
    1. 初始化全局模型
    2. 分发模型给各客户端
    3. 聚合客户端更新
    4. 评估全局模型
    
    Args:
        model: 全局模型
        device: 训练设备
        aggregator: 聚合器实例
    """
    
    def __init__(self, model, device, aggregator=None):
        self.model = model
        self.device = device
        self.aggregator = aggregator or FedAvgAggregator()
        
        self.global_round = 0
        self.history = {
            'round': [],
            'val_loss': [],
            'val_acc': [],
            'avg_client_loss': [],
            'avg_client_acc': []
        }
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """获取全局模型参数"""
        return {
            name: param.clone().detach() 
            for name, param in self.model.state_dict().items()
        }
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """设置全局模型参数"""
        self.model.load_state_dict(parameters)
    
    def broadcast_parameters(self) -> Dict[str, torch.Tensor]:
        """
        广播模型参数给所有客户端
        
        Returns:
            模型参数字典
        """
        return self.get_model_parameters()
    
    def aggregate_updates(
        self, 
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        聚合客户端更新
        
        Args:
            client_updates: 各客户端的模型更新列表
            client_weights: 各客户端的权重（样本数量比例）
        
        Returns:
            聚合后的模型参数
        """
        return self.aggregator.aggregate(client_updates, client_weights)
    
    def evaluate(self, val_loader, criterion) -> Dict[str, float]:
        """
        评估全局模型
        
        Args:
            val_loader: 验证数据加载器
            criterion: 损失函数
        
        Returns:
            包含loss和accuracy的字典
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def save_checkpoint(self, filepath: str):
        """保存服务端检查点"""
        torch.save({
            'round': self.global_round,
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """加载服务端检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_round = checkpoint['round']
        self.history = checkpoint.get('history', self.history)
        
        return checkpoint
    
    def fit(
        self,
        clients: List[Any],
        val_loader,
        criterion,
        num_rounds: int,
        local_epochs: int,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        执行联邦学习训练流程
        
        Args:
            clients: 联邦学习客户端列表
            val_loader: 验证数据加载器
            criterion: 损失函数
            num_rounds: 通信轮数
            local_epochs: 每轮的本地训练轮数
            verbose: 是否打印训练过程
        
        Returns:
            训练历史记录
        """
        client_weights = [client.num_samples for client in clients]
        total_samples = sum(client_weights)
        client_weights = [w / total_samples for w in client_weights]
        
        for round_idx in range(num_rounds):
            self.global_round = round_idx + 1
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Round {self.global_round}/{num_rounds}")
                print(f"{'='*60}")
            
            global_params = self.broadcast_parameters()
            
            client_updates = []
            client_losses = []
            client_accs = []
            
            for client in clients:
                client.set_parameters(global_params)
                
                loss, acc = client.train(
                    epochs=local_epochs,
                    criterion=criterion
                )
                
                client_updates.append(client.get_parameters())
                client_losses.append(loss)
                client_accs.append(acc)
                
                if verbose:
                    print(f"Client {client.client_id}: Loss={loss:.4f}, Acc={acc:.2f}%")
            
            aggregated_params = self.aggregate_updates(
                client_updates, client_weights
            )
            self.set_model_parameters(aggregated_params)
            
            val_results = self.evaluate(val_loader, criterion)
            
            self.history['round'].append(self.global_round)
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_acc'].append(val_results['accuracy'])
            self.history['avg_client_loss'].append(np.mean(client_losses))
            self.history['avg_client_acc'].append(np.mean(client_accs))
            
            if verbose:
                print(f"\nGlobal Model - Val Loss: {val_results['loss']:.4f}, "
                      f"Val Acc: {val_results['accuracy']:.2f}%")
            
            if self.global_round % 10 == 0:
                self.save_checkpoint(f"checkpoints/fedavg_round_{self.global_round}.pth")
        
        return self.history
