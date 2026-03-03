"""
Federated Client Module
=====================
联邦学习客户端：负责本地数据训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any


class FederatedClient:
    """
    联邦学习客户端
    
    每个客户端对应一个卫星节点，拥有本地数据集。
    
    Args:
        client_id: 客户端ID
        model: 模型实例
        train_loader: 本地训练数据加载器
        num_samples: 样本数量
        device: 训练设备
        config: 配置对象
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader,
        num_samples: int,
        device: torch.device,
        config: Any
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.num_samples = num_samples
        self.device = device
        self.config = config
        
        self.local_epochs = 0
    
    def set_parameters(self, global_params: Dict[str, torch.Tensor]):
        """
        设置全局模型参数
        
        Args:
            global_params: 全局模型参数字典
        """
        self.model.load_state_dict(global_params)
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        获取本地模型参数
        
        Returns:
            模型参数字典
        """
        return {
            name: param.clone().detach() 
            for name, param in self.model.state_dict().items()
        }
    
    def train(
        self, 
        epochs: int, 
        criterion=None,
        use_fedprox: bool = False,
        global_params: Dict[str, torch.Tensor] = None,
        mu: float = 0.01
    ) -> tuple:
        """
        本地训练
        
        Args:
            epochs: 训练轮数
            criterion: 损失函数
            use_fedprox: 是否使用FedProx
            global_params: 全局模型参数（用于FedProx）
            mu: FedProx正则化系数
        
        Returns:
            (平均损失, 平均准确率)
        """
        self.model.train()
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay
        )
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                if use_fedprox and global_params is not None:
                    prox_loss = 0.0
                    for name, param in self.model.named_parameters():
                        prox_loss += torch.sum(
                            (param - global_params[name]) ** 2
                        )
                    loss = loss + (mu / 2) * prox_loss
                
                loss.backward()
                
                if self.config.train.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.train.gradient_clip
                    )
                
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        self.local_epochs += epochs
        
        avg_loss = total_loss / (len(self.train_loader) * epochs)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader, criterion) -> Dict[str, float]:
        """
        评估本地模型
        
        Args:
            val_loader: 验证数据加载器
            criterion: 损失函数
        
        Returns:
            评估结果字典
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
    
    def __repr__(self):
        return f"FederatedClient(id={self.client_id}, samples={self.num_samples})"
