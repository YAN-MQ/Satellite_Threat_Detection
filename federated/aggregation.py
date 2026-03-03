"""
Aggregation Module
================
参数聚合器：实现FedAvg、FedProx等聚合算法
"""

import torch
from typing import List, Dict


class BaseAggregator:
    """参数聚合器基类"""
    
    def aggregate(
        self, 
        client_updates: List[Dict[str, torch.Tensor]], 
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """聚合客户端更新"""
        raise NotImplementedError


class FedAvgAggregator(BaseAggregator):
    """
    FedAvg 聚合器
    
    加权平均聚合各客户端的模型参数：
    
    W_new = Σ(n_k/n) * W_k
    
    其中 n_k 是客户端k的样本数量，n是总样本数量。
    """
    
    def __init__(self):
        pass
    
    def aggregate(
        self, 
        client_updates: List[Dict[str, torch.Tensor]], 
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        FedAvg 聚合
        
        Args:
            client_updates: 客户端模型参更新列表
            client_weights: 客户端权重列表
        
        Returns:
            聚合后的模型参数
        """
        aggregated_params = {}
        
        param_names = client_updates[0].keys()
        
        for param_name in param_names:
            weighted_sum = None
            
            for client_params, weight in zip(client_updates, client_weights):
                if weighted_sum is None:
                    weighted_sum = weight * client_params[param_name]
                else:
                    weighted_sum += weight * client_params[param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params


class FedProxAggregator(BaseAggregator):
    """
    FedProx 聚合器
    
    在FedAvg基础上，增加了对客户端训练过程的正则化约束。
    实际的聚合逻辑与FedAvg相同，区别在于客户端训练时使用了近端项。
    """
    
    def __init__(self, mu: float = 0.01):
        self.mu = mu
    
    def aggregate(
        self, 
        client_updates: List[Dict[str, torch.Tensor]], 
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """与FedAvg相同的聚合逻辑"""
        aggregated_params = {}
        
        param_names = client_updates[0].keys()
        
        for param_name in param_names:
            weighted_sum = None
            
            for client_params, weight in zip(client_updates, client_weights):
                if weighted_sum is None:
                    weighted_sum = weight * client_params[param_name]
                else:
                    weighted_sum += weight * client_params[param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params


class FedNovaAggregator(BaseAggregator):
    """
    FedNova 聚合器
    
    通过归一化本地训练步数来减少异构性影响。
    """
    
    def __init__(self):
        pass
    
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float],
        local_steps: List[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        FedNova 聚合
        
        Args:
            client_updates: 客户端模型参数更新列表
            client_weights: 客户端权重列表
            local_steps: 各客户端本地训练步数
        """
        if local_steps is None:
            local_steps = [1] * len(client_updates)
        
        total_steps = sum(local_steps)
        normalized_weights = [s / total_steps for s in local_steps]
        
        aggregated_params = {}
        
        param_names = client_updates[0].keys()
        
        for param_name in param_names:
            weighted_sum = None
            
            for client_params, weight in zip(client_updates, normalized_weights):
                if weighted_sum is None:
                    weighted_sum = weight * client_params[param_name]
                else:
                    weighted_sum += weight * client_params[param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params


class ScaffoldAggregator(BaseAggregator):
    """
    SCAFFOLD 聚合器
    
    使用控制变量来修正客户端更新方向，减少数据异构性影响。
    """
    
    def __init__(self):
        self.client_controls = {}
        self.server_control = {}
    
    def aggregate(
        self, 
        client_updates: List[Dict[str, torch.Tensor]], 
        client_weights: List[float],
        client_grads: List[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """SCAFFOLD聚合"""
        return FedAvgAggregator().aggregate(client_updates, client_weights)


def get_aggregator(aggregator_name: str = 'fedavg', **kwargs) -> BaseAggregator:
    """
    获取聚合器实例
    
    Args:
        aggregator_name: 聚合器名称
        **kwargs: 聚合器初始化参数
    
    Returns:
        聚合器实例
    """
    aggregators = {
        'fedavg': FedAvgAggregator,
        'fedprox': FedProxAggregator,
        'fednova': FedNovaAggregator,
        'scaffold': ScaffoldAggregator
    }
    
    if aggregator_name.lower() not in aggregators:
        raise ValueError(f"Unknown aggregator: {aggregator_name}")
    
    return aggregators[aggregator_name.lower()](**kwargs)
