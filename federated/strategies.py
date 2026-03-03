"""
Federated Learning Strategies Module
==================================
联邦学习策略注册表：管理各种聚合策略
"""

from typing import Dict, Callable, Any
from federated.aggregation import (
    FedAvgAggregator, 
    FedProxAggregator, 
    FedNovaAggregator,
    ScaffoldAggregator
)


class StrategyRegistry:
    """
    联邦学习策略注册表
    
    用于注册和管理各种联邦学习聚合策略。
    """
    
    _strategies: Dict[str, Callable] = {
        'fedavg': FedAvgAggregator,
        'fedprox': FedProxAggregator,
        'fednova': FedNovaAggregator,
        'scaffold': ScaffoldAggregator
    }
    
    @classmethod
    def register(cls, name: str, strategy_class: Callable):
        """注册新的策略"""
        cls._strategies[name.lower()] = strategy_class
    
    @classmethod
    def get(cls, name: str, **kwargs) -> Any:
        """获取策略实例"""
        name = name.lower()
        if name not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy: {name}. "
                f"Available strategies: {available}"
            )
        return cls._strategies[name](**kwargs)
    
    @classmethod
    def list_strategies(cls) -> list:
        """列出所有可用策略"""
        return list(cls._strategies.keys())


def get_strategy(strategy_name: str = 'fedavg', **kwargs):
    """
    获取联邦学习策略
    
    Args:
        strategy_name: 策略名称
        **kwargs: 策略初始化参数
    
    Returns:
        策略实例
    """
    return StrategyRegistry.get(strategy_name, **kwargs)


class FedAvgStrategy:
    """
    FedAvg 策略配置
    
    示例用法:
        strategy = FedAvgStrategy()
        aggregator = strategy.get_aggregator()
    """
    
    def __init__(
        self,
        name: str = 'fedavg',
        client_selection: str = 'all',
        min_clients: int = None
    ):
        self.name = name
        self.client_selection = client_selection
        self.min_clients = min_clients
    
    def get_aggregator(self, **kwargs):
        return get_strategy(self.name, **kwargs)
    
    def __repr__(self):
        return f"FedAvgStrategy(name={self.name}, selection={self.client_selection})"


class FedProxStrategy:
    """
    FedProx 策略配置
    
    增加了近端项正则化来限制本地模型偏离全局模型。
    """
    
    def __init__(
        self,
        name: str = 'fedprox',
        mu: float = 0.01,
        client_selection: str = 'all'
    ):
        self.name = name
        self.mu = mu
        self.client_selection = client_selection
    
    def get_aggregator(self):
        return get_strategy(self.name, mu=self.mu)
    
    def __repr__(self):
        return f"FedProxStrategy(mu={self.mu}, selection={self.client_selection})"


class AdaptiveStrategy:
    """
    自适应策略：根据客户端数据分布自动选择最佳策略
    """
    
    def __init__(self):
        self.current_strategy = 'fedavg'
    
    def select_strategy(self, client_data_stats: Dict) -> str:
        """
        根据客户端数据统计选择策略
        
        Args:
            client_data_stats: 客户端数据统计
        
        Returns:
            策略名称
        """
        non_iid_ratio = client_data_stats.get('non_iid_ratio', 0)
        
        if non_iid_ratio > 0.5:
            self.current_strategy = 'fedprox'
        else:
            self.current_strategy = 'fedavg'
        
        return self.current_strategy
    
    def get_aggregator(self):
        return get_strategy(self.current_strategy)


STRATEGY_CONFIGS = {
    'fedavg': {
        'description': 'Federated Averaging',
        'use_prox': False,
        'parameters': []
    },
    'fedprox': {
        'description': 'Federated Proximal',
        'use_prox': True,
        'parameters': ['mu']
    },
    'fednova': {
        'description': 'Federated Normalized Averaging',
        'use_prox': False,
        'parameters': ['local_steps']
    },
    'scaffold': {
        'description': 'Stochastic Controlled Averaging for Federated Learning',
        'use_prox': False,
        'parameters': ['control_variates']
    }
}
