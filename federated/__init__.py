"""
Federated Learning Module
========================
提供联邦学习的核心组件：服务端、客户端、参数聚合、聚合策略等
"""

from .server import FederatedServer
from .client import FederatedClient
from .aggregation import FedAvgAggregator, FedProxAggregator
from .strategies import StrategyRegistry, get_strategy

__all__ = [
    'FederatedServer',
    'FederatedClient',
    'FedAvgAggregator',
    'FedProxAggregator',
    'StrategyRegistry',
    'get_strategy'
]
