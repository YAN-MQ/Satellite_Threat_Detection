"""
Core Training Module
====================
提供训练相关的核心功能：损失函数、评估指标、训练器基类、日志记录等
"""

from .losses import FocalLoss, WeightedCrossEntropyLoss
from .metrics import AccuracyMetric, F1ScoreMetric, PrecisionRecallMetric
from .trainer import BaseTrainer
from .logger import TrainingLogger, ResultsWriter, get_logger

__all__ = [
    'FocalLoss',
    'WeightedCrossEntropyLoss',
    'AccuracyMetric',
    'F1ScoreMetric',
    'PrecisionRecallMetric',
    'BaseTrainer',
    'TrainingLogger',
    'ResultsWriter',
    'get_logger'
]
