"""
Metrics Module
=============
提供各类评估指标：准确率、F1分数、精确率、召回率等
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, confusion_matrix, classification_report
)


class AccuracyMetric:
    """准确率计算"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total = 0
        self.correct = 0
    
    def update(self, outputs, targets):
        _, predicted = outputs.max(1)
        self.correct += predicted.eq(targets).sum().item()
        self.total += targets.size(0)
    
    def compute(self):
        return 100.0 * self.correct / self.total if self.total > 0 else 0.0


class F1ScoreMetric:
    """F1分数计算 (多分类)"""
    
    def __init__(self, average='macro'):
        self.average = average
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, outputs, targets):
        _, predicted = outputs.max(1)
        self.predictions.extend(predicted.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        if len(self.predictions) == 0:
            return 0.0
        return f1_score(
            self.targets, self.predictions, 
            average=self.average, zero_division=0
        ) * 100


class PrecisionRecallMetric:
    """精确率和召回率计算"""
    
    def __init__(self, average='macro'):
        self.average = average
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, outputs, targets):
        _, predicted = outputs.max(1)
        self.predictions.extend(predicted.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        if len(self.predictions) == 0:
            return 0.0, 0.0
        
        precision = precision_score(
            self.targets, self.predictions, 
            average=self.average, zero_division=0
        )
        recall = recall_score(
            self.targets, self.predictions, 
            average=self.average, zero_division=0
        )
        
        return precision * 100, recall * 100


class ConfusionMatrixMetric:
    """混淆矩阵计算"""
    
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, outputs, targets):
        _, predicted = outputs.max(1)
        self.predictions.extend(predicted.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        return confusion_matrix(self.targets, self.predictions)
    
    def get_report(self):
        return classification_report(
            self.targets, self.predictions, 
            target_names=['Normal', 'DoS', 'Probe', 'Route'],
            zero_division=0
        )


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.metrics = {
            'accuracy': AccuracyMetric(),
            'f1': F1ScoreMetric(average='macro'),
            'precision_recall': PrecisionRecallMetric(average='macro'),
            'confusion_matrix': ConfusionMatrixMetric(num_classes)
        }
    
    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
    
    def update(self, outputs, targets):
        for metric in self.metrics.values():
            metric.update(outputs, targets)
    
    def get_results(self):
        results = {
            'accuracy': self.metrics['accuracy'].compute(),
            'f1': self.metrics['f1'].compute()
        }
        
        precision, recall = self.metrics['precision_recall'].compute()
        results['precision'] = precision
        results['recall'] = recall
        
        return results
    
    def get_confusion_matrix(self):
        return self.metrics['confusion_matrix'].compute()
    
    def get_classification_report(self):
        return self.metrics['confusion_matrix'].get_report()
