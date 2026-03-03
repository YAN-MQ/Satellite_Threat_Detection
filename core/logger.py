"""
Logging Module
=============
提供统一的日志记录功能
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any


class TrainingLogger:
    """
    训练日志记录器
    
    负责记录训练过程中的各项指标和事件。
    """
    
    def __init__(self, log_dir: str = 'logs', experiment_name: str = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = experiment_name
        
        self.log_file = os.path.join(log_dir, f'{experiment_name}.log')
        self.metrics_file = os.path.join(log_dir, f'{experiment_name}_metrics.json')
        
        self.setup_logger()
        
        self.metrics_history = []
    
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict = None):
        """记录每个epoch的结果"""
        metrics = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train': train_metrics
        }
        
        if val_metrics:
            metrics['validation'] = val_metrics
        
        self.metrics_history.append(metrics)
        
        log_msg = f"Epoch {epoch}: Train Loss={train_metrics.get('loss', 0):.4f}, " \
                  f"Train Acc={train_metrics.get('accuracy', 0):.2f}%"
        
        if val_metrics:
            log_msg += f" | Val Loss={val_metrics.get('loss', 0):.4f}, " \
                      f"Val Acc={val_metrics.get('accuracy', 0):.2f}%"
        
        self.info(log_msg)
    
    def log_federated_round(self, round_idx: int, val_metrics: Dict, 
                           client_metrics: list):
        """记录联邦学习每轮的结果"""
        metrics = {
            'round': round_idx,
            'timestamp': datetime.now().isoformat(),
            'validation': val_metrics,
            'clients': client_metrics
        }
        
        self.metrics_history.append(metrics)
        
        log_msg = f"Round {round_idx}: Val Loss={val_metrics.get('loss', 0):.4f}, " \
                  f"Val Acc={val_metrics.get('accuracy', 0):.2f}%"
        
        self.info(log_msg)
    
    def log_config(self, config: Any):
        """记录配置信息"""
        config_dict = {}
        for section in ['model', 'train', 'data', 'federated']:
            if hasattr(config, section):
                section_obj = getattr(config, section)
                config_dict[section] = {
                    k: v for k, v in section_obj.__dict__.items() 
                    if not k.startswith('_')
                }
        
        self.info(f"Configuration: {json.dumps(config_dict, indent=2)}")
        
        config_file = os.path.join(self.log_dir, f'{self.experiment_name}_config.json')
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def save_metrics(self):
        """保存指标历史到JSON文件"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.info(f"Metrics saved to {self.metrics_file}")
    
    def close(self):
        """关闭日志记录器"""
        self.save_metrics()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


class ResultsWriter:
    """
    结果写入器
    
    将训练和评估结果写入指定目录。
    """
    
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.results = {}
    
    def add_result(self, key: str, value: Any):
        """添加结果"""
        self.results[key] = value
    
    def add_dict(self, results_dict: Dict):
        """批量添加结果"""
        self.results.update(results_dict)
    
    def save(self, filename: str = 'results.json'):
        """保存结果到JSON文件"""
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return filepath
    
    def save_csv(self, df, filename: str = 'results.csv'):
        """保存结果到CSV文件"""
        filepath = os.path.join(self.results_dir, filename)
        df.to_csv(filepath, index=False)
        return filepath
    
    def save_text(self, content: str, filename: str = 'results.txt'):
        """保存结果到文本文件"""
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return filepath


def get_logger(name: str = 'satellite_threat', log_dir: str = 'logs') -> TrainingLogger:
    """获取日志记录器"""
    return TrainingLogger(log_dir=log_dir, experiment_name=name)
