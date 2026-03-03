"""
训练脚本 - DSC-CBAM-LSTM V2模型
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

from config import config, get_config, update_config
from models.dsc_cbam_lstm_v2 import DSC_CBAM_LSTM_V2
from utils.data_processor import get_dataloaders, get_data_info
from core.logger import TrainingLogger, ResultsWriter
from core.losses import FocalLoss, WeightedCrossEntropyLoss, LabelSmoothingLoss


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device, gradient_clip=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(val_loader), 100.0 * correct / total


def train(config=None, logger=None):
    """训练主函数"""
    if config is None:
        config = get_config()

    # 设置随机种子
    set_seed(config.seed)

    # 设备
    device = config.device_obj
    msg = f"Using device: {device}"
    print(msg)
    if logger:
        logger.info(msg)

    # 加载数据
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    dataloaders = get_dataloaders(
        data_dir=config.data.data_dir,
        seq_len=config.data.seq_len,
        batch_size=config.train.batch_size,
        feature_cols=config.data.feature_cols,
        scaler_type=config.data.scaler_type,
        use_augment=config.data.use_augment,
        stride=config.data.stride
    )

    print(f"Train samples: {dataloaders['data_info']['train_size']}")
    print(f"Val samples: {dataloaders['data_info']['val_size']}")
    print(f"Test samples: {dataloaders['data_info']['test_size']}")

    # 创建V2模型
    print("\n" + "=" * 60)
    print("Creating DSC-CBAM-LSTM V2 model...")
    print("=" * 60)

    model = DSC_CBAM_LSTM_V2(
        seq_len=config.model.seq_len,
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout
    )
    model = model.to(device)

    # 打印模型统计
    num_params = model.count_parameters()
    print(f"Model: DSC-CBAM-LSTM V2")
    print(f"Parameters: {num_params:,}")
    print(f"Target: < 100K")
    print(f"Status: {'✓ PASS' if num_params < 100000 else '✗ FAIL'}")

    # 损失函数选择
    loss_type = config.train.loss_type
    label_smoothing = config.train.label_smoothing

    if loss_type == "focal":
        if config.train.use_class_weight:
            class_counts = np.bincount(dataloaders['data_info'].get('train_labels',
                                np.array([y for _, y in dataloaders['train_loader'].dataset])))
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.FloatTensor(class_weights).to(device)
            criterion = FocalLoss(alpha=class_weights, gamma=2.0)
            print(f"Using Focal Loss with class weights: {class_weights.cpu().numpy()}")
        else:
            criterion = FocalLoss(gamma=2.0)
            print("Using Focal Loss")
    elif loss_type == "label_smoothing":
        criterion = LabelSmoothingLoss(num_classes=config.model.num_classes, smoothing=label_smoothing)
        print(f"Using Label Smoothing Loss (smoothing={label_smoothing})")
    elif loss_type == "weighted":
        criterion = WeightedCrossEntropyLoss(num_classes=config.model.num_classes)
        print("Using Weighted Cross Entropy Loss")
    else:
        if config.train.use_class_weight:
            class_counts = np.bincount(dataloaders['data_info'].get('train_labels',
                                np.array([y for _, y in dataloaders['train_loader'].dataset])))
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.FloatTensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using CrossEntropyLoss with class weights: {class_weights.cpu().numpy()}")
        else:
            criterion = nn.CrossEntropyLoss()
            print("Using CrossEntropyLoss")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay
    )

    # 学习率调度器
    scheduler = None
    if config.train.use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

    # 训练循环
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_acc = 0.0
    best_epoch = 0
    no_improve_count = 0
    history = []

    for epoch in range(1, config.train.epochs + 1):
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_epoch(
            model, dataloaders['train_loader'],
            criterion, optimizer, device,
            config.train.gradient_clip
        )

        # 验证
        val_loss, val_acc = validate(
            model, dataloaders['val_loader'],
            criterion, device
        )

        # 学习率调度
        if scheduler:
            scheduler.step(val_acc)

        # 记录历史
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # 打印进度
        elapsed = time.time() - start_time
        msg = f"Epoch [{epoch}/{config.train.epochs}] " \
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | " \
              f"Acc: {train_acc:.2f}%/{val_acc:.2f}% | " \
              f"Time: {elapsed:.2f}s"
        print(msg)
        if logger:
            logger.log_epoch(epoch,
                           {'loss': train_loss, 'accuracy': train_acc},
                           {'loss': val_loss, 'accuracy': val_acc})

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve_count = 0

            # 保存模型
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': {
                    'model': config.model.__dict__,
                    'data': config.data.__dict__,
                    'train': config.train.__dict__
                }
            }
            save_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f"  → Best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            no_improve_count += 1

        # 早停
        if config.train.use_early_stopping and no_improve_count >= config.train.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # 训练完成
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    # 加载最佳模型进行最终评估
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, history, dataloaders, best_val_acc, best_epoch


def main():
    # 创建日志记录器
    experiment_name = f"v2_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(log_dir='logs', experiment_name=experiment_name)
    results_writer = ResultsWriter(results_dir='results')

    # 记录配置
    logger.log_config(config)
    logger.info(f"Starting DSC-CBAM-LSTM V2 training experiment: {experiment_name}")

    # 训练模型
    model, history, dataloaders, best_val_acc, best_epoch = train(config, logger)

    # 记录最终结果
    results_writer.add_result('experiment_name', experiment_name)
    results_writer.add_result('best_val_acc', best_val_acc)
    results_writer.add_result('best_epoch', best_epoch)
    results_writer.add_result('training_history', {'epochs': history})
    results_writer.add_result('num_epochs', len(history))

    # 保存结果
    results_path = results_writer.save(f'{experiment_name}_results.json')
    logger.info(f"Results saved to {results_path}")

    logger.close()
    print(f"\nTraining logs saved to logs/{experiment_name}.log")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
