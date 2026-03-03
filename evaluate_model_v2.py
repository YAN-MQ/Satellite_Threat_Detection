"""
评估脚本 - DSC-CBAM-LSTM V2模型
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from config import config, get_config
from models.dsc_cbam_lstm_v2 import DSC_CBAM_LSTM_V2
from utils.data_processor import get_dataloaders


def load_model(checkpoint_path: str, device):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从checkpoint中获取配置
    model_config = checkpoint.get('config', {}).get('model', {})

    # 创建V2模型
    model = DSC_CBAM_LSTM_V2(
        seq_len=model_config.get('seq_len', 10),
        input_dim=model_config.get('input_dim', 8),
        hidden_dim=model_config.get('hidden_dim', 32),
        num_classes=model_config.get('num_classes', 4),
        dropout=model_config.get('dropout', 0.2)
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint.get('epoch', 0), checkpoint.get('val_acc', 0)


def predict(model, test_loader, device):
    """预测并收集结果"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    return np.array(all_labels), np.array(all_preds), all_probs


def evaluate_model(model, test_loader, device, class_names):
    """评估模型并生成报告"""
    y_true, y_pred, y_prob = predict(model, test_loader, device)

    # 准确率
    accuracy = accuracy_score(y_true, y_pred) * 100

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 分类报告
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    return accuracy, cm, report, y_true, y_pred


def measure_inference_time(model, test_loader, device, num_iterations=1000):
    """测量推理时间"""
    model.eval()

    # 预热
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= 10:
                break
            inputs = inputs.to(device)
            _ = model(inputs)

    # 测量
    start_time = time.time()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_iterations:
                break
            inputs = inputs.to(device)
            _ = model(inputs)

    end_time = time.time()
    avg_time_ms = (end_time - start_time) / num_iterations * 1000

    return avg_time_ms


def plot_confusion_matrix(cm, class_names, save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - DSC-CBAM-LSTM V2')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate(checkpoint_path: str = None, config=None):
    """评估主函数"""
    if config is None:
        config = get_config()

    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pth')

    # 设备
    device = config.device_obj
    print(f"Using device: {device}")

    # 加载数据
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    dataloaders = get_dataloaders(
        data_dir=config.data.data_dir,
        seq_len=config.data.seq_len,
        batch_size=config.train.batch_size,
        feature_cols=config.data.feature_cols,
        scaler_type=config.data.scaler_type
    )

    # 加载V2模型
    print("\n" + "=" * 60)
    print("Loading DSC-CBAM-LSTM V2 model...")
    print("=" * 60)

    model, epoch, val_acc = load_model(checkpoint_path, device)

    num_params = model.count_parameters()
    print(f"Model: DSC-CBAM-LSTM V2")
    print(f"Parameters: {num_params:,}")
    print(f"Training epoch: {epoch}")
    print(f"Validation accuracy: {val_acc:.2f}%")

    # 评估
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    accuracy, cm, report, y_true, y_pred = evaluate_model(
        model, dataloaders['test_loader'],
        device, dataloaders['class_names']
    )

    # 打印结果
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # 推理时间
    print("\n" + "=" * 60)
    print("Measuring inference time...")
    print("=" * 60)

    avg_inference_time = measure_inference_time(
        model, dataloaders['test_loader'], device
    )

    print(f"Average inference time: {avg_inference_time:.4f} ms")
    print(f"Target: < 10 ms")
    print(f"Status: {'✓ PASS' if avg_inference_time < 10 else '✗ FAIL'}")

    # 绘制混淆矩阵
    plot_path = os.path.join(config.checkpoint_dir, 'confusion_matrix_v2.png')
    plot_confusion_matrix(cm, dataloaders['class_names'], plot_path)
    print(f"\nConfusion matrix saved to: {plot_path}")

    # 汇总
    print("\n" + "=" * 60)
    print("Evaluation Summary - DSC-CBAM-LSTM V2")
    print("=" * 60)
    print(f"Parameters: {num_params:,} (< 100K: {'✓' if num_params < 100000 else '✗'})")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Inference Time: {avg_inference_time:.4f} ms (< 10ms: {'✓' if avg_inference_time < 10 else '✗'})")

    # 保存结果
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'inference_time_ms': avg_inference_time,
        'num_params': num_params
    }

    return results


def main():
    evaluate()


if __name__ == "__main__":
    main()
