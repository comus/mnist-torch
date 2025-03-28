import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 數據加載函數
def load_mnist(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                              train=True, 
                                              download=True, 
                                              transform=transform)
    
    test_dataset = torchvision.datasets.MNIST(root='./data', 
                                             train=False, 
                                             download=True, 
                                             transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False)
    
    print(f"Data loading complete! Training set: {len(train_dataset)} samples, Test set: {len(test_dataset)} samples")
    return train_loader, test_loader

# 知識蒸餾損失函數
def distillation_loss(student_logits, teacher_logits, targets, temp=3.0, alpha=0.5):
    # 軟目標損失 (蒸餾損失)
    soft_targets = F.softmax(teacher_logits / temp, dim=1)
    soft_prob = F.log_softmax(student_logits / temp, dim=1)
    soft_targets_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temp * temp)
    
    # 硬目標損失 (交叉熵)
    hard_loss = F.cross_entropy(student_logits, targets)
    
    # 總損失 = 軟目標損失 * 權重 + 硬目標損失 * (1-權重)
    return alpha * soft_targets_loss + (1.0 - alpha) * hard_loss

# 結果可視化
def plot_comparison(teacher_acc, student_acc, teacher_params, student_params, teacher_time, student_time):
    """Results visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    models = ['Teacher', 'Student']
    accuracies = [teacher_acc, student_acc]
    ax1.bar(models, accuracies, color=['blue', 'orange'])
    ax1.set_ylim([0, 1])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    
    # Add accuracy values
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # Params count and inference time comparison
    ax2.bar(models, [teacher_time, student_time], color=['blue', 'orange'], alpha=0.7, label='Inference Time (s)')
    
    # Add second y-axis
    ax3 = ax2.twinx()
    ax3.bar(models, [teacher_params/1e6, student_params/1e6], color=['blue', 'orange'], alpha=0.3, label='Parameters (M)')
    
    ax2.set_title('Model Efficiency')
    ax2.set_ylabel('Inference Time (s)')
    ax3.set_ylabel('Parameters (Millions)')
    
    # Add values
    for i, v in enumerate([teacher_time, student_time]):
        ax2.text(i, v + 0.0001, f'{v:.5f}s', ha='center')
    
    for i, v in enumerate([teacher_params/1e6, student_params/1e6]):
        ax3.text(i, v + 0.01, f'{v:.2f}M', ha='center')
    
    plt.tight_layout()
    
    # Save figure
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/distillation_results.png")
    
    return fig

# 測量推理時間
def measure_inference_time(model, test_loader, device, samples=100, repeats=10):
    model.eval()
    # 獲取一批次數據
    dataiter = iter(test_loader)
    inputs, _ = next(dataiter)
    inputs = inputs[:samples].to(device)
    
    # 預熱 GPU/MPS
    with torch.no_grad():
        for _ in range(5):
            model(inputs)
    
    # 測量時間
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
        
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(repeats):
            model(inputs)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
        
    end_time = time.time()
    
    return (end_time - start_time) / repeats 