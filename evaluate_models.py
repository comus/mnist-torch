import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import os

from models import TeacherCNN, StudentCNN, accuracy
from utils import load_mnist, measure_inference_time

def compare_models_on_examples(teacher, student, test_loader, device, num_examples=5):
    # Get a batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Randomly select samples
    indices = torch.randperm(len(images))[:num_examples]
    X_examples = images[indices].to(device)
    y_examples = labels[indices]
    
    # Get model predictions
    teacher.eval()
    student.eval()
    with torch.no_grad():
        teacher_logits = teacher(X_examples)
        student_logits = student(X_examples)
        
        teacher_probs = F.softmax(teacher_logits, dim=1)
        student_probs = F.softmax(student_logits, dim=1)
        
        teacher_preds = torch.argmax(teacher_probs, dim=1)
        student_preds = torch.argmax(student_probs, dim=1)
    
    # Convert to CPU for plotting
    X_examples = X_examples.cpu()
    teacher_probs = teacher_probs.cpu()
    student_probs = student_probs.cpu()
    teacher_preds = teacher_preds.cpu()
    student_preds = student_preds.cpu()
    
    # Plot results
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 3*num_examples))
    
    for i in range(num_examples):
        # Plot image
        img = X_examples[i, 0]  # First channel
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"True Label: {y_examples[i]}")
        axes[i, 0].axis('off')
        
        # Plot teacher model prediction probabilities
        axes[i, 1].bar(range(10), teacher_probs[i])
        axes[i, 1].set_title(f"Teacher Prediction: {teacher_preds[i]}")
        axes[i, 1].set_xticks(range(10))
        
        # Plot student model prediction probabilities
        axes[i, 2].bar(range(10), student_probs[i])
        axes[i, 2].set_title(f"Student Prediction: {student_preds[i]}")
        axes[i, 2].set_xticks(range(10))
    
    plt.tight_layout()
    
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig('results/prediction_examples.png')
    plt.close(fig)
    
    return fig

def main():
    # Ensure results directory exists
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Check for MPS (Apple Silicon) or CUDA
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device: Apple Silicon")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Load data
    _, test_loader = load_mnist()
    
    # Load models
    teacher = TeacherCNN().to(device)
    student = StudentCNN().to(device)
    
    try:
        teacher.load_state_dict(torch.load("models/teacher_model.pth", map_location=device))
        student.load_state_dict(torch.load("models/student_model.pth", map_location=device))
    except:
        print("Error: Model files not found. Please run train_teacher.py and distill_student.py first")
        return
    
    # Set to evaluation mode
    teacher.eval()
    student.eval()
    
    # Evaluate models
    teacher_acc = accuracy(teacher, test_loader, device)
    student_acc = accuracy(student, test_loader, device)
    
    print("=== Model Evaluation ===")
    print(f"Teacher model accuracy: {teacher_acc:.4f}")
    print(f"Student model accuracy: {student_acc:.4f}")
    print(f"Accuracy difference: {teacher_acc - student_acc:.4f}")
    
    # Measure inference time with different batch sizes
    batch_sizes = [1, 4, 16, 64, 256]
    teacher_times = []
    student_times = []
    
    print("\n=== Inference Time with Different Batch Sizes ===")
    print("Batch Size | Teacher | Student | Speedup")
    print("-----------|---------|---------|--------")
    
    for batch_size in batch_sizes:
        # Get a batch of data
        dataiter = iter(test_loader)
        inputs, _ = next(dataiter)
        inputs = inputs[:batch_size].to(device)
        
        # Measure teacher model time
        teacher.eval()
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                teacher(inputs)
        
        # Measure time
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
            
        t_start = time.time()
        with torch.no_grad():
            for _ in range(10):  # Run multiple times for more reliable results
                teacher(inputs)
                
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
            
        t_time = (time.time() - t_start) / 10
        teacher_times.append(t_time)
        
        # Measure student model time
        student.eval()
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                student(inputs)
        
        # Measure time
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
            
        s_start = time.time()
        with torch.no_grad():
            for _ in range(10):
                student(inputs)
                
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
            
        s_time = (time.time() - s_start) / 10
        student_times.append(s_time)
        
        speedup = t_time / s_time
        print(f"{batch_size:9d} | {t_time:.6f}s | {s_time:.6f}s | {speedup:.2f}x")
    
    # Plot batch size vs inference time
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, teacher_times, 'o-', label='Teacher')
    plt.plot(batch_sizes, student_times, 'o-', label='Student')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/inference_time.png')
    plt.close()
    
    # Compare models on specific examples
    compare_models_on_examples(teacher, student, test_loader, device)
    print("Prediction examples saved to results/prediction_examples.png")
    print("Inference time comparison saved to results/inference_time.png")

if __name__ == "__main__":
    main() 