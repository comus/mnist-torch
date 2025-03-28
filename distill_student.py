import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random

from models import TeacherCNN, StudentCNN, count_parameters, accuracy
from utils import load_mnist, distillation_loss, measure_inference_time, plot_comparison

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Add new visualization functions
def visualize_student_predictions(student, teacher, test_loader, device, num_samples=10):
    """Compare student and teacher model predictions"""
    # Get a batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Randomly select samples
    indices = torch.randperm(len(images))[:num_samples]
    X_samples = images[indices].to(device)
    y_samples = labels[indices]
    
    # Get predictions from both models
    student.eval()
    teacher.eval()
    with torch.no_grad():
        student_logits = student(X_samples)
        teacher_logits = teacher(X_samples)
        
        student_preds = torch.argmax(student_logits, dim=1)
        teacher_preds = torch.argmax(teacher_logits, dim=1)
        
        student_probs = torch.softmax(student_logits, dim=1)
        teacher_probs = torch.softmax(teacher_logits, dim=1)
    
    # Convert to CPU for plotting
    X_samples = X_samples.cpu()
    student_preds = student_preds.cpu()
    teacher_preds = teacher_preds.cpu()
    student_probs = student_probs.cpu()
    teacher_probs = teacher_probs.cpu()
    
    # Plot results
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
    
    for i in range(num_samples):
        # Display image
        img = X_samples[i, 0]  # Remove channel dimension
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"True: {y_samples[i]}")
        axes[i, 0].axis('off')
        
        # Display teacher's prediction probabilities
        axes[i, 1].bar(range(10), teacher_probs[i])
        axes[i, 1].set_title(f"Teacher: {teacher_preds[i]}")
        axes[i, 1].set_xticks(range(10))
        
        # Display student's prediction probabilities
        axes[i, 2].bar(range(10), student_probs[i])
        axes[i, 2].set_title(f"Student: {student_preds[i]}")
        axes[i, 2].set_xticks(range(10))
    
    plt.tight_layout()
    
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/student_teacher_comparison.png")
    return fig

def plot_learning_curves(losses, accuracies, teacher_acc):
    """Plot learning curves"""
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'o-', label='Distillation Loss')
    plt.title('Distillation Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, 'o-', label='Student Accuracy', color='orange')
    plt.axhline(y=teacher_acc, color='blue', linestyle='--', label=f'Teacher Accuracy: {teacher_acc:.4f}')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/student_learning_curves.png")
    return plt.gcf()

def plot_confusion_matrix(model, test_loader, device, title, filename):
    """Plot confusion matrix"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig(f"results/{filename}")
    return plt.gcf()

def compare_hard_cases(student, teacher, test_loader, device):
    """Compare models on difficult samples"""
    # Collect all predictions
    teacher.eval()
    student.eval()
    all_images = []
    all_labels = []
    all_teacher_preds = []
    all_student_preds = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)
            
            _, teacher_preds = torch.max(teacher_outputs, 1)
            _, student_preds = torch.max(student_outputs, 1)
            
            all_images.append(inputs.cpu())
            all_labels.append(targets.cpu())
            all_teacher_preds.append(teacher_preds.cpu())
            all_student_preds.append(student_preds.cpu())
    
    # Concatenate all batches
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    all_teacher_preds = torch.cat(all_teacher_preds)
    all_student_preds = torch.cat(all_student_preds)
    
    # Find cases where teacher is correct but student is wrong
    mask = (all_teacher_preds == all_labels) & (all_student_preds != all_labels)
    hard_cases_indices = torch.nonzero(mask).squeeze()
    
    if len(hard_cases_indices) == 0:
        print("No cases found where teacher is correct but student is wrong!")
        return None
    
    # Select at most 10 cases
    num_cases = min(10, len(hard_cases_indices))
    selected_indices = hard_cases_indices[:num_cases]
    
    # Plot results
    fig, axes = plt.subplots(num_cases, 1, figsize=(8, 3 * num_cases))
    if num_cases == 1:
        axes = [axes]
        
    for i in range(num_cases):
        idx = selected_indices[i]
        img = all_images[idx, 0]  # Remove channel dimension
        
        axes[i].imshow(img, cmap='gray')
        y_true = all_labels[idx].item()
        t_pred = all_teacher_preds[idx].item()
        s_pred = all_student_preds[idx].item()
        axes[i].set_title(f"True: {y_true}, Teacher: {t_pred}, Student: {s_pred}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/hard_cases.png")
    return fig

def main():
    # Set random seed
    set_seed()
    
    # Ensure results directory exists
    if not os.path.exists("results"):
        os.makedirs("results")
        
    # Ensure models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")
    
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
    
    # Check if teacher model exists
    if not os.path.exists("models/teacher_model.pth"):
        print("Error: Teacher model not found. Please run train_teacher.py first")
        return
    
    # Load data
    train_loader, test_loader = load_mnist()
    
    # Load teacher model
    teacher = TeacherCNN().to(device)
    teacher.load_state_dict(torch.load("models/teacher_model.pth", map_location=device))
    teacher.eval()  # Set to evaluation mode
    
    # Evaluate teacher model
    teacher_acc = accuracy(teacher, test_loader, device)
    print(f"Loaded teacher model, accuracy: {teacher_acc:.4f}")
    
    # Initialize student model
    print("\n=== Starting Knowledge Distillation ===")
    student = StudentCNN().to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    
    # Distillation parameters
    num_epochs = 5  # Student model may need more epochs
    temp = 3.0      # Temperature parameter
    alpha = 0.5     # Soft and hard targets weight
    
    # Track training metrics
    train_losses = []
    student_accuracies = []
    
    # Start distillation training
    distill_start_time = time.time()
    for epoch in range(num_epochs):
        student.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            student_logits = student(inputs)
            
            # Calculate distillation loss
            loss = distillation_loss(student_logits, teacher_logits, targets, temp=temp, alpha=alpha)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / num_batches
        student_acc = accuracy(student, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Student Accuracy: {student_acc:.4f}, Teacher Accuracy: {teacher_acc:.4f}")
        
        # Track training metrics
        train_losses.append(avg_loss)
        student_accuracies.append(student_acc)
    
    distill_time = time.time() - distill_start_time
    
    # Final evaluation
    student_acc = accuracy(student, test_loader, device)
    
    # Save student model
    torch.save(student.state_dict(), "models/student_model.pth")
    
    # Output results
    print("\n=== Knowledge Distillation Results ===")
    print(f"Distillation time: {distill_time:.2f} seconds")
    print(f"Student model accuracy: {student_acc:.4f}")
    print(f"Teacher model accuracy: {teacher_acc:.4f}")
    print("Student model saved to models/student_model.pth")
    
    # Compare model sizes
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    print("\n=== Model Comparison ===")
    print(f"Teacher model parameter count: {teacher_params:,}")
    print(f"Student model parameter count: {student_params:,}")
    print(f"Size reduction ratio: {teacher_params / student_params:.2f}x")
    
    # Compare inference speed
    print("\n=== Inference Speed Comparison ===")
    teacher_infer_time = measure_inference_time(teacher, test_loader, device)
    student_infer_time = measure_inference_time(student, test_loader, device)
    
    print(f"Teacher model inference time: {teacher_infer_time:.6f} seconds")
    print(f"Student model inference time: {student_infer_time:.6f} seconds")
    print(f"Speed improvement: {teacher_infer_time / student_infer_time:.2f}x")
    
    # Visualize comparison results
    plot_comparison(teacher_acc, student_acc, teacher_params, student_params, 
                    teacher_infer_time, student_infer_time)
    
    # === Additional visualizations ===
    # 1. Learning curves
    print("Generating learning curves...")
    learning_curves = plot_learning_curves(train_losses, student_accuracies, teacher_acc)
    
    # 2. Student vs teacher prediction comparison
    print("Generating student vs teacher prediction comparison...")
    pred_comparison = visualize_student_predictions(student, teacher, test_loader, device)
    
    # 3. Student model confusion matrix
    print("Generating student model confusion matrix...")
    student_cm = plot_confusion_matrix(student, test_loader, device, "Student Model Confusion Matrix", "student_confusion_matrix.png")
    
    # 4. Hard cases analysis
    print("Analyzing hard cases...")
    hard_cases_fig = compare_hard_cases(student, teacher, test_loader, device)
    
    print("\n=== Visualization Results ===")
    print("Model comparison chart saved to results/distillation_results.png")
    print("Learning curves saved to results/student_learning_curves.png")
    print("Student vs teacher prediction comparison saved to results/student_teacher_comparison.png")
    print("Student model confusion matrix saved to results/student_confusion_matrix.png")
    if hard_cases_fig:
        print("Hard cases analysis saved to results/hard_cases.png")

if __name__ == "__main__":
    main() 