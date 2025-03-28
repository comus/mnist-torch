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

from models import TeacherCNN, count_parameters, accuracy
from utils import load_mnist

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

# Ensure model save directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# Visualize predictions
def visualize_predictions(model, test_loader, device, num_samples=10):
    # Get a batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Randomly select samples
    indices = torch.randperm(len(images))[:num_samples]
    X_samples = images[indices].to(device)
    y_samples = labels[indices]
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        logits = model(X_samples)
        preds = torch.argmax(logits, dim=1)
    
    # Convert to CPU for plotting
    X_samples = X_samples.cpu()
    preds = preds.cpu()
    
    # Plot results
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Display image
        img = X_samples[i, 0]  # Remove channel dimension
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Pred: {preds[i]}, True: {y_samples[i]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/prediction_samples.png")
    return fig

# Plot confusion matrix
def plot_confusion_matrix(model, test_loader, device):
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
    plt.title('Confusion Matrix')
    
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/confusion_matrix.png")
    return plt.gcf()

def main():
    # Set random seed
    set_seed()
    
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
    train_loader, test_loader = load_mnist()
    
    # Initialize teacher model
    print("\n=== Starting Teacher Model Training ===")
    teacher = TeacherCNN().to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters
    num_epochs = 3  # Using fewer epochs for quick demonstration
    
    # Data for plotting training progress
    train_losses = []
    val_accuracies = []
    epochs = []
    
    # Start training
    train_start_time = time.time()
    for epoch in range(num_epochs):
        teacher.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / num_batches
        acc = accuracy(teacher, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        
        # Save data for plotting
        train_losses.append(avg_loss)
        val_accuracies.append(acc)
        epochs.append(epoch+1)
    
    train_time = time.time() - train_start_time
    
    # Final evaluation
    final_acc = accuracy(teacher, test_loader, device)
    param_count = count_parameters(teacher)
    
    # Save model
    torch.save(teacher.state_dict(), "models/teacher_model.pth")
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Training Loss')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'o-', label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/training_progress.png")
    
    # Visualize predictions
    pred_fig = visualize_predictions(teacher, test_loader, device)
    
    # Plot confusion matrix
    cm_fig = plot_confusion_matrix(teacher, test_loader, device)
    
    # Output results
    print("\n=== Teacher Model Training Results ===")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Parameter count: {param_count:,}")
    print("Model saved to models/teacher_model.pth")
    print("Training progress chart saved to results/training_progress.png")
    print("Prediction samples chart saved to results/prediction_samples.png")
    print("Confusion matrix saved to results/confusion_matrix.png")

if __name__ == "__main__":
    main() 