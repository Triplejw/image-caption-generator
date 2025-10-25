import torch
import matplotlib.pyplot as plt
import os

def plot_training_curves(save_path='training_curves.png'):
    """Plot training and validation loss curves"""
    
    # Load checkpoints and extract losses
    train_losses = []
    val_losses = []
    epochs = []
    
    for epoch in range(1, 16):
        checkpoint_path = f'models_v2/checkpoint_epoch_{epoch}.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            train_losses.append(checkpoint['train_loss'])
            val_losses.append(checkpoint['val_loss'])
            epochs.append(epoch)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Combined loss curves
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-o', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss difference (overfitting indicator)
    loss_diff = [t - v for t, v in zip(train_losses, val_losses)]
    ax2.plot(epochs, loss_diff, 'g-o', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Train Loss - Val Loss', fontsize=12)
    ax2.set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(epochs, 0, loss_diff, where=[d < 0 for d in loss_diff], 
                      color='green', alpha=0.2, label='Good generalization')
    ax2.fill_between(epochs, 0, loss_diff, where=[d >= 0 for d in loss_diff], 
                      color='red', alpha=0.2, label='Overfitting')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    
    return save_path

if __name__ == "__main__":
    plot_training_curves()
