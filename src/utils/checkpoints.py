import os
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_checkpoint(model, epoch, optimizer, train_loss, val_loss, train_acc, val_acc, best_val_acc=0, improvement_threshold=0.01, dir="checkpoints"):
    """
    Save model checkpoint if there's a significant improvement in validation accuracy.
    
    Args:
        model: The model to save
        epoch: Current epoch number
        optimizer: The optimizer used for training
        train_loss: Current training loss
        val_loss: Current validation loss
        train_acc: Current training accuracy
        val_acc: Current validation accuracy
        best_val_acc: Best validation accuracy so far
        improvement_threshold: Threshold for considering an improvement significant (default: 0.01 or 1%)
    
    Returns:
        best_val_acc: Updated best validation accuracy
    """
    checkpoints_dir = dir
    # Ensure checkpoint directory exists
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Check if there's significant improvement
    is_best = val_acc > best_val_acc
    significant_improvement = val_acc > (best_val_acc + float(improvement_threshold))
    
    # Update best validation accuracy
    if is_best:
        best_val_acc = val_acc
    
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'best_val_acc': best_val_acc
    }
    
    # Save the checkpoint
    checkpoint_filename = os.path.join(checkpoints_dir, f"{model.name}_checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_filename)
    logger.info(f"Checkpoint saved at {checkpoint_filename}")
    
    # If there's a significant improvement, save as best model and delete previous checkpoints
    if significant_improvement:
        best_model_path = os.path.join(checkpoints_dir, f"{model.name}_best.pth")
        torch.save(checkpoint, best_model_path)
        logger.info(f"New best model saved with validation accuracy: {val_acc:.4f} (previous best: {best_val_acc - (val_acc - best_val_acc):.4f})")
        
        # Delete previous checkpoints to save space
        for filename in os.listdir(checkpoints_dir):
            file_path = os.path.join(checkpoints_dir, filename)
            # Skip the best model and current checkpoint
            if file_path != best_model_path and file_path != checkpoint_filename:
                if filename.startswith(f"{model.name}_checkpoint_") and filename.endswith(".pth"):
                    os.remove(file_path)
                    logger.info(f"Deleted previous checkpoint: {filename}")
    
    return best_val_acc