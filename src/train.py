import torch
import torch.nn as nn
import numpy as np
from captum.attr import LayerGradCam
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter

from models.resnet import resnet101, resnet152
from models.model import ResNet
from utils.data_loader import create_data_loaders
from utils.logs import init_log, log_train, log_val, end_log
from utils.checkpoints import save_checkpoint
from utils.visualization import save_feature_maps, FeatureMapVisualizer, plot_loss, plot_accuracy, plot_lr, plot_confusion, plot_training_curves

import argparse
import os
import time
import tqdm
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(project_dir, "results")
checkpoints_dir = None
logs_dir = None
figures_dir = None
gradcam_dir = None
feature_maps_dir = None
architectures_dir = None
start_time = time.strftime("%Y-%m-%d_%H-%M-%S")

writer = None

train_loader, val_loader, test_loader, class_balancer = create_data_loaders()
model: ResNet = None

def save_cam(model, img, label, epoch, folder):
    """Save GradCAM visualization for the last layer.
    
    This is a simplified version that just uses the last layer of layer4.
    For comprehensive feature maps from all layers, use save_feature_maps instead.
    """
    # Generate the GradCAM heatmap
    cam = LayerGradCam(model, model.layer4[-1].conv3)
    attr = cam.attribute(img.unsqueeze(0), target=label)
    heat = attr.squeeze(0).mean(0)
    heat = TF.resize(heat.unsqueeze(0), [224, 224]).squeeze()
    
    # Create a figure with two subplots: original image and heatmap
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot original image
    # Convert from tensor to numpy for plotting
    img_np = img.detach().cpu().numpy().transpose((1, 2, 0))
    # Normalize to 0-1 range if needed
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot heatmap
    heat_np = heat.detach().cpu().numpy()
    heatmap = axes[1].imshow(heat_np, cmap='jet')
    axes[1].set_title('Feature Map (GradCAM)')
    axes[1].axis('off')
    
    # Add colorbar for the heatmap
    plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Adjust layout and save
    plt.tight_layout()
    fname = folder / f"cam_epoch{epoch}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    
    # Log to tensorboard both images
    writer.add_image("Original_Image", img.unsqueeze(0), epoch, dataformats='NCHW')
    writer.add_image("GradCAM", heat.unsqueeze(0), epoch, dataformats='CHW')
    
    # Also create a figure with the heatmap overlaid on the image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np)
    ax.imshow(heat_np, cmap='jet', alpha=0.5)  # Overlay with 50% transparency
    ax.set_title('GradCAM Overlay')
    ax.axis('off')
    overlay_fname = folder / f"cam_overlay_epoch{epoch}.png"
    plt.savefig(overlay_fname, bbox_inches='tight')
    plt.close()
    
    # Log the overlay to tensorboard
    # We need to convert the overlay to a tensor format for tensorboard
    # This is a bit complex because we need to combine the RGB channels with the heatmap
    overlay_tensor = torch.from_numpy(
        img_np * (1 - 0.5) + 0.5 * heat_np.reshape(*heat_np.shape, 1) * np.array([1, 0, 0])
    ).permute(2, 0, 1).float()
    writer.add_image("GradCAM_Overlay", overlay_tensor.unsqueeze(0), epoch, dataformats='NCHW')

def train_model(model, train_loader, val_loader, device, visualize=False, save_model=False):
    model.to(device)

    batch_size = 96
    
    criterion = nn.CrossEntropyLoss(weight=class_balancer.get_class_weights().to(device), label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1*(batch_size/256), momentum=0.9, weight_decay=2e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0.1*(batch_size/256)*1e-6)
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup_epochs)

    best_val_acc = 0.0  # Track best validation accuracy

    for epoch in range(150):
        model.train()
        current_loss = 0
        current_acc = 0
        for images, labels in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            current_acc += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss = current_loss / len(train_loader)
        train_acc = current_acc / len(train_loader.dataset)
        lr = optimizer.param_groups[0]['lr']

        model.eval()
        current_val_loss = 0
        current_val_acc = 0
        for images, labels in tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            current_val_loss += loss.item()
            current_val_acc += (outputs.argmax(dim=1) == labels).sum().item()

        val_loss = current_val_loss / len(val_loader)
        val_acc = current_val_acc / len(val_loader.dataset)

        if epoch > warmup_epochs:
            lr_scheduler.step()
        else:
            lr_scheduler.step()
            warmup_scheduler.step()

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, LR: {lr:.6f}")
        log_train(train_loss, train_acc, lr)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("train/lr", lr, epoch)

        print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        log_val(val_loss, val_acc)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        
        # Save checkpoint and update best validation accuracy
        if save_model:
            best_val_acc = save_checkpoint(model, epoch, optimizer, train_loss, val_loss, train_acc, val_acc, best_val_acc, 0.01, checkpoints_dir)
        
        # Save visualizations if visualize is enabled
        if visualize and epoch % 10 == 0:  # Save every 10 epochs to avoid too many images
            # Get a sample image from validation set
            with torch.no_grad():
                # Ensure directories exist
                os.makedirs(gradcam_dir, exist_ok=True)
                os.makedirs(feature_maps_dir, exist_ok=True)
                os.makedirs(os.path.join(gradcam_dir, "training"), exist_ok=True)

                for i in range(8):
                    sample_images, sample_labels = next(iter(val_loader))
                    sample_img = sample_images[i].to(device)
                    sample_label = sample_labels[i].item()
                
                    if i == 0:
                        # Save standard GradCAM visualization for the last layer
                        save_cam(model, sample_img, sample_label, epoch, Path(os.path.join(gradcam_dir, "training")))
                
                    # Save comprehensive feature maps for all layers
                    image_dir = os.path.join(feature_maps_dir, "training", f"picture{i}")
                    os.makedirs(image_dir, exist_ok=True)
                    save_feature_maps(model, sample_img, sample_label, epoch, image_dir, writer)

        # Final checkpoint at the end of training if save_model is enabled
        if save_model and epoch == 150:  # Last epoch
            final_path = os.path.join(checkpoints_dir, f"{model.name}_final.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }, final_path)
            logger.info(f"Final model saved at {final_path}")

def test_model(model, test_loader, device, classes, visualize=False, save_model=False):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Store for visualization
            all_preds.append(predicted.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            
            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Convert lists to tensors
    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    # Calculate overall accuracy
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    writer.add_scalar("test/accuracy", accuracy)
    
    if visualize:
        # Convert tensors to numpy arrays for visualization
        y_pred = all_preds.numpy()
        y_true = all_labels.numpy()
        y_score = all_probs.numpy()
        
        # Call visualization functions
        from utils.visualization import (
            plot_confusion, plot_roc_curves, plot_precision_recall_curves,
            plot_top_k_accuracy, plot_per_class_metrics, plot_confidence_histogram,
            plot_sorted_confusion_matrix
        )
        
        plot_confusion(y_true, y_pred, classes, figures_dir)
        plot_top_k_accuracy(y_score, y_true, figures_dir)
        plot_per_class_metrics(y_true, y_pred, classes, figures_dir)
        plot_roc_curves(y_true, y_score, classes, "test", figures_dir)
        plot_precision_recall_curves(y_true, y_score, classes, figures_dir)
        plot_confidence_histogram(y_score, y_true, y_pred, figures_dir)
        plot_sorted_confusion_matrix(y_true, y_pred, classes, figures_dir)
    
    # Final checkpoint at the end of training if save_model is enabled
    if save_model:
        final_path = os.path.join(checkpoints_dir, f"{model.name}_tested.pth")
        torch.save(model.state_dict(), final_path)
        logger.info(f"Final model saved at {final_path}")
        
    return accuracy, all_preds.numpy(), all_probs.numpy(), all_labels.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet training script")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Mode to run")
    parser.add_argument("--model", choices=["resnet101", "resnet152"], default="resnet101", help="Model to train")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device to use")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--save-model", action="store_true", help="Save model")

    args = parser.parse_args()

    init_log(args.device)

    if args.mode == "train":
        if args.model == "resnet101":
            model = resnet101()
        elif args.model == "resnet152":
            model = resnet152()

        model_dir = os.path.join(results_dir, model.name)
        checkpoints_dir = os.path.join(model_dir, "checkpoints")
        logs_dir = os.path.join(model_dir, "logs")
        figures_dir = os.path.join(model_dir, "figures")
        gradcam_dir = os.path.join(figures_dir, "gradcam")
        feature_maps_dir = os.path.join(figures_dir, "feature_maps")
        architectures_dir = os.path.join(figures_dir, "architectures")
        
        # Create all required directories
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(gradcam_dir, exist_ok=True)
        os.makedirs(feature_maps_dir, exist_ok=True)
        os.makedirs(architectures_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=str(logs_dir))
        
	# Now start the training process
        train_model(model, train_loader, val_loader, args.device, args.visualize, args.save_model)
    elif args.mode == "test":
        if args.model == "resnet101":
            model = resnet101(pretrained=True)
        elif args.model == "resnet152":
            model = resnet152(pretrained=True)

        model_dir = os.path.join(results_dir, model.name)
        checkpoints_dir = os.path.join(model_dir, "checkpoints")
        logs_dir = os.path.join(model_dir, "logs")
        figures_dir = os.path.join(model_dir, "figures")
        gradcam_dir = os.path.join(figures_dir, "gradcam")
        feature_maps_dir = os.path.join(figures_dir, "feature_maps")
        architectures_dir = os.path.join(figures_dir, "architectures")
        
        # Create all required directories
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(gradcam_dir, exist_ok=True)
        os.makedirs(feature_maps_dir, exist_ok=True)
        os.makedirs(architectures_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=str(logs_dir))
        
        # Get class names from the dataset
        # We need to access the original dataset through the test_dataset's dataset attribute
        classes = test_loader.dataset.dataset.classes
        print(f"Found {len(classes)} classes in the dataset")

        # Now start the testing process
        test_model(model, test_loader, args.device, classes, args.visualize, args.save_model)
        
    end_log(logs_dir)
