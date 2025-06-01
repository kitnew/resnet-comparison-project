import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LogNorm
from pathlib import Path
import torchvision.transforms.functional as TF
from captum.attr import LayerGradCam
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import importlib

from .fm_visualizer import FeatureMapVisualizer

# Directory for saving visualizations
visualization_dir = None
# Tensorboard writer
writer = None

def plot_loss(log, folder):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(log["train_loss"], label="Train Loss")
    ax.plot(log["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
    fig.savefig(folder / f"loss.png"); plt.close(fig)
    writer.add_figure("Loss", fig)

def plot_accuracy(log, folder):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(log["train_accuracy"], label="Train Accuracy")
    ax.plot(log["val_accuracy"], label="Validation Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()
    fig.savefig(folder / f"accuracy.png"); plt.close(fig)
    writer.add_figure("Accuracy", fig)

def plot_lr(log, folder):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(log["lr"], label="Learning Rate")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate"); ax.legend()
    fig.savefig(folder / f"lr.png"); plt.close(fig)
    writer.add_figure("Learning Rate", fig)

def plot_confusion(y_true, y_pred, classes, folder):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, ax=ax, cmap="Blues", norm=LogNorm())
    ax.set_xlabel("Pred"); ax.set_ylabel("True")
    fig.savefig(folder / f"confmat.png"); plt.close(fig)
    writer.add_figure("ConfMat", fig)

def plot_training_curves(log, folder):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot loss curves
    ax1.plot(log["train_loss"], 'b-', label='Training Loss')
    ax1.plot(log["val_loss"], 'r-', label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Plot accuracy curves
    ax2.plot(log["train_accuracy"], 'b-', label='Training Accuracy')
    ax2.plot(log["val_accuracy"], 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    
    plt.tight_layout()
    fig.savefig(folder / f"training_curves.png")
    plt.close(fig)
    writer.add_figure("Training_Curves", fig)


def plot_roc_curves(y_true, y_score, classes, epoch, folder):
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)
    
    # One-hot encode y_true
    y_true_bin = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        y_true_bin[i, y_true[i]] = 1
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    for i, color in zip(range(min(n_classes, 10)), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc="lower right")
    
    fig.savefig(folder / f"roc_curves_ep{epoch}.png")
    plt.close(fig)
    writer.add_figure("ROC_Curves", fig, epoch)


def save_feature_maps(model, img, label, epoch, visualization_dir, writer):
    """Extract and visualize feature maps from all layers of the model.
    
    Args:
        model: The ResNet model
        img: Input image tensor
        label: Target class label
        epoch: Current epoch number
        visualization_dir: Directory to save visualizations
        writer: TensorBoard SummaryWriter instance
    """
    visualizer = FeatureMapVisualizer(model, writer, visualization_dir)
    feature_maps = visualizer.visualize_all_layers(img, label, epoch)
    visualizer.cleanup()  # Clean up hooks
    return feature_maps