import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LogNorm

#from .fm_visualizer import FeatureMapVisualizer

# Directory for saving visualizations
visualization_dir = None
# Tensorboard writer
writer = None

def plot_loss(log, folder):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(log["train_loss"], label="Train Loss")
    ax.plot(log["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
    fig.savefig(folder + "/loss.png"); plt.close(fig)

def plot_accuracy(log, folder):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(log["train_accuracy"], label="Train Accuracy")
    ax.plot(log["val_accuracy"], label="Validation Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()
    fig.savefig(folder + "/accuracy.png"); plt.close(fig)

def plot_lr(log, folder):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(log["lr"], label="Learning Rate")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate"); ax.legend()
    fig.savefig(folder + "/lr.png"); plt.close(fig)

def plot_confusion(y_true, y_pred, classes, folder):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, ax=ax, cmap="Blues", norm=LogNorm())
    ax.set_xlabel("Pred"); ax.set_ylabel("True")
    fig.savefig(folder + "/confmat.png"); plt.close(fig)

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
    fig.savefig(folder + "/training_curves.png")
    plt.close(fig)


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
    
    # Ensure epoch is a string for the filename
    epoch_str = str(epoch)
    fig.savefig(folder + "/roc_curves_ep{epoch_str}.png")
    plt.close(fig)


def plot_precision_recall_curves(y_true, y_score, classes, folder):
    """Plot precision-recall curves for each class.
    
    Args:
        y_true: Array of true class labels
        y_score: Array of predicted probabilities for each class
        classes: List of class names
        folder: Directory to save the plot
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from itertools import cycle
    
    # Compute Precision-Recall curve and average precision for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()
    n_classes = len(classes)
    
    # One-hot encode y_true
    y_true_bin = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        y_true_bin[i, y_true[i]] = 1
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])
    
    # Plot Precision-Recall curves
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    for i, color in zip(range(min(n_classes, 10)), colors):
        ax.plot(recall[i], precision[i], color=color, lw=2,
                label=f'PR curve of class {classes[i]} (AP = {avg_precision[i]:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc="lower left")
    ax.grid(True)
    
    fig.savefig(folder + "/precision_recall_curves.png")
    plt.close(fig)


def plot_top_k_accuracy(y_score, y_true, folder, k_values=None):
    """Plot Top-K accuracy curve.
    
    Args:
        y_score: Array of predicted probabilities for each class
        y_true: Array of true class labels
        folder: Directory to save the plot
        k_values: List of k values to compute accuracy for. If None, uses [1,2,3,4,5]
    """
    if k_values is None:
        k_values = [1, 2, 3, 4, 5]
    
    # Ensure k values don't exceed number of classes
    n_classes = y_score.shape[1]
    k_values = [k for k in k_values if k <= n_classes]
    
    # Calculate top-k accuracy for each k
    topk_accuracies = []
    for k in k_values:
        # Get top k indices for each sample
        topk_indices = np.argsort(-y_score, axis=1)[:, :k]
        
        # Check if true class is in top k predictions
        correct = np.array([y_true[i] in topk_indices[i] for i in range(len(y_true))])
        accuracy = np.mean(correct)
        topk_accuracies.append(accuracy)
    
    # Plot Top-K accuracy curve
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(k_values, topk_accuracies, 'bo-', linewidth=2)
    ax.set_xlabel('k')
    ax.set_ylabel('Top-k Accuracy')
    ax.set_title('Top-K Accuracy Curve')
    ax.set_xticks(k_values)
    ax.set_xticklabels([f'Top-{k}' for k in k_values])
    ax.set_ylim([0, 1.05])
    ax.grid(True)
    
    # Add value labels
    for k, acc in zip(k_values, topk_accuracies):
        ax.annotate(f'{acc:.3f}', (k, acc), xytext=(0, 5),
                    textcoords='offset points', ha='center')
    
    fig.savefig(folder + "/top_k_accuracy.png")
    plt.close(fig)
    
    return topk_accuracies


def plot_per_class_metrics(y_true, y_pred, classes, folder):
    """Plot per-class metrics as a bar chart.
    
    Args:
        y_true: Array of true class labels
        y_pred: Array of predicted class labels
        classes: List of class names
        folder: Directory to save the plot
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Calculate metrics for each class
    precision = precision_score(y_true, y_pred, average=None, labels=range(len(classes)), zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=range(len(classes)), zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=range(len(classes)), zero_division=0)
    
    # Count samples per class
    class_counts = np.bincount(y_true, minlength=len(classes))
    
    # Create a DataFrame for easier plotting
    import pandas as pd
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Count': class_counts
    }, index=classes)
    
    # Sort by count for better readability
    metrics_df = metrics_df.sort_values('Count', ascending=False)
    
    # Plot bar chart
    fig, ax = plt.subplots(figsize=(14, 10))
    metrics_df[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax)
    
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, axis='y')
    
    # Add count labels on top of bars
    for i, count in enumerate(metrics_df['Count']):
        ax.annotate(f'n={count}', 
                   xy=(i, 1.01), 
                   ha='center',
                   va='bottom',
                   fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(folder + "/per_class_metrics.png")
    plt.close(fig)
    
    return metrics_df


def plot_confidence_histogram(y_score, y_true, y_pred, folder, bins=10):
    """Plot histogram of prediction confidence, split by correct/incorrect predictions.
    
    Args:
        y_score: Array of predicted probabilities for each class
        y_true: Array of true class labels
        y_pred: Array of predicted class labels
        folder: Directory to save the plot
        bins: Number of bins for the histogram
    """
    # Get the confidence (probability) assigned to the predicted class for each sample
    confidences = np.array([y_score[i, pred] for i, pred in enumerate(y_pred)])
    
    # Identify correct and incorrect predictions
    correct = (y_pred == y_true)
    incorrect = ~correct
    
    correct_confidences = confidences[correct]
    incorrect_confidences = confidences[incorrect]
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histograms for correct and incorrect predictions
    ax.hist(correct_confidences, bins=bins, alpha=0.6, color='green', label='Correct predictions')
    ax.hist(incorrect_confidences, bins=bins, alpha=0.6, color='red', label='Incorrect predictions')
    
    ax.set_title('Histogram of Prediction Confidence')
    ax.set_xlabel('Confidence (probability)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True)
    
    # Add mean confidence annotations
    mean_correct = np.mean(correct_confidences) if len(correct_confidences) > 0 else 0
    mean_incorrect = np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else 0
    
    ax.axvline(x=mean_correct, color='green', linestyle='--', 
              label=f'Mean correct: {mean_correct:.3f}')
    ax.axvline(x=mean_incorrect, color='red', linestyle='--',
              label=f'Mean incorrect: {mean_incorrect:.3f}')
    
    # Add text showing total counts and percentages
    total = len(y_true)
    n_correct = len(correct_confidences)
    n_incorrect = len(incorrect_confidences)
    
    ax.text(0.02, 0.95, f'Total samples: {total}\n'
                     f'Correct: {n_correct} ({n_correct/total*100:.1f}%)\n'
                     f'Incorrect: {n_incorrect} ({n_incorrect/total*100:.1f}%)',
             transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.savefig(folder + "/confidence_histogram.png")
    plt.close(fig)


def plot_sorted_confusion_matrix(y_true, y_pred, classes, folder):
    """Plot confusion matrix sorted by recall values (ascending).
    
    Args:
        y_true: Array of true class labels
        y_pred: Array of predicted class labels
        classes: List of class names
        folder: Directory to save the plot
    """
    from sklearn.metrics import confusion_matrix, recall_score
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    
    # Calculate per-class recall
    recall = recall_score(y_true, y_pred, average=None, labels=range(len(classes)), zero_division=0)
    
    # Sort classes by recall (ascending)
    sorted_indices = np.argsort(recall)
    sorted_cm = cm[sorted_indices, :]
    sorted_cm = sorted_cm[:, sorted_indices]
    sorted_classes = [classes[i] for i in sorted_indices]
    
    # Plot sorted confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    im = sns.heatmap(sorted_cm, ax=ax, cmap="Blues", norm=LogNorm(), 
                     annot=True, fmt="d", cbar=True)
    
    # Set labels and title
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix (Sorted by Recall - Ascending)')
    
    # Set tick labels
    num_classes = len(sorted_classes)
    if num_classes <= 20:  # Only show class names if there aren't too many
        ax.set_xticks(np.arange(len(sorted_classes)) + 0.5)
        ax.set_yticks(np.arange(len(sorted_classes)) + 0.5)
        ax.set_xticklabels(sorted_classes, rotation=45, ha="right")
        ax.set_yticklabels(sorted_classes)
    
    # Add recall values to y-tick labels
    sorted_recall = recall[sorted_indices]
    if num_classes <= 20:
        labels = [f"{classes[i]} (R:{sorted_recall[idx]:.2f})" 
                for idx, i in enumerate(sorted_indices)]
        ax.set_yticklabels(labels)
    
    fig.tight_layout()
    fig.savefig(folder + "/confusion_matrix_sorted.png")
    plt.close(fig)


def save_feature_maps(model, img, label, epoch, visualization_dir):
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