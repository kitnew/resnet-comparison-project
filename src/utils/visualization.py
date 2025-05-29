from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion(y_true, y_pred, classes, epoch, folder):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, ax=ax, cmap="Blues", norm=LogNorm())
    ax.set_xlabel("Pred"); ax.set_ylabel("True")
    fig.savefig(folder / f"confmat_ep{epoch}.png"); plt.close(fig)
    writer.add_figure("ConfMat", fig, epoch)
