import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


def plot_esm_amino_acids_embeddings(amino_acids_embeddings, out_file_path="heatmap.png"):
    """
    Plots a heatmap of ESM amino acids embeddings (first 100 features for readability).
    :param amino_acids_embeddings: ESM amino acids embeddings of a single sequence (shape: n_amino_acids, n_features).
    :param out_file_path: The path for the output png
    """
    # Creating heatmap
    plt.figure(figsize=(10,10))
    # Plot only first 100 features
    plt.matshow(amino_acids_embeddings[:,:100])

    # Adding labels and title
    plt.xlabel('Features')
    plt.ylabel('Amino Acid')
    plt.title('Heatmap')

    # Display plot
    plt.savefig(out_file_path)

def plot_boxplot(data_dict, out_file_path="boxplot.png"):
    """
    Plots a boxplot of the negative and positive distances
    :param data_dict: Dictionary of positive and negative distances
    :param out_file_path: The path for the output png
    """
    # Creating a list of data to plot
    plot_data = [data_dict[key] for key in data_dict]

    # Labels for x-axis ticks and colors for each box
    labels = list(data_dict.keys())

    # Creating boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(plot_data, patch_artist=True, labels=labels)

    # Adding labels and title
    plt.xlabel('Label')
    plt.ylabel('Score')
    plt.title('Boxplot')

    # Display plot
    plt.grid(True)
    plt.savefig(out_file_path)


def plot_roc_curve(y_test, y_scores, out_file_path="roc_curve.png"):
    """
    Plots a ROC curve of the negative and positive distances
    :param y_test: True data labels (0 for negative, 1 for positive)
    :param y_scores: Distances from the positive training peptides (multiplied by -1)
    :param out_file_path: The path for the output png
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    print(F"AUC: {roc_auc}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out_file_path)


def plot_2dim_reduction(lower_dim_coords, labels, out_file_path):
    """
    Scatter plot 2D coordinates with points colored by `labels`.
    :param lower_dim_coords: 2D coordinates
    :param labels: Class or cluster labels used to color points.
    :param out_file_path: The path for the output png
    """

    unique_labels = np.unique(labels)
    labels = np.array(labels)
    lower_dim_coords = np.array(lower_dim_coords)

    plt.figure(figsize=(6, 5))
    for lab in unique_labels:
        idx = (labels == lab)
        plt.scatter(
            lower_dim_coords[idx, 0],
            lower_dim_coords[idx, 1],
            label=str(lab),
        )

    plt.xlabel("DIM-1")
    plt.ylabel("DIM-2")
    plt.legend(title="Label", loc="best")
    plt.tight_layout()

    plt.savefig(out_file_path)
