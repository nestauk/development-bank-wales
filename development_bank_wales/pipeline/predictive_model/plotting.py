# File: development_bankd_wales/pipeline/predictive_model/plotting.py
"""
Create various charts for evaluating and studying predictive models.
For example: plot feature coefficients or confusion matrix.
"""
# ----------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix

from development_bank_wales import PROJECT_DIR, get_yaml_config, Path

# ----------------------------------------------------------------------------------


# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/development_bank_wales/config/base.yaml")
)

FIG_PATH = Path(PROJECT_DIR) / config["SUPERVISED_MODEL_FIG_PATH"]


def get_most_important_coefficients(model, feature_names, title, X, top_features=10):
    """Get features with strongest coefficients (both positive or negative).

    Args:
        model (sklearn.Model): Fitted sklearn model.
        feature_names (list): List of feature names.
        title (str): Plot title.
        X (np,array): Training data.
        top_features (int, optional): How many features to display on each side. Defaults to 10.
    """

    if model.__class__.__name__ == "SVC":
        coef = model.coef_
        coef2 = coef.toarray().ravel() * X.std(axis=0)
        coef1 = coef2[: len(feature_names)]
    else:
        coef2 = model.coef_.ravel() * X.std(axis=0)
        coef1 = coef2[: len(feature_names)]

    # Get top coefficients
    top_positive_coefficients = np.argsort(coef1)[-top_features:]
    top_negative_coefficients = np.argsort(coef1)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    # Create plot
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef1[top_coefficients]]
    plt.bar(
        np.arange(2 * top_features),
        coef1[top_coefficients],
        color=colors,
        align="center",
    )

    feature_names = np.array(feature_names)

    plt.xticks(
        np.arange(0, 2 * top_features),
        feature_names[top_coefficients],
        rotation=45,
        ha="right",
    )
    plt.title(title)
    plt.savefig(
        FIG_PATH / (title.replace(":", " -") + ".png"), dpi=200, bbox_inches="tight"
    )

    plt.show()


def get_sorted_coefficients(classifier, feature_names):
    """Get features and coefficients sorted by coeffience strength in Linear SVM.

    Args:
        classifier (sklearn.model): Fitted sklearn model.
        feature_names (list): List of feature names.

    Returns:
        sort_idx (np.array): Sorting array for features (feature with strongest coeffienct first).
        sorted_coef (np.array): Sorted coefficient values.
        sorted_fnames (list): Feature names sorted by coefficient strength.
    """

    # Sort the feature indices according absolute coefficients (highest coefficient first)
    sort_idx = np.argsort(-abs(classifier.coef_).max(axis=0))

    # Get sorted coefficients and feature names
    sorted_coef = classifier.coef_[:, sort_idx]
    sorted_fnames = feature_names[sort_idx].tolist()

    sorted_fnames = [feature_names[i] for i in sort_idx]

    return sort_idx, sorted_coef, sorted_fnames


def plot_feature_coefficients(classifier, feature_names, label_set, title):
    """Plot the feature coefficients for each label given an SVM classifier.

    Args:
        classifier (sklearn.classifier_model): Fitted sklearn model.
        feature_names (list): List of feature names.
        label_set (list): List of ground truth labels.
        title (str): Plot title.
    """

    # Layout settings depending un number of labels
    if len(label_set) > 4:
        FIGSIZE = (80, 30)
        ROTATION = 35
        RIGHT = 0.81

    else:
        FIGSIZE = (40, 12)
        ROTATION = 45
        RIGHT = 0.58

    # Sort the feature indices according coefficients (highest coefficient first)
    sort_idx = np.argsort(-abs(classifier.coef_).max(axis=0))

    # Get sorted coefficients and feature names
    sorted_coef = classifier.coef_[:, sort_idx]
    sorted_fnames = feature_names[sort_idx]

    # Make subplots
    x_fig, x_axis = plt.subplots(2, 1, figsize=FIGSIZE)

    odd_n_rows = False
    if (sorted_coef.shape[1] % 2) != 0:
        odd_n_rows = True
        second_row_n = (sorted_coef.shape[1] // 2) + 1

    # Plot coefficients on two different lines
    im_0 = x_axis[0].imshow(
        sorted_coef[:, : sorted_coef.shape[1] // 2],
        interpolation="none",
        cmap="seismic",
        vmin=-2.5,
        vmax=2.5,
    )
    im_1 = x_axis[1].imshow(
        sorted_coef[:, sorted_coef.shape[1] // 2 :],
        interpolation="none",
        cmap="seismic",
        vmin=-2.5,
        vmax=2.5,
    )

    x_axis[0].set_title(
        title + "\n",
        fontdict={"fontsize": 30, "fontweight": "medium"},
    )

    # Set y ticks (number of classes)
    x_axis[0].set_yticks(range(len(label_set)))
    x_axis[1].set_yticks(range(len(label_set)))

    # Set the y labels (classes/labels)
    x_axis[0].set_yticklabels(label_set, fontsize=24)
    x_axis[1].set_yticklabels(label_set, fontsize=24)

    # Set x ticks (half the number of features) and labels
    x_axis[0].set_xticks(range(len(feature_names) // 2))
    if odd_n_rows:
        x_axis[1].set_xticks(range(second_row_n))
    else:
        x_axis[1].set_xticks(range(len(feature_names) // 2))

    # Set the x labels (feature names)
    x_axis[0].set_xticklabels(
        sorted_fnames[: len(feature_names) // 2],
        rotation=ROTATION,
        ha="right",
        fontsize=20,
    )
    x_axis[1].set_xticklabels(
        sorted_fnames[sorted_coef.shape[1] // 2 :],
        rotation=ROTATION,
        ha="right",
        fontsize=20,
    )

    plt.tight_layout()

    # Move plot to the right
    x_fig.subplots_adjust(right=RIGHT)

    # Set color bar
    cbar_ax = x_fig.add_axes([0.605, 0.15, 0.02, 0.7])
    cbar = x_fig.colorbar(im_0, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=24)

    # plt.title(title)
    plt.savefig(FIG_PATH / (title.replace(":", " -") + ".png"), format="png", dpi=500)

    # Show
    plt.show()


def plot_confusion_matrix(
    solutions, predictions, label_set=None, title="", plot_type="plt"
):
    """Plot the confusion matrix for different classes given correct labels and predictions.

    Args:
        solutions (np.array) : Ground truth labels.
        predictions (np.array): Predicted labels.
        label_set (list, optional): Set list of labels. Defaults to None.
        title (str, optional): Plot title. Defaults to "".
        plot_type (str, optional): Whether to use "plt" or "searborn" style. Defaults to "plt".
    """

    if plot_type == "plt":

        # Compute confusion matrix
        cm = sklearn.metrics.confusion_matrix(
            solutions, predictions, labels=range(len(label_set))
        )

        # Set figure size
        if len(label_set) > 5:
            plt.figure(figsize=(10, 10))
        else:
            plt.figure(figsize=(3, 3))

        # Plot  confusion matrix with blue color map
        plt.imshow(cm, interpolation="none", cmap="Blues")

        # Write out the number of instances per cell
        for (i, j), z in np.ndenumerate(cm):
            plt.text(j, i, z, ha="center", va="center")

        # Assign labels and title
        plt.xlabel("Prediction")
        plt.ylabel("Ground truth")
        title = "Confusion Matrix for {}".format(title)
        plt.title(title)

        # Set x ticks and labels
        plt.gca().set_xticks(range(len(label_set)))
        plt.gca().set_xticklabels(label_set, rotation=50)

        # Set y ticks and labels
        plt.gca().set_yticks(range(len(label_set)))
        plt.gca().set_yticklabels(label_set)
        plt.gca().invert_yaxis()

        plt.savefig(
            FIG_PATH / (title.replace(":\n", " -") + ".png"),
            format="png",
            dpi=500,
            bbox_inches="tight",
        )

        # Show plot
        plt.show()

    else:

        matrix = confusion_matrix(solutions, predictions)
        matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

        # Build the plot
        plt.figure(figsize=(10, 7))
        sns.set(font_scale=1.4)
        sns.heatmap(
            matrix,
            annot=True,
            annot_kws={"size": 10},
            cmap=plt.cm.Greens,
            linewidths=0.2,
        )

        # Add labels to the plot
        class_names = ["Not upgradable", "Upgradable"]
        tick_marks = np.arange(len(label_set))
        tick_marks2 = tick_marks + 0.5
        plt.xticks(tick_marks, label_set, rotation=25)
        plt.yticks(tick_marks2, label_set, rotation=0)
        plt.xlabel("Predicted label")
        plt.ylabel("Ground truth label")
        plt.title("Confusion Matrix for {}".format(title))
        plt.savefig(
            FIG_PATH / (title.replace(":\n", " -") + "_seaborn.png"),
            format="png",
            dpi=500,
            bbox_inches="tight",
        )
        plt.show()


def plot_explained_variance(dim_reduction, title):
    """Plot percentage of variance explained by each of the selected components
    after performing dimensionality reduction (e.g. PCA, LSA).

    Explained variance ratio: how much is covered by how many components

    Args:
        dim_reduction (sklearn.decomposition): Dimensionality reduction on features with PCA or LSA.
        title (_type_): Plot title.
    """

    # Per component
    plt.plot(dim_reduction.explained_variance_ratio_)
    # Cumulative
    plt.plot(np.cumsum(dim_reduction.explained_variance_ratio_))

    title = "Explained Variance Ratio by Dimensions " + title

    # Assign labels and title
    plt.xlabel("Dimensions")
    plt.ylabel("Explained variance")
    plt.legend(["Explained Variance Ratio", "Summed Expl. Variance Ratio"])
    plt.title(title)

    plt.savefig(FIG_PATH / title, format="png", dpi=500)

    # Show plot
    plt.show()
