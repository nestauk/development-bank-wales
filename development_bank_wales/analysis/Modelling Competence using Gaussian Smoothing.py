# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: development_bank_wales
#     language: python
#     name: development_bank_wales
# ---

# %% [markdown]
# ## Modelling Competence using Gaussian Smoothing
#
# Last updated: 18 Jult 2022
#
# We hope to identify the properties with the potential for upgrades, meaning those that have not been upgraded yet but are very similar to those you have (e.g. blue line splitting feature space below).
#
# However, our upgradability model is based on observed upgrades and recommendations, which represents the _performance upgradability_ of a property, but not its competence. A predictive model will try to find a barrier between properties with actual upgrades and those without (purple line), so a 'perfect' model will not give us the information we actually need.
#
# ![comp_vs_perf.png](attachment:comp_vs_perf.png)

# %% [markdown]
# This notebook is an attempt of modelling 'competence' using gaussian smoothing, which gives less weight to the absolute position of a sample in the feature space and instead focuses on its neighbours. The original label of each sample is smoothed by integrating information about how many neighbours have had upgrades.
#
# This analysis is supposed to identify whether there is a signifant gap between performance and competence, i.e. the respective models for them. The logistic regression model represents the performance, the gaussian smoothing model represents the competenence.
#
# ![gaussian_smoothing.png](attachment:gaussian_smoothing.png)

# %% [markdown]
# ## Loading and preparing the data
#
# The first time, this takes about 5min. Time for a quick break!

# %%
# %load_ext autoreload
# %autoreload 2

import pandas as pd


from development_bank_wales import PROJECT_DIR, Path

from development_bank_wales.pipeline.feature_preparation import (
    recommendations,
    upgrades,
    feature_engineering,
)

from development_bank_wales.pipeline.predictive_model import (
    model_preparation,
    plotting,
    evaluation,
    training,
    gaussian_smoothing,
)

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import numpy as np
import scipy

import matplotlib.pyplot as plt

plt.style.use("default")
plt.style.use("seaborn")
# %matplotlib inline
plt.rcParams["figure.figsize"] = (7, 7)


import warnings

warnings.simplefilter(action="ignore")

# %% [markdown]
# ### Loading the data

# %%
output_path = PROJECT_DIR / "outputs/data/wales_epc_with_recs.csv"
fig_output_path = PROJECT_DIR / "outputs/figures/"

if not Path(output_path).is_file():

    print("Loading and preparing the data...")

    wales_df = recommendations.load_epc_certs_and_recs(
        data_path="S3", subset="Wales", n_samples=None, remove_duplicates=False
    )

    wales_df.to_csv(output_path, index=False)

    print("Done!")

else:

    print("Loading the data...")
    wales_df = pd.read_csv(output_path)
    print("Done!")

# %% [markdown]
# ### Prepare features
#
# We clean the description features, reduce the set to owner-occupied properties and retrieve information about upgrades and upgradability scores for the different categories.
#
# Put the features through the preparation pipeline that handles the following:
# - sample balancing
# - feature selection (given specific label)
# - feature encoding
# - imputing
# - scaling
# - if needed: principal component analysis
#
# Finally, get a test and train split for evaluation.
#

# %%
wales_df = feature_engineering.get_upgrade_features(wales_df)

label = "ROOF_UPGRADABILITY"
features, labels, feature_list = model_preparation.feature_prep_pipeline(
    wales_df, label
)

# %%
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42, stratify=labels
)

print("Training Features Shape:", train_features.shape)
print("Training Labels Shape:", train_labels.shape)
print("Testing Features Shape:", test_features.shape)
print("Testing Labels Shape:", test_labels.shape)

# %% [markdown]
# ## Gaussian smoothing
#
# Apply gaussian smoothing to the original labels. For sanity check, test whether properties with an original 'True' label turn are sorted into the right-hand side of the distribution if they are flipped to False.

# %%
smoothed_labels, org_labels, _ = gaussian_smoothing.get_smoothed_labels(
    test_features, test_labels, n_samples=10000, flip_indices=False, random_state=42
)
(
    smoothed_labels_flipped,
    flipped_labels,
    flipped_indices,
) = gaussian_smoothing.get_smoothed_labels(
    test_features, test_labels, n_samples=10000, flip_indices=True, random_state=42
)

# %% [markdown]
# ### Plotting smoothed labels
#
# When looking at the properties originally labeled as 'not upgradable', we see that there seems to be split of the data into two peaks. We expect that most properties with flipped labels (from True to False) will come out at the right-hand peak of the distribution, proving that Gaussian smoothing can raise the upgradability score of a "non upgradable" property that is very similar to the upgraded ones.
#
# We evaluate by flipping 5 different sets of 100 samples. On average, more than 60% are on the higher end up on the upgradability scale.

# %%
plt.style.use("seaborn")
plt.hist(smoothed_labels, bins=100)

plt.ylabel("# samples")
plt.xlabel("Smoothed label")
plt.tight_layout()
plt.title("Histogram (all smothed labels)")
plt.savefig(fig_output_path / "Histogram_all_smoothed_labels.png")
plt.show()

# %%
upgradables = smoothed_labels[~org_labels.astype(bool)]

plt.hist(upgradables, bins=100)

plt.title("Histogram for properties labelled False originally")
plt.ylabel("# samples")
plt.xlabel("Smoothed upgradability label")
plt.tight_layout()
plt.savefig(fig_output_path / "Histogram_only_False_labelled.png")
plt.show()

# %% [markdown]
#

# %%
smoothed_labels_flipped, _, flipped_indices = gaussian_smoothing.get_smoothed_labels(
    test_features, test_labels, n_samples=10000, flip_indices=True, random_state=42
)

plt.hist(smoothed_labels_flipped[flipped_indices], bins=100)
plt.title("Histogram: flipped samples")
plt.ylabel("# samples")
plt.xlabel("Smoothed upgradability label")
plt.tight_layout()
plt.savefig(fig_output_path / "Histogram_flipped_samples.png")
plt.show()

# %%
upper_end_ratios = []
percentile = 70

for i in range(5):

    (
        smoothed_labels_flipped,
        _,
        flipped_indices,
    ) = gaussian_smoothing.get_smoothed_labels(
        test_features, test_labels, n_samples=10000, flip_indices=True
    )

    upper_end_ratio = smoothed_labels_flipped[flipped_indices] > np.percentile(
        smoothed_labels_flipped, 70
    )
    upper_end_ratio = upper_end_ratio.sum() / upper_end_ratio.shape[0]

    print(
        "{}) Flipped samples that come out at the highest 30%? {}%".format(
            i + 1, upper_end_ratio
        )
    )

    upper_end_ratios.append(upper_end_ratio)

print(
    "\nFlipped samples that come out at the highest 30%? Mean: {:1.1f}%".format(
        sum(upper_end_ratios) / len(upper_end_ratios) * 100
    )
)

# %% [markdown]
# ### Feature space before and after Gaussian smoothing
#
# These t-SNE plots represent the feature space of the properties. The first plot shows the orignal labels of the samples. The 'non-upgradable' properties are clustered together.
#
# The second shows the smoothed labels: properties with similar

# %%
n_samples = 10000
(
    smoothed_labels,
    original_labels,
    flipped_indices,
) = gaussian_smoothing.get_smoothed_labels(
    test_features, test_labels, n_samples=10000, flip_indices=True, random_state=42
)

tsne = TSNE()
test_tsne = tsne.fit_transform(test_features[:n_samples])

# %%
scatter = plt.scatter(
    test_tsne[:, 0], test_tsne[:, 1], c=original_labels, cmap="viridis", alpha=0.25
)
plt.title("Upgradable and non-upgradable properties in feature space")

plt.legend(
    handles=scatter.legend_elements()[0],
    labels=["1", "0"],
    title="Original upgradability",
)
plt.savefig(fig_output_path / "Feature_space_labes.png")

# %%
scatter = plt.scatter(
    test_tsne[~original_labels.astype(bool).flatten(), 0],
    test_tsne[~original_labels.astype(bool).flatten(), 1],
    c=smoothed_labels[~original_labels.astype(bool).flatten()],
    alpha=0.25,
    cmap="viridis",
)

plt.colorbar(label="Smoothed upgradability", orientation="horizontal")
plt.title("Originally non-upgradable properties")
plt.savefig(fig_output_path / "Feature_space_labes.png")

# %%
plt.scatter(
    test_tsne[flipped_indices, 0],
    test_tsne[flipped_indices, 1],
    c=smoothed_labels[flipped_indices],
    alpha=0.25,
    cmap="viridis",
)

plt.colorbar(label="Smoothed upgradability", orientation="horizontal")
plt.title("'Flipped' properties")
plt.savefig(fig_output_path / "Feature_space_labes.png")

# %% [markdown]
# ## Gaussian Smoothing vs. Logistic Regression
#
# Gaussian smoothing is not ideal for predicting new data, but we can compare the results to a predictive model to study the difference between competence and performance.
#
# The properties with the highest 25% upgradability of all data points are considered "upgradable", no matter the original label. We achieve an accuracy of 76% compared to 82%, meaning that there is a difference between the two models. However, the different might not be large enough to affect the original upgradability predictions.

# %%
n_samples = 10000

smoothed_labels, _, _ = gaussian_smoothing.get_smoothed_labels(
    test_features, test_labels, n_samples=n_samples, flip_indices=False
)

smooth_predictions = (smoothed_labels > np.percentile(smoothed_labels, 75)).astype(int)

evaluation.print_metrics(test_labels[:n_samples], smooth_predictions[:n_samples])

# %%
model = training.model_dict["Logistic Regression"]
model.fit(train_features, train_labels)

y_pred_test = model.predict(test_features)
y_pred_train = model.predict(train_features)
evaluation.print_metrics(test_labels[:n_samples], y_pred_test[:n_samples])

# %%
plotting.plot_confusion_matrix(
    test_labels[:n_samples],
    smooth_predictions[:n_samples],
    label_set=["Not upgradable", "Upgradable"],
    title="Gaussian Smoothing",
    plot_type="seaborn",
)

# %%
plotting.plot_confusion_matrix(
    test_labels[:n_samples],
    y_pred_test[:n_samples],
    label_set=["Not upgradable", "Upgradable"],
    title="Logistic Regression",
    plot_type="seaborn",
)

# %%
smooth_log_comp = smooth_predictions.flatten()[:n_samples] == y_pred_test[:n_samples]
ratio = smooth_log_comp.sum() / smooth_log_comp.shape[0] * 100
print(
    "Overlap of smoothed predictions and logistic regression predictions: {:0.1f}%".format(
        ratio
    )
)

# %%
smooth_label_comp = smooth_predictions.flatten()[:n_samples] == test_labels[:n_samples]
ratio = smooth_label_comp.sum() / smooth_label_comp.shape[0] * 100
print("Overlap of smoothed predictions and labels: {:0.1f}%".format(ratio))

# %%
log_label_comp = y_pred_test[:n_samples] == test_labels[:n_samples]
ratio = log_label_comp.sum() / log_label_comp.shape[0] * 100
print("Overlap of logistic regression predictions and labels: {:0.1f}%".format(ratio))

# %%
smooth_log_label_comp = (
    smooth_predictions.flatten()[:n_samples] == y_pred_test[:n_samples]
) & (y_pred_test[:n_samples] == test_labels[:n_samples])
ratio = smooth_log_label_comp.sum() / smooth_log_label_comp.shape[0] * 100
print(
    "Overlap of smoothed predictions and logistic regression predictions and labels: {:0.1f}%".format(
        ratio
    )
)

# %%
