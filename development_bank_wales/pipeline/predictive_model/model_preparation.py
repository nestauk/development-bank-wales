# File: development_bank_wales/pipeline/predictive_model/model_preparation.py
"""
Prepare the features and model before training, including pipeline for easy processing.
"""
# ----------------------------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from development_bank_wales.pipeline.feature_preparation import (
    feature_encoding,
    feature_selection,
)

from development_bank_wales import PROJECT_DIR, get_yaml_config

config = get_yaml_config(PROJECT_DIR / "development_bank_wales/config/base.yaml")


# ----------------------------------------------------------------------------------


class FeatureSelection(BaseEstimator, TransformerMixin):
    """Class for selecting the right features based on labels."""

    def __init__(self, label):
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = feature_selection.feature_dict[self.label]
        X = X[features]
        return X


class FeatureEncoding(BaseEstimator, TransformerMixin):
    """Class for encoding the features while maintaining feature names."""

    def __init__(self, unaltered_features):
        self.unaltered_features = unaltered_features
        self.feature_list = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = feature_encoding.feature_encoding_pipeline(
            X, unaltered_features=self.unaltered_features
        )
        self.feature_list = list(X.columns)
        return X


class SampleBalancing(BaseEstimator, TransformerMixin):
    """Class for balancing samples with binary labels."""

    def __init__(self, label, target_false_ratio, binary):
        self.label = label
        self.target_false_ratio = target_false_ratio
        self.binary = binary

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.target_false_ratio is None:
            return X
        else:
            X = balance_set(X, self.label, self.target_false_ratio, self.binary)
            return X


class NumpyFeatures(BaseEstimator, TransformerMixin):
    """Class for turning pd.DataFrame features into np.array features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.array(X)
        return X


def balance_set(X, target_variable, target_false_ratio=0.8, binary=True):
    """Balance the training set.
    If false ratio set to 0.8, then 80% of the training data
    will have "False/No HP Installed" as a label.

    Args:
        X (pd.DataFrame): Training set.
        target_variable (str): Variable/feature that is going to be predicted.
        target_false_ratio (float, optional): How many False samples.
            When re-balancing the set, use the false_ratio
            to determine the amount of False labels. Defaults to 0.8.
        binary (bool, optional): Binary labels. Defaults to True.

    Returns:
        X (pd.DataFrame): Re-balanced training set.
    """

    if target_false_ratio > 1.0:
        raise IOError("False ratio needs to be lower than 1.0")

    multiplier = target_false_ratio / (1 - target_false_ratio)

    if binary:
        # Seperate samples with and without heat pumps
        X_true = X[X[target_variable]]
        X_false = X[~X[target_variable]]

        # If zero vs. non-zero
    else:
        X_true = X.loc[X[target_variable] > 0.0]
        X_false = X.loc[X[target_variable] == 0.0]

    # Shuffle and adjust size
    X_false = X_false.sample(frac=1)

    # Get the appropriate amount of "false" samples
    X_false = X_false[: int(X_true.shape[0] * multiplier)]

    # Concatenate "true" and "false" samples
    X = pd.concat([X_true, X_false], axis=0)

    # Reshuffle
    X = X.sample(frac=1)

    return X


def feature_prep_pipeline(features, label, pca=False):
    """Pipeline for preparing features for predictive model.
    Pipeline includes sample balancing, feature selection and encoding,
    as well as data imputing and scaling.
    The pipeline also captures the feature names (as after one-hot encoding).

    Args:
        features (pd.DataFrame): Features (including label) as dataframe.
        label (str): Label feature.
        pca (bool, optional): Whether or not reduce dimensions with PCA. Defaults to False.

    Returns:
        features (np.array): Prepared features.
        labels  (np.array): Labels.
        feature_list (list): List of features (left after the pipeline).

    """

    balancing_dict = {
        "ROOF_UPGRADABILITY": config["roof_upgradability_false_ratio"],
        "WALLS_UPGRADABILITY": None,  # balancing not necessary
        "FLOOR_UPGRADABILITY": None,  # balancing not necessary
    }

    balance_ratio = balancing_dict[label]

    pipeline_elements = [
        ("sample_balancing", SampleBalancing(label, balance_ratio, True)),
        ("feature_selection", FeatureSelection(label)),
        ("feature_encoding", FeatureEncoding([])),
        ("numpy_features", NumpyFeatures()),
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("min_max_scaler", MinMaxScaler()),
    ]

    if pca:
        pipeline_elements.append(("pca", PCA(n_components=0.9, random_state=42)))

    feature_prepping = Pipeline(pipeline_elements)

    labels = np.array(features[label])

    features = feature_prepping.fit_transform(features)
    feature_list = feature_prepping["feature_encoding"].feature_list

    return features, labels, feature_list
