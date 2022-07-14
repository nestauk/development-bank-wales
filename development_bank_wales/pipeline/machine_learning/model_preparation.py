import pandas as pd
import numpy as np
from development_bank_wales.pipeline.feature_preparation import (
    feature_selection,
    one_hot_encoding,
)


from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, label):
        # self.something enables you to include the passed parameters
        # as object attributes and use it in other methods of the class
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = feature_selection.feature_dict[self.label]
        X = X[features]
        return X


class FeatureEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, unaltered_features):
        self.unaltered_features = unaltered_features
        self.feature_list = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = one_hot_encoding.feature_encoding_pipeline(
            X, unaltered_features=self.unaltered_features
        )
        self.feature_list = list(X.columns)
        return X


class SampleBalancing(BaseEstimator, TransformerMixin):
    def __init__(self, label, false_ratio, binary):
        self.label = label
        self.false_ratio = false_ratio
        self.binary = binary

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.false_ratio is None:
            return X
        else:
            X = balance_set(X, self.label, self.false_ratio, self.binary)
            return X


class NumpyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.array(X)
        # X = np.nan_to_num(X)
        return X


def balance_set(X, target_variable, false_ratio=0.8, binary=True):
    """Balance the training set.
    If false ratio set to 0.8, then 80% of the training data
    will have "False/No HP Installed" as a label.

    Parameters
    ----------
    X: pandas.DataFrame
        Training set.

    target_variable : str
        Variable/feature that is going to be predicted.

    false_ratio : float, default=0.8
        When re-balancing the set, use the false_ratio
        to determine the amount of False labels.

    Return
    ---------
    X: pandas.DataFrame
        Re-balanced training set."""

    multiplier = false_ratio / (1 - false_ratio)

    if binary:
        # Seperate samples with and without heat pumps
        X_true = X.loc[X[target_variable] == True]
        X_false = X.loc[X[target_variable] == False]
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


def feature_preparation(features, label, pca=False):

    balancing_dict = {
        "ROOF_UPGRADABILITY": 0.75,
        "WALLS_UPGRADABILITY": None,
        "FLOOR_UPGRADABILITY": None,
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
