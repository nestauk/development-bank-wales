# File: development_bankd_wales/pipeline/feature_preparation/feature_encoding.py
"""
Encoding categorical features with ordinal and one-hot encoding.
Note: this script will be integrated to asf-core-data.
"""

# ----------------------------------------------------------------------------------

# Import
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------


order_dict = {
    "CURRENT_ENERGY_RATING": ["unknown", "G", "F", "E", "D", "C", "B", "A"],
    "POTENTIAL_ENERGY_RATING": ["unknown", "G", "F", "E", "D", "C", "B", "A"],
    "NUMBER_HABITABLE_ROOMS": [
        "unknown",
        "0.0",
        "1.0",
        "2.0",
        "3.0",
        "4.0",
        "5.0",
        "6.0",
        "7.0",
        "8.0",
        "9.0",
        "10+",
    ],
    "MAINS_GAS_FLAG": ["N", "unknown", "Y"],
    "CONSTRUCTION_AGE_BAND_ORIGINAL": [
        "England and Wales: before 1900",
        "Scotland: before 1919",
        "England and Wales: 1900-1929",
        "Scotland: 1919-1929",
        "England and Wales: 1930-1949",
        "Scotland: 1930-1949",
        "Scotland: 1950-1964",
        "England and Wales: 1950-1966",
        "England and Wales: 1967-1975",
        "Scotland: 1965-1975",
        "England and Wales: 1976-1982",
        "Scotland: 1976-1983",
        "England and Wales: 1983-1990",
        "Scotland: 1984-1991",
        "England and Wales: 1991-1995",
        "Scotland: 1992-1998",
        "England and Wales: 1996-2002",
        "Scotland: 1999-2002",
        "England and Wales: 2003-2006",
        "Scotland: 2003-2007",
        "England and Wales: 2007 onwards",
        "Scotland: 2008 onwards",
        "England and Wales: 2012 onwards",
        "unknown",
    ],
    "CONSTRUCTION_AGE_BAND": [
        "England and Wales: before 1900",
        "Scotland: before 1919",
        "1900-1929",
        "1930-1949",
        "1950-1966",
        "1965-1975",
        "1976-1983",
        "1983-1991",
        "1991-1998",
        "1996-2002",
        "2003-2007",
        "2007 onwards",
        "unknown",
    ],
    "N_ENTRIES_BUILD_ID": ["1", "2", "3", "4", "5.0+"],
    "N_ENTRIES": ["1", "2", "3", "4", "5.0+"],
    "FLOOR_LEVEL": [
        "unknown",
        "-1",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10+",
    ],
    "ENERGY_RATING_CAT": ["unknown", "E-G", "C-D", "A-B"],
    "GLAZED_TYPE": ["unknown", "single glazing", "double glazing", "triple glazing"],
    "N_SAME_UPRN_ENTRIES": ["1", "2", "3", "4", "5.0+"],
}

ordinal_features = list(order_dict.keys())

eff_value_dict = {
    "Very Poor": 1,
    "Poor": 2,
    "Average": 3,
    "Good": 4,
    "Very Good": 5,
}


def create_efficiency_mapping(efficiency_set):
    """Create dict to map efficiency label(s) to numeric value.

    Args:
        efficiency_set (list): List of efficiencies as encoded as strings.

    Returns:
        efficiency_map (dict): Dict to map efficiency labels to numeric values.
    """

    efficiency_map = {}

    for eff in efficiency_set:

        # If efficiency is float (incl. NaN)
        if isinstance(eff, float):
            efficiency_map[eff] = 0.0
            continue

        # Split parts of label (especially for Scotland data)
        eff_parts = [
            part.strip()
            for part in eff.split("|")
            if part.strip() not in ["N/A", "unknown", ""]
        ]

        if not eff_parts:
            efficiency_map[eff] = 0
            continue

        # Map labels to numeric value and take mean
        eff_value = sum([eff_value_dict[part] for part in eff_parts]) / float(
            len(eff_parts)
        )

        efficiency_map[eff] = round(eff_value, 1)

    return efficiency_map


def ordinal_encode_cat_features(df, features):
    """Ordinal encode given categorical features.

    Args:
        df (pd.DataFrame): Dataframe including features to encode.
        features (list): Features to ordinal encode.

    Returns:
        df (pd.DataFrame): Dataframe with ordinal encoded features.
    """

    for feat in features:

        if feat not in df.columns:
            continue

        # If efficiency feature, get respective mapping
        if feat.endswith("_EFF"):
            map_dict = create_efficiency_mapping(list(df[feat].unique()))
        else:
            # General mapping given ordered list of categories
            map_dict = dict(zip(order_dict[feat], range(1, len(order_dict[feat]) + 1)))

        # Encode features
        df[feat] = df[feat].map(map_dict)

    return df


def one_hot_encoding(df, features, limit=40, verbose=True):
    """One-hot encode given categorical features.

    Args:
        df (pd.DataFrame):  Dataframe including features to encode.
        features (list): Features to ordinal encode.
        limit (int, optional): _description_. Defaults to 40.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
         df (pd.DataFrame): Dataframe with one-hot encoded features
    """

    if verbose:
        print("Before one hot encoding:", df.shape)

    # Drop features with only one value or NaN
    df.dropna(axis=1, how="all", inplace=True)
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df.drop(columns=cols_to_drop, inplace=True)

    for feat in features:

        if feat in cols_to_drop:
            continue

        if len(df[feat].unique()) >= limit:
            del df[feat]
            continue

        one_hot = pd.get_dummies(df[feat])

        # Create new column names
        one_hot.columns = [feat + ": " + str(cat) for cat in one_hot.columns]
        false_columns = [col for col in one_hot.columns if col.endswith("False")]
        one_hot.drop(columns=false_columns, inplace=True)

        # Join enocoded features with original df
        df = df.join(one_hot)

        # Drop the original feature
        df = df.drop(feat, axis=1)

    if verbose:
        print("After one hot encoding:", df.shape)

    return df


def feature_encoding_pipeline(
    df,
    ordinal_features=ordinal_features,
    onehot_features="auto",
    unaltered_features=["LMK_KEY", "UPRN"],
    drop_features=None,
    verbose=False,
):
    """Pipeline for encoding ordinal features and one-hot encoding categorical features.

    Args:
        df (pd.DataFrame): Dataframe including features to encode.
        ordinal_features (list, optional): Features to ordinal encode. Defaults to ordinal_features.
        onehot_features (list, optional): Features to one-hot encode.
            If set to "auto", suitable features will be identified automatically.
            To disable one-hot encoding, pass empty list or None. Defaults to "auto".
        unaltered_features (list, optional): Variables that should not be encoded.
            Defaults to ["LMK_KEY", "UPRN"].
        drop_features (list, optional): Features to discard. Defaults to None.
        verbose (bool, optional): Print encoding process. Defaults to False.

    Returns:
        df (pd.DataFrame): Updated and encoded dataframe.
    """

    # Drop featuress
    if drop_features is not None:
        df = df.drop(columns=drop_features)

    # Get all only numeric features
    num_features = df.select_dtypes(include=np.number).columns.tolist()
    num_features = [f for f in num_features if f not in ["BUILDING_ID", "UPRN"]]

    # Ordinal encoding
    df = ordinal_encode_cat_features(df, ordinal_features)

    if verbose:
        print("Numeric featurs:", num_features)
        print("Ordinal features:", ordinal_features)

    # Optional one-hot encoding
    if (onehot_features is not None) or onehot_features:

        # If automatically identifying one-hot features
        if onehot_features == "auto":

            # Get categorical features (exclude ordinally encoded ones)
            categorical_features = [
                feature
                for feature in df.columns
                if (feature not in ordinal_features) and (feature not in num_features)
            ]

            # Convert target variables into list
            keep_features = [] if unaltered_features is None else unaltered_features
            keep_features = (
                list(keep_features) if isinstance(keep_features, str) else keep_features
            )

            # Select features to be one-hot encoded, exclude target variables
            one_hot_features = [
                f for f in categorical_features if f not in keep_features
            ]

        if verbose:
            print("Categorical features:", categorical_features)
            print("Keep features:", keep_features)
            print("One hot features:", one_hot_features)
            print("Unaltered features", unaltered_features)

        # One-hot encoding
        df = one_hot_encoding(df, one_hot_features, verbose=verbose)

    df.sort_index(inplace=True)
    return df
