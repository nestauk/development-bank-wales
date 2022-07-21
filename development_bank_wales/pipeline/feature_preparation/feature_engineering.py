# File: development_bank_wales/pipeline/feature_preparation/feature_engineering.py
"""Extract new features from original description features, e.g. ROOF_DESCRIPTION."""

# ---------------------------------------------------------------------------------

import pandas as pd
from asf_core_data.getters.epc import epc_data

# ---------------------------------------------------------------------------------


cats = [
    "ROOF",
    "WINDOWS",
    "WALLS",
    "FLOOR",
    "LIGHTING",
    "HOT_WATER",
    "MAINHEAT",
]


def roof_description_features(df):
    """Extract roof features from ROOF_DESCRIPTION.
    Note: This function will be added to the asf-core-data package as part of the processing pipeline.

    Args:
        df (pd.DataFrame): Dataframe including column ROOF_DESCRIPTION.

    Returns:
        df (pd.DataFrame): Dataframe with additional roof features.
    """

    df["ROOF TYPE"] = df["ROOF_DESCRIPTION"].str.extract(
        r"(Pitched|Flat|Roof room\(s\)|Flat|Ar oleddf|\(other premises above\)|Other premises above|\(another dwelling above\)|\(eiddo arall uwchben\))|\(annedd arall uwchben\)"
    )

    df["ROOF TYPE"] = df["ROOF TYPE"].replace(["Ar oleddf"], "Pitched")

    df["LOFT INSULATION [in mm]"] = (
        df["ROOF_DESCRIPTION"]
        .str.extract(r"(\d{1,3})\+?\s+mm loft insulation")
        .astype(float)
    )

    df["ROOF THERMAL TRANSMIT"] = (
        df["ROOF_DESCRIPTION"].str.extract(r"\s*(0\.\d{1,3})\s*W\/m").astype(float)
    )
    df["ROOF INSULATION"] = df["ROOF_DESCRIPTION"].str.extract(
        r"(no insulation|insulated at rafters|limited insulation|ceiling insulated|insulated \(assumed\))"
    )

    df["ROOF TYPE"] = df["ROOF TYPE"].replace(
        [
            "(other premises above)",
            "(eiddo arall uwchben)",
            "(another dwelling above)",
            "Other premises above",
            "(annedd arall uwchben)",
        ],
        "another dwelling above",
    )

    return df


def walls_description_features(df):
    """Extract walls features from WALLS_DESCRIPTION.
    Note: This function will be added to the asf-core-data package as part of the processing pipeline.

    Args:
        df (pd.DataFrame): Dataframe including column WALLS_DESCRIPTION.

    Returns:
        df (pd.DataFrame): Dataframe with additional walls features.
    """

    df["WALL TYPE"] = df["WALLS_DESCRIPTION"].str.extract(
        r"(Cavity wall|Sandstone|Solid brick|Sandstone or limestone|System built|Timber frame|Granite or whin|Park home wall|Waliau ceudod|Gwenithfaen|\(other premises below\)|\(another dwelling below\)|\(anheddiad arall islaw\)|\(Same dwelling below\))"
    )

    df["WALL TYPE"] = df["WALL TYPE"].replace(["Waliau ceudod"], "Cavity wall")
    df["WALL TYPE"] = df["WALL TYPE"].replace(["Gwenithfaen"], "Granite or whin")

    df["WALLS THERMAL TRANSMIT"] = (
        df["WALLS_DESCRIPTION"].str.extract(r"\s*(0\.\d{1,3})\s*W\/m").astype(float)
    )

    df["WALLS INSULATION"] = df["WALLS_DESCRIPTION"].str.extract(
        r"(insulated|no insulation|filled cavity|with external insulation|with internal insulation|partial insulated)"
    )

    return df


def floor_description_features(df):
    """Extract floor features from FLOOR_DESCRIPTION.
    Note: This function will be added to the asf-core-data package as part of the processing pipeline.

    Args:
        df (pd.DataFrame): Dataframe including column FLOOR_DESCRIPTION.

    Returns:
        df (pd.DataFrame): Dataframe with additional floor features.
    """

    df["FLOOR TYPE"] = df["FLOOR_DESCRIPTION"].str.extract(
        r"(Solid|Suspended|To unheated space|Solet|To external air)"
    )

    df["FLOOR TYPE"] = df["FLOOR TYPE"].replace(["Solet"], "Solid")
    df["FLOOR TYPE"] = df["FLOOR TYPE"].replace(
        ["I ofod heb ei wresogi"], "To unheated space"
    )

    df["FLOOR TYPE"] = df["FLOOR TYPE"].replace(
        [
            "(other premises below)",
            "(anheddiad arall islaw)",
            "(another dwelling below)",
            "(Same dwelling below)",
        ],
        "another dwelling below",
    )

    df["FLOOR THERMAL TRANSMIT"] = (
        df["FLOOR_DESCRIPTION"].str.extract(r"\s*(0\.\d{1,3})\s*W\/m").astype(float)
    )
    df["FLOOR INSULATION"] = df["FLOOR_DESCRIPTION"].str.extract(
        r"(insulated|no insulation|limited insulation|partial insulated|uninsulated)"
    )

    df["FLOOR INSULATION"] = df["FLOOR INSULATION"].replace(
        ["uninsulated"], "no insulation"
    )
    df["FLOOR INSULATION"] = df["FLOOR INSULATION"].replace(
        ["limited insulatio"], "partial insulated"
    )

    return df


def extract_features_from_desc(df):
    """Extract detailed features from description features such as ROOF_DESCRIPTION (available
    for each category).
    Note: Further functions, e.g. for WINDOWS, will be added in the future.

    Args:
        df (pd.DataFrame): Dataframe including description features.

    Returns:
        df (pd.DataFrame): Dataframe with new features added.
    """

    df = roof_description_features(df)
    df = walls_description_features(df)
    df = floor_description_features(df)

    return df


def get_diff_in_energy_eff(df1, df2, keep="first", identifier="UPRN"):
    """Get difference in energy efficiency between first to latest entry for all categories.

    Args:
        df1 (pd.DataFrame): Earliest/first property records, including ENERGY_EFF features.
        df2 (pd.DataFrame): Latest property records, including ENERGY_EFF features.
        keep (str, optional): Which to keep: earliest/first or latest. Defaults to "first".
        identifier (str, optional): Unique property identifier. Defaults to "UPRN".

    Returns:
        df (pd.DataFrame): Dataframe with difference in energy efficiency for all categories.
    """

    quality_dict = {"Very Good": 5, "Good": 4, "Average": 3, "Poor": 2, "Very Poor": 1}

    for cat in cats:

        # Transform efficiency label to numeric score and compute difference
        eff_feature = "{}_ENERGY_EFF".format(cat)
        df1[eff_feature + "_NUM"] = df1[eff_feature].map(quality_dict)
        df2[eff_feature + "_NUM"] = df2[eff_feature].map(quality_dict)

        combo = pd.merge(
            df1[[eff_feature + "_NUM", identifier]],
            df2[[eff_feature + "_NUM", identifier]],
            on=identifier,
        )

        combo[cat + "_EFF_DIFF"] = (
            combo[eff_feature + "_NUM_y"] - combo[eff_feature + "_NUM_x"]
        )

        # Fill NaNs and fix negative values
        combo[cat + "_EFF_DIFF"].fillna(0.0, inplace=True)
        combo.loc[combo[cat + "_EFF_DIFF"] < 0.0, cat + "_EFF_DIFF"] = 0.0

        # Keep first or latest records for further processing
        if keep == "first":
            df1 = pd.merge(
                df1,
                combo[[cat + "_EFF_DIFF", identifier]],
                on=identifier,
            )
        else:
            df2 = pd.merge(
                df2,
                combo[[cat + "_EFF_DIFF", identifier]],
                on=identifier,
            )

    if keep == "first":
        df = df1
    else:
        df = df2

    # Mean energy efficiency
    df["TOTAL_ENERGY_EFF_NUM"] = df[
        [
            "ROOF_ENERGY_EFF_NUM",
            "WALLS_ENERGY_EFF_NUM",
            "HOT_WATER_ENERGY_EFF_NUM",
            "MAINHEAT_ENERGY_EFF_NUM",
            "LIGHTING_ENERGY_EFF_NUM",
            "FLOOR_ENERGY_EFF_NUM",
            "WINDOWS_ENERGY_EFF_NUM",
        ]
    ].mean(axis=1)

    return df


def compute_upgradability(df, verbose=False):
    """Compute upgradability for different categories, e.g. ROOF_UPGRADABILITY.
    A property is considered upgradable in a specific category if an upgrade in that category
    could be observed or if there is an EPC recommendation for this category.

    Args:
        df (pd.DataFrame): Dataframe with recommendations and energy efficiency differences over time.
        verbose (bool, optional): Whether to print summary about upgrades and recommendations.
        Defaults to False.

    Returns:
        df (pd.DataFrame): Dataframe with upgradability score features.
    """

    for cat in cats:

        total_props = df.shape[0]
        total_props_with_rec = df.loc[(df["{}_RECOMMENDATION".format(cat)])].shape[0]
        total_props_with_upgr = df.loc[(df["{}_EFF_DIFF".format(cat)] > 0)].shape[0]

        # Energy efficiency difference or recommendation --> upgradable
        props_w_upgr_and_rec = df.loc[
            (
                (df["{}_EFF_DIFF".format(cat)] > 0)
                & (df["{}_RECOMMENDATION".format(cat)])
            )
        ].shape[0]

        props_w_upgr_or_rec = df.loc[
            (
                (df["{}_EFF_DIFF".format(cat)] > 0)
                | (df["{}_RECOMMENDATION".format(cat)])
            )
        ].shape[0]

        df["{}_UPGRADABILITY".format(cat)] = (df["{}_EFF_DIFF".format(cat)] > 0.0) | (
            df["{}_RECOMMENDATION".format(cat)]
        )

        if verbose:
            print(cat)
            print(
                "Recommends:\t{:.2f}%".format(total_props_with_rec / total_props * 100)
            )
            print(
                "Upgrades:\t{:.2f}%".format(total_props_with_upgr / total_props * 100)
            )
            print(
                "Upgrade + Rec:\t{:.2f}%".format(
                    props_w_upgr_and_rec / total_props * 100
                )
            )
            print(
                "Upgrade | Rec:\t{:.2f}%".format(
                    props_w_upgr_or_rec / total_props * 100
                )
            )
            print(
                "Upgrade + Rec (of those with recommendation):\t{:.2f}%".format(
                    props_w_upgr_and_rec / total_props_with_rec * 100
                )
            )
            print("Coverage:\t{:.2f}%".format(props_w_upgr_or_rec / total_props * 100))
            print(
                "Upgr.ility:\t{:.2f}".format(df["{}_UPGRADABILITY".format(cat)].mean())
            )
            print()

    return df


def get_upgrade_features(df):
    """For properties with multiple records, get information about upgrades and upgradability score.

    Args:
        df (pd.DataFrame): Property features, including ENERGY_EFF and recommendations.

    Returns:
        upgrade_df (pd.DataFrame): Dataframe with additional upgrade features.
    """

    # Clean features (to be moved to ASF core data)
    df = extract_features_from_desc(df)

    # Only consider owner-occupied properties
    df = df.loc[df["TENURE"] == "owner-occupied"]

    # Get upgrade information based on properties with multiple entries
    latest_df = epc_data.filter_by_year(df, None, selection="latest entry")
    first_df = epc_data.filter_by_year(df, None, selection="first entry")

    upgrade_df = get_diff_in_energy_eff(first_df, latest_df, keep="first")
    upgrade_df = compute_upgradability(upgrade_df)

    return upgrade_df
