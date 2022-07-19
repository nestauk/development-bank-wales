# File: development_bank_wales/pipeline/feature_preparation/data_aggregregation.py
"""Aggregate data on hex or local authority level."""

# ---------------------------------------------------------------------------------

import pandas as pd

from asf_core_data.getters.supplementary_data.deprivation import imd_data
from asf_core_data.getters.supplementary_data.geospatial import coordinates
from asf_core_data.utils.geospatial import data_agglomeration

from development_bank_wales.pipeline.feature_preparation.legacy import upgrades

# ---------------------------------------------------------------------------------


def get_supplementary_data(df, data_path="S3"):
    """Get coordinates and IMD data. Geographical data is required for aggregation.

    Args:
        df (pd.DataFrame): Dataframe that supplementary data is added to.
        data_path (str, optional): Data source: local dir or S3. Defaults to "S3".

    Returns:
        df_suppl: Dataframe with added supplementary data.
    """

    # Get coordinates and hex ids
    coordinates_df = coordinates.get_postcode_coordinates(data_path=data_path)
    df_suppl = pd.merge(df, coordinates_df, on="POSTCODE", how="left")
    df_suppl = data_agglomeration.add_hex_id(df_suppl, resolution=6)

    # Get Wales IMD data
    wales_imd = imd_data.get_country_imd_data("Wales", data_path=data_path)[
        ["Postcode", "IMD Decile"]
    ]

    df_suppl = imd_data.merge_imd_with_other_set(
        wales_imd, df_suppl, postcode_label="POSTCODE"
    )

    return df_suppl


def get_mean_per_group(features_df, label_set, agglo_f="hex_id"):
    """Get mean upgrade probability for each group, e.g. hex id or local authority label.
    Furthermore, compute the IMD mean and number of properties per group.

    Args:
        features_df (pd.DataFrame): _description_
        label_set (list): Label list, e.g. upgrade categories ROOF, WALLS or FLOOR.
        agglo_f (str, optional): Feature by which to group. Defaults to "hex_id".

    Returns:
        group_probas (pd.DataFrame): upgrade probabilities per group.
    """

    weight_dict = {
        "ROOF_UPGRADABILITY": 0.35,
        "WALLS_UPGRADABILITY": 0.5,
        "FLOOR_UPGRADABILITY": 0.15,
    }

    group_probas = (
        features_df.groupby([agglo_f])[["LMK_KEY"]]
        .count()
        .reset_index()
        .rename(columns={"LMK_KEY": "# Properties"})
    )

    group_probas["IMD Decile (mean)"] = (
        features_df.groupby([agglo_f])[["IMD Decile"]]
        .mean()
        .reset_index()["IMD Decile"]
    )

    group_probas["IMD Decile (mean)"] = group_probas["IMD Decile (mean)"].round(0)

    for label in label_set:

        proba_label = "proba {}".format(label)
        group_probas[proba_label] = (
            features_df.groupby([agglo_f])[[proba_label]]
            .mean()
            .reset_index()[proba_label]
        )

    group_probas["weighted proba"] = sum(
        [
            group_probas["proba {}".format(label)] * weight_dict[label]
            for label in label_set
        ]
    ) / len(label_set)

    # Assign the most frequent local authority label to each hex.
    if agglo_f == "LOCAL_AUTHORITY_LABEL":
        hex_to_LA = data_agglomeration.map_hex_to_feature(
            features_df, "LOCAL_AUTHORITY_LABEL"
        ).rename(
            columns={"MOST_FREQUENT_LOCAL_AUTHORITY_LABEL": "LOCAL_AUTHORITY_LABEL"}
        )

        print(hex_to_LA.columns)
        print(group_probas.columns)
        group_probas = pd.merge(hex_to_LA, group_probas, on=["LOCAL_AUTHORITY_LABEL"])

    return group_probas


def get_aggregated_upgrade_data(df, agglo_f="hex_id"):
    """Aggregate upgrade data by grouping by agglo_f, e.g. hex id or local authority.
    Get the mean upgradability, energy efficiency and difference in energy efficiency.
    Only keep one sample per hex (as they all hold the same information after grouping).

    Args:
        df (pd.DataFrame): Dataframe including information about upgrades, energy efficiency etc. for each property.
        agglo_f (str, optional): Feature by which to group. Defaults to "hex_id".

    Returns:
        aggregated_df: Data aggregated by hex id or local authority.
    """

    hex_to_LA = data_agglomeration.map_hex_to_feature(df, "LOCAL_AUTHORITY_LABEL")

    # Compute means
    aggregated_df = upgrades.get_sector_info_by_area(df, agglo_f)

    # Only keep one entry per hex id to keep df small (easier to load maps from it)
    aggregated_df = aggregated_df.drop_duplicates(
        subset="hex_id", keep="first", inplace=False, ignore_index=False
    )

    # Assign the most frequent local authority label to each hex.
    if agglo_f == "LOCAL_AUTHORITY_LABEL":
        aggregated_df = pd.merge(hex_to_LA, aggregated_df, on=["hex_id"])

    return aggregated_df
