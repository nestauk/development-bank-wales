# File: development_bank_wales/pipeline/feature_preparation/legacy/upgrades.py
"""LEGACY: Legacy script for extracting features related to upgrades.
Only used in initial analysis notebook. No need to review in detail.
Note: sector and category are used interchangeably here."""

# ---------------------------------------------------------------------------------


import itertools
import pandas as pd
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt


from development_bank_wales import PROJECT_DIR, get_yaml_config, Path

# Load config file
config = get_yaml_config(PROJECT_DIR / "development_bank_wales/config/base.yaml")


eff_dict = {
    "MAINHEAT": ("MAINHEAT_ENERGY_EFF", "MAINHEAT_DESCRIPTION"),
    "FLOOR": ("FLOOR_ENERGY_EFF", "FLOOR_DESCRIPTION"),
    "WINDOWS": ("WINDOWS_ENERGY_EFF", "WINDOWS_DESCRIPTION"),
    "WALLS": ("WALLS_ENERGY_EFF", "WALLS_DESCRIPTION"),
    "ROOF": ("ROOF_ENERGY_EFF", "ROOF_DESCRIPTION"),
    "LIGHTING": ("LIGHTING_ENERGY_EFF", "LIGHTING_DESCRIPTION"),
    "HOT_WATER": ("HOT_WATER_ENERGY_EFF", "HOT_WATER_DESCRIPTION"),
}


sectors = ["WALLS", "ROOF", "MAINHEAT", "HOT_WATER", "LIGHTING", "FLOOR", "WINDOWS"]
quality_dict = {"Very Good": 5, "Good": 4, "Average": 3, "Poor": 2, "Very Poor": 1}


def get_upgrade_features(df1, df2, keep="first", identifier="UPRN", verbose=False):
    """Get features about upgrades/retrofits for each sector.
     Features include energy efficiency improvements, transitions,
     general sector upgrades, possible sector upgrades, mean upgradability scores,
     mean energy efficiency and energy efficiency difference and whether there were any upgrades.

     Only used in initial analysis.


    Args:
         df1 (pd.DataFrame): earlier EPC data
         df2 (pd.DataFrame): later EPC data
         keep (str): "first" or "latest", defaults to "first".
         identifier (str): property identifier, defaults to "UPRN".
         verbose (bool): print upgraded and upgradable scores, defaults to True.

     Returns:
        df (pd.DataFrame): EPC data with new features about upgrades.

    """

    for sector in sectors:

        if verbose:
            print(sector)

        # Get energy efficiency improvement for each sector
        eff, desc = eff_dict[sector]
        df1[eff + "_NUM"] = df1[eff].map(quality_dict)
        df2[eff + "_NUM"] = df2[eff].map(quality_dict)

        combo = pd.merge(
            df1[[desc, eff + "_NUM", "UPRN"]],
            df2[[desc, eff + "_NUM", "UPRN"]],
            on="UPRN",
        )
        combo[sector + "_EFF_DIFF"] = combo[eff + "_NUM_y"] - combo[eff + "_NUM_x"]
        combo[sector + "_EFF_DIFF"].fillna(0.0, inplace=True)
        combo.loc[combo[sector + "_EFF_DIFF"] < 0.0, sector + "_EFF_DIFF"] = 0.0

        # If there was an update, track transition
        combo["CHANGE_" + desc] = np.where(
            (
                (combo[desc + "_x"] != combo[desc + "_y"])
                & (combo[sector + "_EFF_DIFF"] > 0.0)
            ),
            combo[desc + "_x"] + " --> " + combo[desc + "_y"],
            "no upgrade",
        )

        # Keep first or latest records for further processing
        if keep == "first":
            df = pd.merge(
                df1,
                combo[[sector + "_EFF_DIFF", "CHANGE_" + desc, identifier]],
                on=identifier,
            )
        else:
            df = pd.merge(
                df2,
                combo[[sector + "_EFF_DIFF", "CHANGE_" + desc, identifier]],
                on=identifier,
            )

        # Boolean feature for sector upgrades
        df["UPGRADED_" + desc] = np.where(
            df["CHANGE_" + desc] != "no upgrade", True, False
        )

        if verbose:
            print(
                "Upgraded: {}%".format(
                    round(
                        df.loc[df["UPGRADED_" + desc]].shape[0] / df.shape[0] * 100, 2
                    )
                )
            )

        upgradables = df.loc[df["UPGRADED_" + desc]][desc].unique()

        if verbose:
            print(
                "Upgradable: {}%".format(
                    round(
                        df.loc[df[sector + "_DESCRIPTION"].isin(upgradables)].shape[0]
                        / df.shape[0]
                        * 100,
                        2,
                    )
                )
            )
            print()

        # Boolean feature indicating whether upgrade is possible
        df["UPGRADABLE_" + sector] = np.where(df[desc].isin(upgradables), True, False)

        # Get mean upgradability for sector
        mapping = dict(
            df.loc[df["UPGRADED_" + desc]].groupby(desc)[sector + "_EFF_DIFF"].mean()
        )

        df["UPGRADABILITY_" + sector] = df[desc].map(mapping)
        df["UPGRADABILITY_" + sector].fillna(0.0, inplace=True)
        df["UPGRADABILITY_" + sector] = df["UPGRADABILITY_" + sector].astype("float")

        # Restore original dataframe structure for processing next sector
        if keep == "first":
            df1 = df
        else:
            df2 = df

    # Keep first or latest records
    if keep == "first":
        df = df1
    else:
        df = df2

    # Mean upgradability for sectors
    df["UPGRADABILITY_TOTAL"] = df[
        [
            "UPGRADABILITY_ROOF",
            "UPGRADABILITY_WALLS",
            "UPGRADABILITY_HOT_WATER",
            "UPGRADABILITY_MAINHEAT",
            "UPGRADABILITY_LIGHTING",
            "UPGRADABILITY_FLOOR",
            "UPGRADABILITY_WINDOWS",
        ]
    ].mean(axis=1)

    # Mean energy efficiency difference
    df["TOTAL_EFF_DIFF"] = df[
        [
            "ROOF_EFF_DIFF",
            "WALLS_EFF_DIFF",
            "HOT_WATER_EFF_DIFF",
            "MAINHEAT_EFF_DIFF",
            "LIGHTING_EFF_DIFF",
            "FLOOR_EFF_DIFF",
            "WINDOWS_EFF_DIFF",
        ]
    ].mean(axis=1)

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

    # Are there any upgrades
    df["ANY_UPGRADES"] = df[
        [
            "UPGRADED_ROOF_DESCRIPTION",
            "UPGRADED_WALLS_DESCRIPTION",
            "UPGRADED_HOT_WATER_DESCRIPTION",
            "UPGRADED_MAINHEAT_DESCRIPTION",
            "UPGRADED_LIGHTING_DESCRIPTION",
            "UPGRADED_FLOOR_DESCRIPTION",
            "UPGRADED_WINDOWS_DESCRIPTION",
        ]
    ].max(axis=1)

    return df


def uprade_connections(df):
    """Generate a graph showing which sectors are frequently upgraded together.
    Only used in initial analysis.

    Args:
         df (pd.DataFrame): dataframe with upgrade information.

    Returns: None

    """

    # Create graph
    G = nx.Graph()

    plt.rcParams["figure.figsize"] = (7, 7)

    # For all sector combinations,
    # get percentage of upgraded properties for which both sectors had upgrades
    for sector_1, sector_2 in list(itertools.combinations(sectors, 2)):

        upgraded_properties = df.loc[df["ANY_UPGRADES"]]
        n_any_upgrades = upgraded_properties.shape[0]
        n_combo = (
            upgraded_properties.loc[
                upgraded_properties["UPGRADED_{}_DESCRIPTION".format(sector_1)]
                & upgraded_properties["UPGRADED_{}_DESCRIPTION".format(sector_2)]
            ].shape[0]
            / n_any_upgrades
            * 100
        )

        G.add_edge(sector_1, sector_2, weight=n_combo)

    # Create graph
    edge_weights = [G[u][v]["weight"] / 5 for u, v in G.edges()]
    nx.draw(
        G,
        pos=nx.circular_layout(G),
        node_size=5000,
        node_color="orange",
        width=edge_weights,
        with_labels=True,
    )

    edge_labels = dict(
        [((n1, n2), str(round(d["weight"])) + "%") for n1, n2, d in G.edges(data=True)]
    )

    nx.draw_networkx_edge_labels(
        G, pos=nx.circular_layout(G), edge_labels=edge_labels, font_color="red"
    )

    plt.tight_layout()
    plt.show()

    file_path = Path(PROJECT_DIR) / config["FIGURE_OUT"] / "Upgrade_connections.png"

    plt.savefig(file_path, format="PNG")


def get_sector_info_by_area(df, agglo_f, include_imd=True):
    """Add aggregated features (average scores) by area for each sector.
    Only used in intial anlysis.

    Args:
         df (pd.DataFrame): dataframe with upgrade information.
         agglo_f (str): feature for agglomeration, ideally hex_id or LOCAL_AUTHORTIY_LABEL.
         inclued_imd (bool): whether to include IMD data, defaults to True.

     Returns:
        df (pd.DataFrame): df with additional feature representing area averages.

    """

    sectors = [
        "ROOF",
        "WALLS",
        "MAINHEAT",
        "HOT_WATER",
        "LIGHTING",
        "FLOOR",
        "WINDOWS",
        "TOTAL",
    ]
    features = ["UPGRADABILITY_{}", "{}_EFF_DIFF", "{}_ENERGY_EFF_NUM"]

    # For each sector and feature, get mean values for given area.
    for sector in sectors:
        for feature in features:

            feature_name = feature.format(sector)
            mapping = dict(df.groupby(agglo_f)[feature_name].mean())
            df[feature_name + "_MEAN"] = (
                round(df[agglo_f].map(mapping), 2)
                .fillna(0.0)
                .apply(lambda x: x if x > 0 else 0)  # do not allow negative values
            )

    # Get mean IMD decile for given area
    if include_imd:
        mapping = dict(df.groupby(agglo_f)["IMD Decile"].mean())
        df["IMD Decile Hex"] = round(df[agglo_f].map(mapping), 2).fillna(0.0)

    return df
