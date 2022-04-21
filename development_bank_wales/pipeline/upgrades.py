import pandas as pd
import numpy as np

quality_dict = {"Very Good": 5, "Good": 4, "Average": 3, "Poor": 2, "Very Poor": 1}


eff_dict = {
    "MAINHEAT": ("MAINHEAT_ENERGY_EFF", "MAINHEAT_DESCRIPTION"),
    "WALLS": ("WALLS_ENERGY_EFF", "WALLS_DESCRIPTION"),
    "FLOOR": ("FLOOR_ENERGY_EFF", "FLOOR_DESCRIPTION"),
    "WINDOWS": ("WINDOWS_ENERGY_EFF", "WINDOWS_DESCRIPTION"),
    "WALLS": ("WALLS_ENERGY_EFF", "WALLS_DESCRIPTION"),
    "ROOF": ("ROOF_ENERGY_EFF", "ROOF_DESCRIPTION"),
    "LIGHTING": ("LIGHTING_ENERGY_EFF", "LIGHTING_DESCRIPTION"),
    "HOT_WATER": ("HOT_WATER_ENERGY_EFF", "HOT_WATER_DESCRIPTION"),
}


sectors = ["WALLS", "ROOF", "MAINHEAT", "HOT_WATER", "LIGHTING", "FLOOR", "WINDOWS"]


def get_upgrades(df1, df2, keep="first"):

    for sector in sectors:

        print(sector)

        eff, desc = eff_dict[sector]
        df1[eff + "_NUM"] = df1[eff].map(quality_dict)
        df2[eff + "_NUM"] = df2[eff].map(quality_dict)

        combo = pd.merge(
            df1[[desc, eff + "_NUM", "UPRN"]],
            df2[[desc, eff + "_NUM", "UPRN"]],
            on="UPRN",
        )
        combo[desc + "_DIFF"] = combo[eff + "_NUM_y"] - combo[eff + "_NUM_x"]
        combo[desc + "_DIFF"].fillna(0.0, inplace=True)

        combo["CHANGE_" + desc] = np.where(
            (
                (combo[desc + "_x"] != combo[desc + "_y"])
                & (combo[desc + "_DIFF"] > 0.0)
            ),
            combo[desc + "_x"] + " --> " + combo[desc + "_y"],
            "no upgrade",
        )

        if keep == "first":
            df = pd.merge(
                df1, combo[[desc + "_DIFF", "CHANGE_" + desc, "UPRN"]], on="UPRN"
            )
        else:
            df = pd.merge(
                df2, combo[[desc + "_DIFF", "CHANGE_" + desc, "UPRN"]], on="UPRN"
            )

        df["UPGRADED_" + desc] = np.where(
            df["CHANGE_" + desc] != "no upgrade", True, False
        )

        print(
            "Upgraded: {}%".format(
                round(df.loc[df["UPGRADED_" + desc]].shape[0] / df.shape[0] * 100, 2)
            )
        )

        upgradables = df.loc[df["UPGRADED_" + desc]][desc].unique()

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

        df["UPGRADABLE_" + sector] = np.where(df[desc].isin(upgradables), True, False)

        mapping = dict(
            df.loc[df["UPGRADED_" + desc]].groupby(desc)[desc + "_DIFF"].mean()
        )

        df["UPGRADABILITY_" + sector] = df[desc].map(mapping)
        df["UPGRADABILITY_" + sector].fillna(0.0, inplace=True)
        df["UPGRADABILITY_" + sector] = df["UPGRADABILITY_" + sector].astype("float")

        if keep == "first":
            df1 = df
        else:
            df2 = df

    if keep == "first":
        df = df1
    else:
        df = df2
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

    df["TOTAL_DESCRIPTION_DIFF"] = df[
        [
            "ROOF_DESCRIPTION_DIFF",
            "WALLS_DESCRIPTION_DIFF",
            "HOT_WATER_DESCRIPTION_DIFF",
            "MAINHEAT_DESCRIPTION_DIFF",
            "LIGHTING_DESCRIPTION_DIFF",
            "FLOOR_DESCRIPTION_DIFF",
            "WINDOWS_DESCRIPTION_DIFF",
        ]
    ].mean(axis=1)

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
    ].min(axis=1)

    return df
