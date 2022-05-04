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

# %%
# %load_ext autoreload
# %autoreload 2

import asf_core_data
from asf_core_data.pipeline.preprocessing import (
    preprocess_epc_data,
    feature_engineering,
)
from asf_core_data.getters.epc import epc_data
from asf_core_data.getters.supplementary_data.deprivation import imd_data

from asf_core_data.utils.visualisation import easy_plotting, kepler
from asf_core_data.utils.geospatial import data_agglomeration

from development_bank_wales import PROJECT_DIR, Path
from development_bank_wales.pipeline import recommendations, upgrades, settings

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keplergl import KeplerGl

from ipywidgets import interact, fixed

# %% [markdown]
# ## Loading the data

# %%
LOCAL_DATA_DIR = "/Users/juliasuter/Documents/ASF_data"

# %%
wales_df = recommendations.load_epc_certs_and_recs(
    data_path=LOCAL_DATA_DIR, subset="Wales", n_samples=None, remove_duplicates=False
)

# %% [markdown]
# ## Recommendations
#
# There's 42 different recommendations in EPC Wales.

# %%
wales_df = recommendations.get_bool_recom_features(wales_df)

print(
    "Number of different recommendations:",
    len(wales_df["IMPROVEMENT_ID_TEXT"].unique()),
)
wales_df["HAS_RECOM"].value_counts(dropna=False, normalize=True)

# %%
wales_df["IMPROVEMENT_ID_TEXT"].value_counts(dropna=False, normalize=True)

# %%
recs = [
    rec
    for rec in list(wales_df["IMPROVEMENT_ID_TEXT"].unique()) + ["HAS_RECOM"]
    if type(rec) == str
]
out_path = PROJECT_DIR / "outputs/figures/"


@interact(rec=recs)
def plot_recommendations(rec):
    easy_plotting.plot_subcats_by_other_subcats(
        wales_df,
        "PROPERTY_TYPE",
        rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        fig_save_path=out_path,
    )
    easy_plotting.plot_subcats_by_other_subcats(
        wales_df,
        "BUILT_FORM",
        rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        fig_save_path=out_path,
    )
    easy_plotting.plot_subcats_by_other_subcats(
        wales_df,
        "CONSTRUCTION_AGE_BAND",
        rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        fig_save_path=out_path,
    )
    easy_plotting.plot_subcats_by_other_subcats(
        wales_df,
        "CURRENT_ENERGY_RATING",
        rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        fig_save_path=out_path,
    )
    easy_plotting.plot_subcats_by_other_subcats(
        wales_df,
        "LOCAL_AUTHORITY_LABEL",
        rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        fig_save_path=out_path,
    )
    easy_plotting.plot_subcats_by_other_subcats(
        wales_df,
        "TENURE",
        rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        fig_save_path=out_path,
    )


# %% [markdown]
# ## Recommendations over Time

# %%
latest_wales = feature_engineering.filter_by_year(
    wales_df.loc[wales_df["N_ENTRIES"].isin(["2", "3", "4", "5.0+"])],
    "UPRN",
    None,
    selection="latest entry",
)
first_wales = feature_engineering.filter_by_year(
    wales_df.loc[wales_df["N_ENTRIES"].isin(["2", "3", "4", "5.0+"])],
    "UPRN",
    None,
    selection="first entry",
)

# %%
rec = "Increase loft insulation to 270 mm"
latest_wales = recommendations.check_for_implemented_rec(
    rec, first_wales, latest_wales, identifier="UPRN"
)
latest_wales[rec].value_counts(normalize=True)

# %%
wales_df = wales_df.loc[wales_df["TENURE"] == "owner-occupied"]

latest_wales = feature_engineering.filter_by_year(
    wales_df, "UPRN", None, selection="latest entry"
)
first_wales = feature_engineering.filter_by_year(
    wales_df, "UPRN", None, selection="first entry"
)

for rec in list(wales_df["IMPROVEMENT_ID_TEXT"].unique()):

    if rec == "nan" or isinstance(rec, float):
        continue
    first_wales = recommendations.check_for_implemented_rec(
        rec, first_wales, latest_wales, identifier="UPRN", keep="first"
    )
    print(first_wales[rec].value_counts(normalize=True))
    print()

# %% [markdown]
# ## Upgrades and Upgradables

# %%
first_wales = upgrades.get_upgrade_features(first_wales, latest_wales, keep="first")


# %%
@interact(rec=recs, wales_df=fixed(wales_df))
def plot_recommendations(rec, wales_df):

    # Get first and last wales info
    wales_df = wales_df.loc[wales_df["TENURE"] == "owner-occupied"]
    latest_wales = feature_engineering.filter_by_year(
        wales_df, "UPRN", None, selection="latest entry"
    )
    first_wales = feature_engineering.filter_by_year(
        wales_df, "UPRN", None, selection="first entry"
    )

    first_wales = recommendations.check_for_implemented_rec(
        rec, first_wales, latest_wales, keep="first", identifier="UPRN"
    )

    # Only look at samples with this recommendation
    first_wales = first_wales.loc[first_wales[rec]]

    easy_plotting.plot_subcats_by_other_subcats(
        first_wales,
        "PROPERTY_TYPE",
        "IMPLEMENTED_" + rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        fig_save_path=out_path,
    )
    easy_plotting.plot_subcats_by_other_subcats(
        first_wales,
        "BUILT_FORM",
        "IMPLEMENTED_" + rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        fig_save_path=out_path,
    )
    easy_plotting.plot_subcats_by_other_subcats(
        first_wales,
        "CONSTRUCTION_AGE_BAND",
        "IMPLEMENTED_" + rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        fig_save_path=out_path,
    )
    easy_plotting.plot_subcats_by_other_subcats(
        first_wales,
        "CURRENT_ENERGY_RATING",
        "IMPLEMENTED_" + rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        fig_save_path=out_path,
    )
    easy_plotting.plot_subcats_by_other_subcats(
        first_wales,
        "LOCAL_AUTHORITY_LABEL",
        "IMPLEMENTED_" + rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        fig_save_path=out_path,
    )
    easy_plotting.plot_subcats_by_other_subcats(
        first_wales,
        "TENURE",
        "IMPLEMENTED_" + rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        fig_save_path=out_path,
    )


# %% [markdown]
# ## IMD Data

# %%
wales_imd = imd_data.get_country_imd_data(country="Wales", data_path=LOCAL_DATA_DIR)[
    ["Postcode", "IMD Decile"]
]
print(first_wales.shape)

first_wales = imd_data.merge_imd_with_other_set(
    wales_imd, first_wales, postcode_label="POSTCODE"
)
print(first_wales.shape)

# %% [markdown]
# ## Geospatial Info

# %%
if "LATITUDE" in first_wales.columns:
    first_wales.drop(columns=["LATITUDE", "LONGITUDE"], inplace=True)

first_wales = data_agglomeration.get_postcode_coordinates(
    first_wales, data_path=LOCAL_DATA_DIR
)
first_wales = data_agglomeration.add_hex_id(first_wales, resolution=7.5)
hex_to_LA = data_agglomeration.map_hex_to_feature(first_wales, "LOCAL_AUTHORITY_LABEL")


# %%
def get_agglomerated_kepler_data(df, agglo_f):

    agglomerated_wales_df = upgrades.get_sector_info_by_area(df, agglo_f)
    kepler_df = agglomerated_wales_df.drop_duplicates(
        subset="hex_id", keep="first", inplace=False, ignore_index=False
    )

    if agglo_f == "LOCAL_AUTHORITY_LABEL":
        kepler_df = pd.merge(hex_to_LA, kepler_df, on=["hex_id"])

    return kepler_df


# %% [markdown]
# ## Kepler Maps
#
# ### Upgradability

# %%
kepler_df = get_agglomerated_kepler_data(first_wales, "hex_id")

config = kepler.get_config("upgrades.txt", data_path=PROJECT_DIR)

upgrade_map = KeplerGl(height=500, config=config)

upgrade_map.add_data(
    data=kepler_df[["UPGRADABILITY_TOTAL_MEAN", "hex_id"]], name="Total Upgradability"
)

upgrade_map.add_data(
    data=kepler_df[["UPGRADABILITY_ROOF_MEAN", "hex_id"]], name="Roof Upgradability"
)

upgrade_map.add_data(
    data=kepler_df[["UPGRADABILITY_WINDOWS_MEAN", "hex_id"]],
    name="Windows Upgradability",
)

upgrade_map.add_data(
    data=kepler_df[["UPGRADABILITY_MAINHEAT_MEAN", "hex_id"]],
    name="Heating Upgradability",
)

upgrade_map.add_data(
    data=kepler_df[["UPGRADABILITY_LIGHTING_MEAN", "hex_id"]],
    name="Lighting Upgradability",
)

upgrade_map.add_data(
    data=kepler_df[["IMD Decile Hex", "hex_id"]], name="IMD Decile for Hex"
)

# upgrade_map.add_data(
#   data=first_wales[["UPGRADABILITY_WALLS_MEAN", "hex_id"]], name="Walls Upgradability")

# upgrade_map.add_data(
#  data=first_wales[["UPGRADABILITY_FLOOR_MEAN", "hex_id"]], name="Floor Upgradability")

# upgrade_map.add_data(
#    data=first_wales[["UPGRADABILITY_HOTWATER_MEAN", "hex_id"]], name="Hotwater Upgradability")


upgrade_map

# %%
kepler.save_config(upgrade_map, "upgrades.txt", data_path=PROJECT_DIR)
kepler.save_map(upgrade_map, "Upgrades.html", data_path=PROJECT_DIR)

# %% [markdown]
# ### Upgradability by Local Authority

# %%
kepler_df = get_agglomerated_kepler_data(first_wales, "LOCAL_AUTHORITY_LABEL")

config = kepler.get_config("LA_upgrades.txt", data_path=PROJECT_DIR)

upgrade_map = KeplerGl(height=500, config=config)

upgrade_map.add_data(
    data=kepler_df[
        ["UPGRADABILITY_TOTAL_MEAN", "MOST_FREQUENT_LOCAL_AUTHORITY_LABEL", "hex_id"]
    ],
    name="Total Upgradability",
)

upgrade_map.add_data(
    data=kepler_df[
        ["UPGRADABILITY_ROOF_MEAN", "MOST_FREQUENT_LOCAL_AUTHORITY_LABEL", "hex_id"]
    ],
    name="Roof Upgradability",
)

upgrade_map.add_data(
    data=kepler_df[
        ["UPGRADABILITY_WINDOWS_MEAN", "MOST_FREQUENT_LOCAL_AUTHORITY_LABEL", "hex_id"]
    ],
    name="Windows Upgradability",
)

upgrade_map.add_data(
    data=kepler_df[
        ["UPGRADABILITY_MAINHEAT_MEAN", "MOST_FREQUENT_LOCAL_AUTHORITY_LABEL", "hex_id"]
    ],
    name="Heating Upgradability",
)

upgrade_map.add_data(
    data=kepler_df[
        ["UPGRADABILITY_LIGHTING_MEAN", "MOST_FREQUENT_LOCAL_AUTHORITY_LABEL", "hex_id"]
    ],
    name="Lighting Upgradability",
)

upgrade_map.add_data(
    data=kepler_df[["IMD Decile Hex", "MOST_FREQUENT_LOCAL_AUTHORITY_LABEL", "hex_id"]],
    name="IMD Decile for Hex",
)

# upgrade_map.add_data(
#   data=first_wales[["UPGRADABILITY_WALLS_MEAN", "hex_id"]], name="Walls Upgradability")

# upgrade_map.add_data(
#  data=first_wales[["UPGRADABILITY_FLOOR_MEAN", "hex_id"]], name="Floor Upgradability")

# upgrade_map.add_data(
#    data=first_wales[["UPGRADABILITY_HOTWATER_MEAN", "hex_id"]], name="Hotwater Upgradability")


upgrade_map

# %%
kepler.save_config(upgrade_map, "LA_upgrades.txt", data_path=PROJECT_DIR)
kepler.save_map(upgrade_map, "LA_Upgrades.html", data_path=PROJECT_DIR)

# %% [markdown]
# ## Energy Efficiency

# %%
kepler_df = get_agglomerated_kepler_data(first_wales, "hex_id")

config = kepler.get_config("efficiencies.txt", data_path=PROJECT_DIR)

upgrade_map = KeplerGl(height=500, config=config)

upgrade_map.add_data(
    data=kepler_df[["TOTAL_ENERGY_EFF_NUM_MEAN", "hex_id"]],
    name="Total Energy Efficiency",
)

upgrade_map.add_data(
    data=kepler_df[["ROOF_ENERGY_EFF_NUM_MEAN", "hex_id"]],
    name="Roof Energy Efficiency",
)

upgrade_map.add_data(
    data=kepler_df[["WINDOWS_ENERGY_EFF_NUM_MEAN", "hex_id"]],
    name="Windows Energy Efficiency",
)

upgrade_map.add_data(
    data=kepler_df[["MAINHEAT_ENERGY_EFF_NUM_MEAN", "hex_id"]],
    name="Heating Energy Efficiency",
)

upgrade_map.add_data(
    data=kepler_df[["LIGHTING_ENERGY_EFF_NUM_MEAN", "hex_id"]],
    name="Lighting Energy Efficiency",
)

upgrade_map.add_data(
    data=kepler_df[["IMD Decile Hex", "hex_id"]], name="IMD Decile for Hex"
)

# upgrade_map.add_data(
#   data=first_wales[["UPGRADABILITY_WALLS_MEAN", "hex_id"]], name="Walls Upgradability")

# upgrade_map.add_data(
#  data=first_wales[["UPGRADABILITY_FLOOR_MEAN", "hex_id"]], name="Floor Upgradability")

# upgrade_map.add_data(
#    data=first_wales[["UPGRADABILITY_HOTWATER_MEAN", "hex_id"]], name="Hotwater Upgradability")


upgrade_map

# %%
kepler.save_config(upgrade_map, "efficiencies.txt", data_path=PROJECT_DIR)
kepler.save_map(upgrade_map, "Efficiencies.html", data_path=PROJECT_DIR)

# %% [markdown]
# ## Rating Diff

# %%
kepler_df = get_agglomerated_kepler_data(first_wales, "hex_id")

config = kepler.get_config("diffs.txt", data_path=PROJECT_DIR)

upgrade_map = KeplerGl(height=500, config=config)

upgrade_map.add_data(
    data=kepler_df[["TOTAL_EFF_DIFF_MEAN", "hex_id"]], name="Total Upgrades"
)

upgrade_map.add_data(
    data=kepler_df[["ROOF_EFF_DIFF_MEAN", "hex_id"]], name="Roof Upgrades"
)

upgrade_map.add_data(
    data=kepler_df[["WINDOWS_EFF_DIFF_MEAN", "hex_id"]], name="Windows Upgrades"
)

upgrade_map.add_data(
    data=first_wales[["WALLS_EFF_DIFF_MEAN", "hex_id"]], name="Walls Upgrades"
)

upgrade_map.add_data(
    data=kepler_df[["MAINHEAT_EFF_DIFF_MEAN", "hex_id"]], name="Heating Upgrades"
)

upgrade_map.add_data(
    data=kepler_df[["LIGHTING_EFF_DIFF_MEAN", "hex_id"]], name="Lighting Upgrades"
)

upgrade_map.add_data(
    data=kepler_df[["IMD Decile Hex", "hex_id"]], name="IMD Decile for Hex"
)

# upgrade_map.add_data(
#  data=first_wales[["UPGRADABILITY_FLOOR_MEAN", "hex_id"]], name="Floor Upgradability")

# upgrade_map.add_data(
#    data=first_wales[["UPGRADABILITY_HOTWATER_MEAN", "hex_id"]], name="Hotwater Upgradability")


upgrade_map

# %%
kepler.save_config(upgrade_map, "diffs.txt", data_path=PROJECT_DIR)
kepler.save_map(upgrade_map, "Differences.html", data_path=PROJECT_DIR)

# %% [markdown]
# ## Transitions

# %%
latest_wales = feature_engineering.filter_by_year(
    wales_df, "UPRN", None, selection="latest entry"
)
first_wales = feature_engineering.filter_by_year(
    wales_df, "UPRN", None, selection="first entry"
)

latest_wales = upgrades.get_upgrade_features(
    first_wales, latest_wales, keep="latest", verbose=False
)

# %%
pd.set_option("display.max_rows", 100)
latest_wales["CHANGE_LIGHTING_DESCRIPTION"].value_counts(normalize=True) * 100


# %%
@interact(
    sector=["ROOF", "WALLS", "MAINHEAT", "HOT_WATER", "LIGHTING", "FLOOR", "WINDOW"]
)
def plotting(sector):
    easy_plotting.plot_subcats_by_other_subcats(
        latest_wales,
        "UPGRADED_{}_DESCRIPTION".format(sector),
        "TRANSACTION_TYPE",
        plotting_colors="inferno",
        x_tick_rotation=45,
        feature_2_order=[
            "ECO assessment",
            "marketed sale",
            "FiT application",
            "non marketed sale",
            "unknown",
            "assessment for green deal",
            "RHI application",
            "new dwelling",
        ],
        legend_loc="outside",
        fig_save_path=out_path,
    )


# %%
@interact(
    sector=["ROOF", "WALLS", "MAINHEAT", "HOT_WATER", "LIGHTING", "FLOOR", "WINDOW"],
    feature=["BUILT_FORM", "CONSTRUCTION_AGE_BAND"],
)
def plotting(sector, feature):
    easy_plotting.plot_subcats_by_other_subcats(
        latest_wales,
        "UPGRADED_" + sector + "_DESCRIPTION",
        feature,
        plotting_colors="inferno",
        x_tick_rotation=45,
        legend_loc="outside",
        fig_save_path=out_path,
    )


# %%
upgrades.uprade_connections(latest_wales)

# %%
