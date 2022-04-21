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
from asf_core_data.utils.visualisation import easy_plotting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
from development_bank_wales import PROJECT_DIR
from development_bank_wales.pipeline import recommendations, upgrades, settings

from ipywidgets import interact

# %%
LOCAL_DATA_DIR = "/Users/juliasuter/Documents/ASF_data"

# %%
wales_df = recommendations.load_epc_certs_and_recs(
    data_path=LOCAL_DATA_DIR,
    subset="Wales",
    usecols=settings.EPC_FEAT_SELECTION,
    n_samples=None,
    remove_duplicates=False,
)

# %%
# wales_df = pd.read_csv("/Users/juliasuter/Documents/ASF_data/outputs/EPC/preprocessed_data/2021_Q4_0721/EPC_Wales_preprocessed.csv")

# %%
wales_df = wales_df.rename(columns={"HOTWATER_DESCRIPTION": "HOT_WATER_DESCRIPTION"})


# %%
print(len(wales_df["IMPROVEMENT_ID_TEXT"].unique()))

# %%
wales_df = recommendations.get_recommendation_features(wales_df)
wales_df["HAS_RECOM"].value_counts(dropna=False, normalize=True)

# %%
wales_df["IMPROVEMENT_ID_TEXT"].value_counts(dropna=False, normalize=True)

# %%
recs = [
    rec
    for rec in list(certificates_df["IMPROVEMENT_ID_TEXT"].unique()) + ["HAS_RECOM"]
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
print(latest_wales.shape)

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

for rec in list(certificates_df["IMPROVEMENT_ID_TEXT"].unique()):

    if rec == "nan" or isinstance(rec, float):
        continue
    first_wales = recommendations.check_for_implemented_rec(
        rec, first_wales, latest_wales, identifier="UPRN", keep="first"
    )
    print(first_wales[rec].value_counts(normalize=True))
    print()

# %%
first_wales = upgrades.get_upgrades(first_wales, latest_wales, keep="first")

# %%
from heat_pump_adoption_modelling.pipeline.supervised_model.utils import (
    data_agglomeration,
)

if "LATITUDE" in first_wales.columns:
    first_wales.drop(columns=["LATITUDE", "LONGITUDE"], inplace=True)

first_wales = feature_engineering.get_coordinates(first_wales)
first_wales = data_agglomeration.add_hex_id(first_wales, resolution=7.5)

hex_to_LA = data_agglomeration.map_hex_to_feature(first_wales, "LOCAL_AUTHORITY_LABEL")

# %%
from heat_pump_adoption_modelling.getters import deprivation_data

wales_imd = deprivation_data.get_country_imd_data(country="Wales")[
    ["Postcode", "IMD Decile"]
]
wales_imd.rename(columns={"Postcode": "POSTCODE"}, inplace=True)
wales_imd.head()

print(first_wales.shape)
print(wales_imd.shape)
first_wales = pd.merge(first_wales, wales_imd, on="POSTCODE")
first_wales.shape

# %%
sectors = [
    "ROOF",
    "WALLS",
    "MAINHEAT",
    "HOTWATER",
    "LIGHTING",
    "FLOOR",
    "WINDOWS",
    "TOTAL",
]

agglo_f = "LOCAL_AUTHORITY_LABEL"

for sector in sectors:

    mapping = dict(first_wales.groupby(agglo_f)["UPGRADABILITY_" + sector].mean())
    first_wales["UPGRADABILITY_" + sector + "_MEAN"] = round(
        first_wales["hex_id"].map(mapping)
    )
    first_wales["UPGRADABILITY_" + sector + "_MEAN"].fillna(0.0, inplace=True)

    mapping = dict(first_wales.groupby(agglo_f)[sector + "_DESCRIPTION_DIFF"].mean())
    first_wales[sector + "_DESCRIPTION_DIFF_MEAN"] = round(
        first_wales[agglo_f].map(mapping)
    )
    first_wales[sector + "_DESCRIPTION_DIFF_MEAN"].fillna(0.0, inplace=True)
    first_wales[sector + "_DESCRIPTION_DIFF_MEAN"] = first_wales[
        sector + "_DESCRIPTION_DIFF_MEAN"
    ].apply(lambda x: x if x > 0 else 0)

    mapping = dict(first_wales.groupby(agglo_f)[sector + "_ENERGY_EFF_NUM"].mean())
    first_wales[sector + "_ENERGY_EFF_NUM_MEAN"] = round(
        first_wales[agglo_f].map(mapping)
    )
    first_wales[sector + "_ENERGY_EFF_NUM_MEAN"].fillna(0.0, inplace=True)
    # first_wales[sector+'_ENERGY_EFF_NUM_MEAN'] = first_wales[sector+'_ENERGY_EFF_NUM_MEAN'].apply(lambda x : x if x > 0 else 0)


mapping = dict(first_wales.groupby(agglo_f)["IMD Decile"].mean())

first_wales["IMD Decile Hex"] = round(first_wales[agglo_f].map(mapping))
first_wales["IMD Decile Hex"].fillna(0.0, inplace=True)
first_wales.head()


# %%
print(first_wales.shape)
print(len(first_wales["hex_id"].unique()))
kepler_df = first_wales.drop_duplicates(
    subset="hex_id", keep="first", inplace=False, ignore_index=False
)
print(kepler_df.shape)

# %%
kepler_df["UPGRADABILITY_TOTAL_MEAN"].value_counts()

# %%
from keplergl import KeplerGl

config_file = kepler.KEPLER_PATH + "upgrades.txt"
config = kepler.get_config(config_file)

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

# upgrade_map.add_data(
#   data=first_wales[["UPGRADABILITY_WALLS_MEAN", "hex_id"]], name="Walls Upgradability")

# upgrade_map.add_data(
#  data=first_wales[["UPGRADABILITY_FLOOR_MEAN", "hex_id"]], name="Floor Upgradability")

# upgrade_map.add_data(
#    data=first_wales[["UPGRADABILITY_HOTWATER_MEAN", "hex_id"]], name="Hotwater Upgradability")

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


upgrade_map

# %%
kepler.save_config(upgrade_map, kepler.KEPLER_PATH + "eff.txt")

upgrade_map.save_to_html(file_name=kepler.KEPLER_PATH + "Efficiencies.html")

# %%
from heat_pump_adoption_modelling.pipeline.supervised_model.utils import kepler

kepler.save_config(upgrade_map, kepler.KEPLER_PATH + "upgradesxx.txt")

upgrade_map.save_to_html(file_name=kepler.KEPLER_PATH + "Upgrades.html")

# %%

# %%
from keplergl import KeplerGl

# config_file = kepler.KEPLER_PATH + "eff.txt"
# config = kepler.get_config(config_file)

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

# upgrade_map.add_data(
#   data=first_wales[["UPGRADABILITY_WALLS_MEAN", "hex_id"]], name="Walls Upgradability")

# upgrade_map.add_data(
#  data=first_wales[["UPGRADABILITY_FLOOR_MEAN", "hex_id"]], name="Floor Upgradability")

# upgrade_map.add_data(
#    data=first_wales[["UPGRADABILITY_HOTWATER_MEAN", "hex_id"]], name="Hotwater Upgradability")

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


upgrade_map

# %%
latest_wales.head(20)

# %%
from keplergl import KeplerGl

config_file = kepler.KEPLER_PATH + "upgrades.txt"
config = kepler.get_config(config_file)

upgrade_map = KeplerGl(height=500, config=config)

upgrade_map.add_data(
    data=kepler_df[["TOTAL_DESCRIPTION_DIFF_MEAN", "hex_id"]], name="Total Upgrades"
)

upgrade_map.add_data(
    data=kepler_df[["ROOF_DESCRIPTION_DIFF_MEAN", "hex_id"]], name="Roof Upgrades"
)

upgrade_map.add_data(
    data=kepler_df[["WINDOWS_DESCRIPTION_DIFF_MEAN", "hex_id"]], name="Windows Upgrades"
)

upgrade_map.add_data(
    data=first_wales[["WALLS_DESCRIPTION_DIFF_MEAN", "hex_id"]], name="Walls Upgrades"
)

# upgrade_map.add_data(
#  data=first_wales[["UPGRADABILITY_FLOOR_MEAN", "hex_id"]], name="Floor Upgradability")

# upgrade_map.add_data(
#    data=first_wales[["UPGRADABILITY_HOTWATER_MEAN", "hex_id"]], name="Hotwater Upgradability")

upgrade_map.add_data(
    data=kepler_df[["MAINHEAT_DESCRIPTION_DIFF_MEAN", "hex_id"]],
    name="Heating Upgrades",
)

upgrade_map.add_data(
    data=kepler_df[["LIGHTING_DESCRIPTION_DIFF_MEAN", "hex_id"]],
    name="Lighting Upgrades",
)

upgrade_map.add_data(
    data=kepler_df[["IMD Decile Hex", "hex_id"]], name="IMD Decile for Hex"
)


upgrade_map

# %%
kepler.save_config(upgrade_map, kepler.KEPLER_PATH + "ups.txt")

upgrade_map.save_to_html(file_name=kepler.KEPLER_PATH + "Upgrades.html")

# %%
wales_df.groupby(["UPRN"]).size().reset_index(name="count")["count"].value_counts()


# %%
pd.set_option("display.max_rows", 5000)

first_wales["CHANGE_LIGHTING_DESCRIPTION"].value_counts(normalize=True) * 100


# %%
@interact(
    sector=["ROOF", "WALLS", "MAINHEAT", "HOTWATER", "LIGHTING", "FLOOR", "WINDOW"]
)
def plotting(sector):
    plot_subcats_by_other_subcats(
        latest_wales,
        "UPGRADED_" + sector + "_DESCRIPTION",
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
    )


# %%
@interact(
    sector=["ROOF", "WALLS", "MAINHEAT", "HOTWATER", "LIGHTING", "FLOOR", "WINDOW"],
    feature=["BUILT_FORM", "CONSTRUCTION_AGE_BAND"],
)
def plotting(sector, feature):
    plot_subcats_by_other_subcats(
        latest_wales,
        "UPGRADED_" + sector + "_DESCRIPTION",
        feature,
        plotting_colors="inferno",
        x_tick_rotation=45,
        legend_loc="outside",
    )


# %%
import networkx as nx

G = nx.Graph()

for sector_1 in [
    "ROOF",
    "WALLS",
    "MAINHEAT",
    "HOTWATER",
    "LIGHTING",
    "FLOOR",
    "WINDOWS",
]:
    for sector_2 in [
        "ROOF",
        "WALLS",
        "MAINHEAT",
        "HOTWATER",
        "LIGHTING",
        "FLOOR",
        "WINDOWS",
    ]:

        if sector_1 == sector_2:
            continue
        n_combo = (
            latest_wales.loc[
                latest_wales["UPGRADED_" + sector_1 + "_DESCRIPTION"]
                & latest_wales["UPGRADED_" + sector_2 + "_DESCRIPTION"]
            ].shape[0]
            / latest_wales.loc[latest_wales["ANY_UPGRADES"]].shape[0]
            * 100
        )

        G.add_edge(sector_1, sector_2, weight=n_combo)

edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
nx.draw(
    G,
    pos=nx.circular_layout(G),
    node_size=5000,
    node_color="orange",
    width=edge_weights * 50,
    with_labels=True,
)

edge_labels = dict(
    [((n1, n2), str(round(d["weight"])) + "%") for n1, n2, d in G.edges(data=True)]
)


nx.draw_networkx_edge_labels(
    G, pos=nx.circular_layout(G), edge_labels=edge_labels, font_color="red"
)

plt.tight_layout()

plt.savefig("Graph.png", format="PNG")
plt.show()


# %%
# @interact(rec=list(certificates_df['IMPROVEMENT_ID_TEXT'].unique())+['HAS_RECOM'])
def plot_recommendations(rec, wales_df, first_wales, latest_wales):

    wales_df = wales_df.loc[wales_df["TENURE"] == "owner-occupied"]

    latest_wales = feature_engineering.filter_by_year(
        wales_df, "UPRN", None, selection="latest entry"
    )
    first_wales = feature_engineering.filter_by_year(
        wales_df, "UPRN", None, selection="first entry"
    )

    # first_wales.rename(columns={rec:rec+' at first'}, inplace=True)

    # latest_wales = pd.merge(latest_wales, first_wales[[rec, 'BUILDING_ID']], on='BUILDING_ID')

    # latest_wales['IMPLEMENTED_'+rec] = (latest_wales[rec+' at first'] & ~latest_wales[rec])

    # rec = 'Increase loft insulation to 270 mm'
    combo = pd.merge(first_wales[[rec, "UPRN"]], latest_wales[[rec, "UPRN"]], on="UPRN")
    combo["IMPLEMENTED_" + rec] = combo[rec + "_x"] & ~combo[rec + "_y"]
    first_wales = pd.merge(
        first_wales, combo[["IMPLEMENTED_" + rec, "UPRN"]], on="UPRN"
    )

    first_wales = first_wales.loc[first_wales[rec]]

    # print(first_wales['IMPLEMENTED_'+rec].value_counts())
    # print(first_wales['IMPLEMENTED_'+rec].value_counts(normalize=True))

    # plot_subcategory_distribution(
    #  first_wales,
    #'IMPLEMENTED_'+rec,
    # normalize=False,
    # color="lightseagreen",
    # plot_title=None)

    plot_subcats_by_other_subcats(
        first_wales,
        "PROPERTY_TYPE",
        "IMPLEMENTED_" + rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        feature_1_order=prop_type_order,
    )
    plot_subcats_by_other_subcats(
        first_wales,
        "BUILT_FORM",
        "IMPLEMENTED_" + rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        feature_1_order=built_form_order,
    )
    plot_subcats_by_other_subcats(
        first_wales,
        "CONSTRUCTION_AGE_BAND",
        "IMPLEMENTED_" + rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        feature_1_order=const_year_order_merged[1:],
    )
    plot_subcats_by_other_subcats(
        first_wales,
        "CURRENT_ENERGY_RATING",
        "IMPLEMENTED_" + rec,
        plotting_colors="copper",
        x_tick_rotation=45,
        feature_1_order=rating_order,
    )
    plot_subcats_by_other_subcats(
        first_wales,
        "LOCAL_AUTHORITY_LABEL",
        "IMPLEMENTED_" + rec,
        plotting_colors="copper",
        x_tick_rotation=45,
    )
    plot_subcats_by_other_subcats(
        first_wales,
        "TENURE",
        "IMPLEMENTED_" + rec,
        plotting_colors="copper",
        x_tick_rotation=45,
    )


recommendations = [
    "Increase loft insulation to 270 mm",
    "Cavity wall insulation",
    "Replace boiler with new condensing boiler",
    "Solid floor insulation",
    "Low energy lighting for all fixed outlets",
    "Upgrade heating controls",
    "Solar water heating",
    "Solar photovoltaic panels, 2.5 kWp",
]

plot_recommendations(recommendations[2], wales_df, first_wales, latest_wales)

# %%
quality_dict = {"Very Good": 5, "Good": 4, "Average": 3, "Poor": 2, "Very Poor": 1}

# %%
effs = [
    "MAINHEAT_ENERGY_EFF",
    # "SHEATING_ENERGY_EFF",
    "HOT_WATER_ENERGY_EFF",
    "FLOOR_ENERGY_EFF",
    "WINDOWS_ENERGY_EFF",
    "WALLS_ENERGY_EFF",
    "ROOF_ENERGY_EFF",
    #  "MAINHEATC_ENERGY_EFF",
    "LIGHTING_ENERGY_EFF",
]


latest_wales = feature_engineering.filter_by_year(
    owner_occs, "UPRN", None, selection="latest entry"
)
first_wales = feature_engineering.filter_by_year(
    owner_occs, "UPRN", None, selection="first entry"
)

for eff in effs:

    latest_wales[eff + "_NUM"] = latest_wales[eff].map(quality_dict)
    first_wales[eff + "_NUM"] = first_wales[eff].map(quality_dict)

    combo = pd.merge(
        first_wales[[eff + "_NUM", "UPRN"]],
        latest_wales[[eff + "_NUM", "UPRN"]],
        on="UPRN",
    )
    combo[eff + "_DIFF"] = combo[eff + "_NUM_y"] - combo[eff + "_NUM_x"]
    latest_wales = pd.merge(latest_wales, combo[[eff + "_DIFF", "UPRN"]], on="UPRN")
    latest_wales[eff + "_DIFF"].fillna(0.0)
    print(combo[eff + "_DIFF"].value_counts(normalize=True) * 100)


latest_wales["EFF_DIFFS"] = (
    latest_wales["MAINHEAT_ENERGY_EFF_DIFF"]
    + latest_wales["HOT_WATER_ENERGY_EFF_DIFF"]
    + latest_wales["FLOOR_ENERGY_EFF_DIFF"]
    + latest_wales["WINDOWS_ENERGY_EFF_DIFF"]
    + latest_wales["WALLS_ENERGY_EFF_DIFF"]
    + latest_wales["ROOF_ENERGY_EFF_DIFF"]
    + latest_wales["LIGHTING_ENERGY_EFF_DIFF"]
) / 7

latest_wales["EFF_AVG"] = (
    latest_wales["MAINHEAT_ENERGY_EFF_NUM"]
    + latest_wales["HOT_WATER_ENERGY_EFF_NUM"]
    + latest_wales["FLOOR_ENERGY_EFF_NUM"]
    + latest_wales["WINDOWS_ENERGY_EFF_NUM"]
    + latest_wales["WALLS_ENERGY_EFF_NUM"]
    + latest_wales["ROOF_ENERGY_EFF_NUM"]
    + latest_wales["LIGHTING_ENERGY_EFF_NUM"]
) / 7

latest_wales["EFF_AVG"].value_counts(normalize=False)


# %%
@interact(eff=effs)
def plot_improvements(eff):
    print(eff)

    latest_wales[eff + "_NUM"] = latest_wales[eff].map(quality_dict)
    first_wales[eff + "_NUM"] = first_wales[eff].map(quality_dict)

    owner_occs[eff + "_DIFF"] = latest_wales[eff + "_NUM"] - first_wales[eff + "_NUM"]

    print("dome")

    # plot_subcats_by_other_subcats(owner_occs, 'PROPERTY_TYPE', eff+'_DIFF',
    #                              plotting_colors='copper',x_tick_rotation=45,
    #                             feature_1_order=prop_type_order)
    # plot_subcats_by_other_subcats(owner_occs, 'BUILT_FORM',eff+'_DIFF',
    #                              plotting_colors='copper',x_tick_rotation=45,
    #                             feature_1_order=built_form_order)
    # plot_subcats_by_other_subcats(owner_occs, 'CONSTRUCTION_AGE_BAND',eff+'_DIFF',
    #                              plotting_colors='copper',x_tick_rotation=45,
    #                             feature_1_order=const_year_order_merged[1:])


# %%
latest_wales = feature_engineering.get_coordinates(latest_wales)
latest_wales = data_agglomeration.add_hex_id(latest_wales, resolution=7.5)

hex_to_LA = data_agglomeration.map_hex_to_feature(latest_wales, "LOCAL_AUTHORITY_LABEL")

# %%
