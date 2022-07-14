# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: development_bank_wales
#     language: python
#     name: development_bank_wales
# ---

# +
# %load_ext autoreload
# %autoreload 2

import pandas as pd

from development_bank_wales import PROJECT_DIR, Path

from development_bank_wales.pipeline.feature_preparation import (
    recommendations,
    upgrades,
    feature_engineering,
    data_aggregation,
)
from development_bank_wales.pipeline.machine_learning import (
    model_preparation,
    plotting,
    evaluation,
    machine_learning,
)

from keplergl import KeplerGl

import warnings

warnings.simplefilter(action="ignore")

# +
LOCAL_DATA_DIR = "/Users/juliasuter/Documents/ASF_data"

wales_df = recommendations.load_epc_certs_and_recs(
    data_path=LOCAL_DATA_DIR, subset="Wales", n_samples=None, remove_duplicates=False
)
# -

wales_df = pd.read_csv(
    "/Users/juliasuter/Documents/ASF_data/outputs/EPC/preprocessed_data/2021_Q4_0721/EPC_Wales_preprocessed.csv"
)

wales_df = feature_engineering.preprocess_features(wales_df)
wales_df.head()

# +
features_df = wales_df.copy()

label = "ROOF_UPGRADABILITY"
label_set = ["ROOF_UPGRADABILITY", "WALLS_UPGRADABILITY", "FLOOR_UPGRADABILITY"]

for label in label_set:

    processed_features, labels, feature_list = model_preparation.feature_preparation(
        wales_df, label
    )
    probas = machine_learning.train_and_evaluate_model(
        processed_features, labels, "Logistic Regression", label, feature_list
    )

    features_df["proba {}".format(label)] = probas
# -

features_df = data_aggregation.get_supplementary_data(features_df, LOCAL_DATA_DIR)
hex_probas = data_aggregation.get_proba_per_hex(features_df, label_set)
hex_probas.head()

# +
from keplergl import KeplerGl
import yaml

from asf_core_data.utils.visualisation import kepler

config = kepler.get_config("upgradability.txt", data_path="../../")

upgradability_map = KeplerGl(height=500, config=config)

upgradability_map.add_data(
    data=hex_probas[
        [
            "proba ROOF_UPGRADABILITY",
            "hex_id",
        ]
    ],
    name="Roof",
)

upgradability_map.add_data(
    data=hex_probas[
        [
            "proba WALLS_UPGRADABILITY",
            "hex_id",
        ]
    ],
    name="Walls",
)

# upgradability_map.add_data(
#    data=feat_map_df[["proba FLOOR_UPGRADABILITY", "hex_id",]], name="Floor")

upgradability_map.add_data(
    data=hex_probas[
        [
            "weighted proba",
            "hex_id",
        ]
    ],
    name="Combo",
)


upgradability_map.add_data(data=hex_probas[["hex_id", "IMD Decile (mean)"]], name="IMD")
upgradability_map.add_data(data=hex_probas[["hex_id", "# Properties"]], name="Density")


upgradability_map
# -

kepler.save_config(upgradability_map, "upgradability.txt", data_path="../../")
kepler.save_map(upgradability_map, "Upgradability.html", data_path="../../")
