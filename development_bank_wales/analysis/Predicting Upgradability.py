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

from asf_core_data.utils.visualisation import kepler

from development_bank_wales import PROJECT_DIR, Path

from development_bank_wales.pipeline.feature_preparation import (
    recommendations,
    feature_engineering,
    data_aggregation,
)
from development_bank_wales.pipeline.predictive_model import (
    model_preparation,
    plotting,
    evaluation,
    training,
)

from keplergl import KeplerGl

import warnings

warnings.simplefilter(action="ignore")

# +
output_path = PROJECT_DIR / "outputs/data/wales_epc_with_recs.csv"
fig_output_path = PROJECT_DIR / "outputs/figures/"

if not Path(output_path).is_file():

    print("Loading and preparing the data...")

    wales_df = recommendations.load_epc_certs_and_recs(
        data_path="S3", subset="Wales", n_samples=None, remove_duplicates=False
    )

    wales_df.to_csv(output_path, index=False)

    print("Done!")

else:

    print("Loading the data...")
    wales_df = pd.read_csv(output_path)
    print("Done!")
# -

wales_df = feature_engineering.get_upgrade_features(wales_df)
wales_df.head()

# +
features_df = wales_df.copy()

label = "ROOF_UPGRADABILITY"
label_set = ["ROOF_UPGRADABILITY", "WALLS_UPGRADABILITY", "FLOOR_UPGRADABILITY"]

for label in label_set:

    processed_features, labels, feature_list = model_preparation.feature_prep_pipeline(
        wales_df, label
    )
    probas = training.train_and_evaluate_model(
        processed_features, labels, "Logistic Regression", label, feature_list
    )

    features_df["proba {}".format(label)] = probas
# -

features_df = data_aggregation.get_supplementary_data(features_df, data_path="S3")
data_per_group = data_aggregation.get_mean_per_group(features_df, label_set)
data_per_group.head()

# +
config = kepler.get_config("upgradability.txt", data_path=PROJECT_DIR)

upgradability_map = KeplerGl(height=500, config=config)

upgradability_map.add_data(
    data=data_per_group[
        [
            "proba ROOF_UPGRADABILITY",
            "hex_id",
        ]
    ],
    name="Roof",
)

upgradability_map.add_data(
    data=data_per_group[
        [
            "proba WALLS_UPGRADABILITY",
            "hex_id",
        ]
    ],
    name="Walls",
)

upgradability_map.add_data(
    data=data_per_group[
        [
            "weighted proba",
            "hex_id",
        ]
    ],
    name="Weighted combo upgradability",
)


upgradability_map.add_data(
    data=data_per_group[["hex_id", "IMD Decile (mean)"]], name="IMD"
)
upgradability_map.add_data(
    data=data_per_group[["hex_id", "# Properties"]], name="Density"
)


upgradability_map
# -

kepler.save_config(upgradability_map, "upgradability.txt", data_path=PROJECT_DIR)
kepler.save_map(upgradability_map, "Upgradability.html", data_path=PROJECT_DIR)

data_per_group = data_aggregation.get_mean_per_group(
    features_df, label_set, agglo_f="LOCAL_AUTHORITY_LABEL"
)

# +
config = kepler.get_config("LA_upgradability.txt", data_path=PROJECT_DIR)

upgradability_map = KeplerGl(height=500, config=config)

upgradability_map.add_data(
    data=data_per_group[
        ["proba ROOF_UPGRADABILITY", "hex_id", "LOCAL_AUTHORITY_LABEL"]
    ],
    name="Roof",
)

upgradability_map.add_data(
    data=data_per_group[
        ["proba WALLS_UPGRADABILITY", "hex_id", "LOCAL_AUTHORITY_LABEL"]
    ],
    name="Walls",
)

upgradability_map.add_data(
    data=data_per_group[["weighted proba", "hex_id", "LOCAL_AUTHORITY_LABEL"]],
    name="Weighted combo upgradability",
)


upgradability_map.add_data(
    data=data_per_group[["hex_id", "IMD Decile (mean)", "LOCAL_AUTHORITY_LABEL"]],
    name="IMD",
)
upgradability_map.add_data(
    data=data_per_group[["hex_id", "# Properties", "LOCAL_AUTHORITY_LABEL"]],
    name="Density",
)


upgradability_map
# -

kepler.save_config(upgradability_map, "LA_upgradability.txt", data_path=PROJECT_DIR)
kepler.save_map(upgradability_map, "LA_Upgradability.html", data_path=PROJECT_DIR)
