import pandas as pd


from asf_core_data.getters.supplementary_data.deprivation import imd_data
from asf_core_data.getters.supplementary_data.geospatial import coordinates
from asf_core_data.utils.geospatial import data_agglomeration


def get_supplementary_data(df, data_path):

    coordinates_df = coordinates.get_postcode_coordinates(data_path=data_path)
    feat_map_df = pd.merge(df, coordinates_df, on="POSTCODE", how="left")
    feat_map_df = data_agglomeration.add_hex_id(feat_map_df, resolution=6)

    imd_df = imd_data.get_gb_imd_data(data_path=data_path)
    wales_imd = imd_df.loc[imd_df["Country"] == "Wales"]

    feat_map_df = imd_data.merge_imd_with_other_set(
        wales_imd, feat_map_df, postcode_label="POSTCODE"
    )

    return feat_map_df


def get_proba_per_hex(features_df, label_set):

    weight_dict = {
        "ROOF_UPGRADABILITY": 0.35,
        "WALLS_UPGRADABILITY": 0.5,
        "FLOOR_UPGRADABILITY": 0.15,
    }

    hex_probas = (
        features_df.groupby(["hex_id"])[["LMK_KEY"]]
        .count()
        .reset_index()
        .rename(columns={"LMK_KEY": "# Properties"})
    )

    hex_probas["IMD Decile (mean)"] = (
        features_df.groupby(["hex_id"])[["IMD Decile"]]
        .mean()
        .reset_index()["IMD Decile"]
    )

    hex_probas["IMD Decile (mean)"] = hex_probas["IMD Decile (mean)"].round(0)

    print(hex_probas.shape)

    for label in label_set:

        proba_label = "proba {}".format(label)
        hex_probas[proba_label] = (
            features_df.groupby(["hex_id"])[[proba_label]]
            .mean()
            .reset_index()[proba_label]
        )

    hex_probas["weighted proba"] = sum(
        [
            hex_probas["proba {}".format(label)] * weight_dict[label]
            for label in label_set
        ]
    ) / len(label_set)

    return hex_probas
