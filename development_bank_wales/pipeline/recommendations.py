import numpy as np
import pandas as pd

from asf_core_data.getters.epc import epc_data
from asf_core_data.pipeline.preprocessing import preprocess_epc_data


def get_recommendation_features(df):

    all_keys = []

    for rec in df["IMPROVEMENT_ID_TEXT"].unique():
        rec_keys = df.loc[df["IMPROVEMENT_ID_TEXT"] == rec]["LMK_KEY"].unique()
        all_keys += list(rec_keys)
        df[rec] = np.where(df["LMK_KEY"].isin(rec_keys), True, False)

    df["HAS_RECOM"] = np.where(df["LMK_KEY"].isin(set(all_keys)), True, False)
    return df


def load_epc_certs_and_recs(
    data_path, subset="GB", usecols=None, n_samples=None, remove_duplicates=True
):

    df = preprocess_epc_data.load_and_preprocess_epc_data(
        data_path=data_path,
        subset=subset,
        usecols=usecols,
        n_samples=None,
        remove_duplicates=False,
    )

    recommendations = epc_data.load_wales_recommendations(
        data_path=data_path, subset="Wales"
    )

    merged = pd.merge(
        df, recommendations.drop(columns=["COUNTRY"]), on=["LMK_KEY"], how="inner"
    )

    merged = merged.rename(columns={"HOTWATER_DESCRIPTION": "HOT_WATER_DESCRIPTION"})

    return merged


def check_for_implemented_rec(rec, df1, df2, keep="first", identifier="UPRN"):

    combo = pd.merge(df1[[rec, "UPRN"]], df2[[rec, "UPRN"]], on="UPRN")
    combo["IMPLEMENTED_" + rec] = combo[rec + "_x"] & ~combo[rec + "_y"]

    if keep == "first":
        df = pd.merge(df1, combo[["IMPLEMENTED_" + rec, "UPRN"]], on="UPRN")

    else:
        df = pd.merge(df2, combo[["IMPLEMENTED_" + rec, "UPRN"]], on="UPRN")
    return df
