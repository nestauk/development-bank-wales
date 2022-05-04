import numpy as np
import pandas as pd

from asf_core_data.getters.epc import epc_data
from asf_core_data.pipeline.preprocessing import preprocess_epc_data

from development_bank_wales import PROJECT_DIR, get_yaml_config


# Load config file
config = get_yaml_config(PROJECT_DIR / "development_bank_wales/config/base.yaml")


def get_bool_recom_features(df):

    """Get boolean recommendation features.
    New features indicate whether property received recommendations or not.

    Args:
        df (pd.DataFrame): needs to include "IMPROVEMENT_ID_TEXT" and "LMK_KEY".

    Returns:
        df (pd.DataFrame): updated df with boolean recommendation features.

    """

    all_keys = []

    # For each recommendation,
    # create bool feature indicating whether recommendation is suggested for property
    for rec in df["IMPROVEMENT_ID_TEXT"].unique():
        rec_keys = df.loc[df["IMPROVEMENT_ID_TEXT"] == rec]["LMK_KEY"].unique()
        all_keys += list(rec_keys)
        df[rec] = np.where(df["LMK_KEY"].isin(rec_keys), True, False)

    # Check whether property has any recommendation
    df["HAS_RECOM"] = np.where(df["LMK_KEY"].isin(set(all_keys)), True, False)
    return df


def load_epc_certs_and_recs(
    data_path,
    subset="GB",
    usecols=config["EPC_FEAT_SELECTION"],
    n_samples=None,
    remove_duplicates=True,
):
    """Load EPC records and recommendations and merge into one dataframe.

    Args:
        data_path (str/Path): path to EPC data.
        subset (str): 'England', 'Scotland', 'Wales' or 'GB', defaults to 'GB'.
        usecols (list): columns to use, default to selection defined in base config.
        n_samples (int): number of samples to use, defaults to None.
        remove_duplicates (bool): whether to remove duplicates, defauls to True.

    Returns:
        epc_rec_df (pd.DataFrame): EPC records and recommendation data

    """

    df = preprocess_epc_data.load_and_preprocess_epc_data(
        data_path=data_path,
        subset=subset,
        usecols=usecols,
        n_samples=n_samples,
        remove_duplicates=remove_duplicates,
    )

    recommendations = epc_data.load_wales_recommendations(
        data_path=data_path, subset="Wales"
    )

    epc_rec_df = pd.merge(
        df, recommendations.drop(columns=["COUNTRY"]), on=["LMK_KEY"], how="inner"
    )

    # Rename feature for consistent processing later
    epc_rec_df = epc_rec_df.rename(
        columns={"HOTWATER_DESCRIPTION": "HOT_WATER_DESCRIPTION"}
    )

    return epc_rec_df


def check_for_implemented_rec(rec, df1, df2, keep="first", identifier="UPRN"):
    """Check for implemented recommendations for two dataframes,
    representing same properties over time.

    Args:
        rec (str): recommendation
        df1 (pd.DataFrame): earlier EPC data
        df2 (pd.DataFrame): later EPC data
        keep (str): "first" or "latest", defaults to "first".
        identifier (str): property identifier, defaults to "UPRN".

    Returns:
        df (pd.DataFrame): EPC data with info on implementated recommendations.

    """

    # Merge two dataframes and check which ones had implemented recommendations
    combo = pd.merge(df1[[rec, identifier]], df2[[rec, identifier]], on=identifier)
    combo["IMPLEMENTED_" + rec] = combo[rec + "_x"] & ~combo[rec + "_y"]

    # Keep first or latest entry
    if keep == "first":
        df = pd.merge(df1, combo[["IMPLEMENTED_" + rec, identifier]], on=identifier)
    else:
        df = pd.merge(df2, combo[["IMPLEMENTED_" + rec, identifier]], on=identifier)
    return df
