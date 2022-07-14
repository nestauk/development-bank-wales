import numpy as np
import pandas as pd

from asf_core_data.getters.epc import epc_data
from asf_core_data.pipeline.preprocessing import preprocess_epc_data

from development_bank_wales import PROJECT_DIR, get_yaml_config


# Load config file
config = get_yaml_config(PROJECT_DIR / "development_bank_wales/config/base.yaml")


rec_cat_dict = {
    "Increase loft insulation to 270 mm": "ROOF",
    "Room-in-roof insulation": "ROOF",
    "Flat roof insulation": "ROOF",
    "Cavity wall insulation": "WALLS",
    "High performance external doors": "WALLS",
    "50 mm internal or external wall insulation": "WALLS",
    "Party wall insulation": "WALLS",
    "Solid floor insulation": "FLOOR",
    "Floor insulation": "FLOOR",
    "Suspended floor insulation": "FLOOR",
    "Replace boiler with new condensing boiler": "MAINHEAT",
    "Upgrade heating controls": "HEATING",
    "Change heating to gas condensing boiler": "MAINHEAT",
    "Upgrading heating controls": "MAINHEAT",
    "High heat retention storage heaters": "MAINHEAT",
    "Flue gas heat recovery device in conjunction with boiler": "MAINHEAT",
    "Change room heaters to condensing boiler": "MAINHEAT",
    "Time and temperature zone control": "MAINHEAT",
    "Fan assisted storage heaters and dual immersion cylinder": "MAINHEAT",
    "Fan assisted storage heaters": "MAINHEAT",
    "Replace heating unit with condensing unit": "MAINHEAT",
    "Fan-assisted storage heaters": "MAINHEAT",
    "High heat retention storage heaters and dual immersion cylinder": "MAINHEAT",
    "Replace boiler with biomass boiler": "MAINHEAT",
    "Install condensing boiler": "MAINHEAT",
    "Replacement warm air unit": "MAINHEAT",
    "Condensing oil boiler with radiators": "MAINHEAT",
    "Wood pellet stove with boiler and radiators": "MAINHEAT",
    "Solar water heating": "HOT_WATER",
    "Hot water cylinder thermostat": "HOT_WATER",
    "Increase hot water cylinder insulation": "HOT_WATER",
    "Insulate hot water cylinder with 80 mm jacket": "HOT_WATER",
    "Add additional 80 mm jacket to hot water cylinder": "HOT_WATER",
    "Heat recovery system for mixer showers": "HOT_WATER",
    "Solar photovoltaic panels, 2.5 kWp": "ENERGY",
    "Wind turbine": "ENERGY",
    "Draughtproof single-glazed windows": "WINDOWS",
    "Replace single glazed windows with low-E double glazing": "WINDOWS",
    "Replacement glazing units": "WINDOWS",
    "Secondary glazing to single glazed windows": "WINDOWS",
    "Low energy lighting for all fixed outlets": "LIGHTING",
}


def get_bool_recom_features(df, unique_recs):

    """Get boolean recommendation features.
    New features indicate whether property received recommendations or not.

    Args:
        df (pd.DataFrame): needs to include "IMPROVEMENT_ID_TEXT" and "LMK_KEY".
        all_recs (list): list of unique default recommendations

    Returns:
        df (pd.DataFrame): updated df with boolean recommendation features.

    """

    all_keys = []

    # For each recommendation,
    # create bool feature indicating whether recommendation is suggested for property
    for rec in unique_recs:

        # Skip over NaN
        if isinstance(rec, float):
            continue

        # Mask for EPC records with given recommendation
        mask = df[df["RECOMMENDATIONS"].notna()].RECOMMENDATIONS.apply(
            lambda x: any(item for item in [rec] if item in x)
        )

        # Get identifiers for properties with this recommendation
        rec_keys = list(df[df["RECOMMENDATIONS"].notna()][mask]["LMK_KEY"].unique())
        all_keys += rec_keys

        df["Rec: {}".format(rec)] = np.where(df["LMK_KEY"].isin(rec_keys), True, False)

        rec_cat = rec_cat_dict[rec] + "_RECOMMENDATION"
        df.loc[df["LMK_KEY"].isin(rec_keys), rec_cat] = True

    # Fill up the NaNs with False
    for category in rec_cat_dict.values():
        df[category + "_RECOMMENDATION"].fillna(False, inplace=True)

    # Check whether property has any recommendation
    df["HAS_ANY_RECOM"] = np.where(df["LMK_KEY"].isin(set(all_keys)), True, False)

    return df


def load_epc_certs_and_recs(
    data_path,
    subset="GB",
    usecols=config["EPC_FEAT_SELECTION"],
    n_samples=None,
    remove_duplicates=False,
    reload=True,
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

    if reload:
        df = preprocess_epc_data.load_and_preprocess_epc_data(
            data_path=data_path,
            subset=subset,
            usecols=usecols,
            n_samples=n_samples,
            remove_duplicates=remove_duplicates,
        )

    else:

        version = "preprocessed_dedupl" if remove_duplicates else "preprocessed"
        df = epc_data.load_preprocessed_epc_data(
            data_path=data_path,
            batch="newest",
            version=version,
            usecols=usecols,
            n_samples=n_samples,
            low_memory=True,
        )

    # Currently not implement for Scotland data
    if subset == "GB":
        df = df.loc[df["COUNTRY"] != "Scotland"]

    recommendations = epc_data.load_england_wales_recommendations(
        data_path=data_path, subset=subset
    )

    # Rename feature for consistent processing later
    df = df.rename(columns={"HOTWATER_DESCRIPTION": "HOT_WATER_DESCRIPTION"})

    # Get all recommendations
    recs = list(recommendations["IMPROVEMENT_ID_TEXT"].unique())

    rec_dict = (
        recommendations.groupby(["LMK_KEY"])["IMPROVEMENT_ID_TEXT"]
        .apply(list)
        .to_dict()
    )

    # Map recommendations to EPC records
    df["RECOMMENDATIONS"] = df["LMK_KEY"].map(rec_dict)

    df = get_bool_recom_features(df, recs)

    return df


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
