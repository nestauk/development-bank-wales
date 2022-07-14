from development_bank_wales.pipeline.feature_preparation import upgrades
from asf_core_data.getters.epc import epc_data


def roof_description_features(df):

    df["ROOF TYPE"] = df["ROOF_DESCRIPTION"].str.extract(
        r"(Pitched|Flat|Roof room\(s\)|Flat|Ar oleddf|\(other premises above\)|Other premises above|\(another dwelling above\)|\(eiddo arall uwchben\))|\(annedd arall uwchben\)"
    )

    df["ROOF TYPE"] = df["ROOF TYPE"].replace(["Ar oleddf"], "Pitched")

    df["LOFT INSULATION [in mm]"] = df["ROOF_DESCRIPTION"].str.extract(
        r"(\d{1,3})\+?\s+mm loft insulation"
    )
    df["ROOF THERMAL TRANSMIT"] = df["ROOF_DESCRIPTION"].str.extract(
        r"Average thermal transmittance|Trawsyriannedd thermol cyfartalog=?\s*(0\.\d{1,4})\s*W"
    )
    df["ROOF INSULATION"] = df["ROOF_DESCRIPTION"].str.extract(
        r"(no insulation|insulated at rafters|limited insulation|ceiling insulated|insulated \(assumed\))"
    )

    df["ROOF TYPE"] = df["ROOF TYPE"].replace(
        [
            "(other premises above)",
            "(eiddo arall uwchben)",
            "(another dwelling above)",
            "Other premises above",
            "(annedd arall uwchben)",
        ],
        "another dwelling above",
    )

    # .drop(columns=["ROOF_DESCRIPTION"], inplace=True)

    return df


def walls_description_features(df):

    df["WALL TYPE"] = df["WALLS_DESCRIPTION"].str.extract(
        r"(Cavity wall|Sandstone|Solid brick|Sandstone or limestone|System built|Timber frame|Granite or whin|Park home wall|Waliau ceudod|Gwenithfaen|\(other premises below\)|\(another dwelling below\)|\(anheddiad arall islaw\)|\(Same dwelling below\))"
    )

    df["WALL TYPE"] = df["WALL TYPE"].replace(["Waliau ceudod"], "Cavity wall")
    df["WALL TYPE"] = df["WALL TYPE"].replace(["Gwenithfaen"], "Granite or whin")

    df["WALLS THERMAL TRANSMIT"] = df["WALLS_DESCRIPTION"].str.extract(
        r"Average thermal transmittance|Trawsyriannedd thermol cyfartalog=?\s*(0\.\d{1,4})\s*W"
    )
    df["WALLS INSULATION"] = df["WALLS_DESCRIPTION"].str.extract(
        r"(insulated|no insulation|filled cavity|with external insulation|with internal insulation|partial insulated)"
    )

    # df.drop(columns=["WALLS_DESCRIPTION"], inplace=True)

    return df


def floor_description_features(df):

    df["FLOOR TYPE"] = df["FLOOR_DESCRIPTION"].str.extract(
        r"(Solid|Suspended|To unheated space|Solet|To external air)"
    )

    df["FLOOR TYPE"] = df["FLOOR TYPE"].replace(["Solet"], "Solid")
    df["FLOOR TYPE"] = df["FLOOR TYPE"].replace(
        ["I ofod heb ei wresogi"], "To unheated space"
    )

    df["FLOOR TYPE"] = df["FLOOR TYPE"].replace(
        [
            "(other premises below)",
            "(anheddiad arall islaw)",
            "(another dwelling below)",
            "(Same dwelling below)",
        ],
        "another dwelling below",
    )

    df["FLOOR THERMAL TRANSMIT"] = df["FLOOR_DESCRIPTION"].str.extract(
        r"Average thermal transmittance|Trawsyriannedd thermol cyfartalog=?\s*(0\.\d{1,4})\s*W"
    )
    df["FLOOR INSULATION"] = df["FLOOR_DESCRIPTION"].str.extract(
        r"(insulated|no insulation|limited insulation|partial insulated|uninsulated)"
    )

    df["FLOOR INSULATION"] = df["FLOOR INSULATION"].replace(
        ["uninsulated"], "no insulation"
    )
    df["FLOOR INSULATION"] = df["FLOOR INSULATION"].replace(
        ["limited insulatio"], "partial insulated"
    )

    # df.drop(columns=["FLOOR_DESCRIPTION"], inplace=True)

    return df


def clean_description_features(df):

    df = roof_description_features(df)
    df = walls_description_features(df)
    df = floor_description_features(df)

    return df


def computing_upgradability(df, verbose=False):

    for cat in [
        "ROOF",
        "WINDOWS",
        "WALLS",
        "FLOOR",
        "LIGHTING",
        "HOT_WATER",
        "MAINHEAT",
    ]:

        total_props = df.shape[0]
        total_props_with_rec = df.loc[(df["{}_RECOMMENDATION".format(cat)])].shape[0]
        total_props_with_upgr = df.loc[(df["{}_EFF_DIFF".format(cat)] > 0)].shape[0]

        props_w_upgr_and_rec = df.loc[
            (
                (df["{}_EFF_DIFF".format(cat)] > 0)
                & (df["{}_RECOMMENDATION".format(cat)])
            )
        ].shape[0]

        props_w_upgr_or_rec = df.loc[
            (
                (df["{}_EFF_DIFF".format(cat)] > 0)
                | (df["{}_RECOMMENDATION".format(cat)])
            )
        ].shape[0]

        df["{}_UPGRADABILITY".format(cat)] = (df["{}_EFF_DIFF".format(cat)] > 0.0) | (
            df["{}_RECOMMENDATION".format(cat)]
        )

        if verbose:
            print(cat)
            print(
                "Recommends:\t{:.2f}%".format(total_props_with_rec / total_props * 100)
            )
            print(
                "Upgrades:\t{:.2f}%".format(total_props_with_upgr / total_props * 100)
            )
            print(
                "Upgrade + Rec:\t{:.2f}%".format(
                    props_w_upgr_and_rec / total_props * 100
                )
            )
            print(
                "Upgrade | Rec:\t{:.2f}%".format(
                    props_w_upgr_or_rec / total_props * 100
                )
            )
            print(
                "Upgrade + Rec (of those with recommendation):\t{:.2f}%".format(
                    props_w_upgr_and_rec / total_props_with_rec * 100
                )
            )
            print("Coverage:\t{:.2f}%".format(props_w_upgr_or_rec / total_props * 100))
            print(
                "Upgr.ility:\t{:.2f}".format(df["{}_UPGRADABILITY".format(cat)].mean())
            )
            print()

    return df


def preprocess_features(wales_df):

    # Clean features (to be moved to ASF core data)
    wales_df = clean_description_features(wales_df)

    # Only consider owner-occupied properties
    wales_df = wales_df.loc[wales_df["TENURE"] == "owner-occupied"]

    # Get upgrade information based on properties with multiple entries
    latest_wales = epc_data.filter_by_year(wales_df, None, selection="latest entry")
    first_wales = epc_data.filter_by_year(wales_df, None, selection="first entry")

    upgrade_df = upgrades.get_upgrade_features(first_wales, latest_wales, keep="first")
    upgrade_df = computing_upgradability(upgrade_df)

    return upgrade_df
