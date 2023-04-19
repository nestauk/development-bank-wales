# File: development_bank_wales/getters/wales_epc.py
"""Get and process EPC data, both EPC records and recommendations."""

# ----------------------------------------------------------------------------------

from development_bank_wales import PROJECT_DIR, Path
from development_bank_wales.pipeline.feature_preparation import recommendations

import pandas as pd

# ---------------------------------------------------------------------------------

output_path = PROJECT_DIR / "outputs/data/wales_epc_with_recs.csv"
fig_output_path = PROJECT_DIR / "outputs/figures/"


def get_wales_data():
    """Get Wales EPC data, both EPC records and recommendations.

    Returns:
        wales_df: Wales EPC data.
    """

    if not Path(output_path).is_file():

        print("Loading and preparing the data...")

        wales_df = recommendations.load_epc_certs_and_recs(
            data_path="S3", subset="Wales", n_samples=None, remove_duplicates=False
        )

        wales_df.to_csv(output_path, index=False)

    else:

        print("Loading the data...")
        wales_df = pd.read_csv(output_path)

    print("Done!")
    return wales_df
