import os
from pathlib import Path

import pandas as pd


def reformat_netmhc_data(netmhc_data_folder: Path):
    """ Reformats BA data from NetMHCpan download into one CSV"""

    # Load in MHC to pseudo sequence mapping
    mhc_to_pseudo_seq = pd.read_csv(netmhc_data_folder / "MHC_pseudo.dat",
                                    delim_whitespace=True,
                                    names=["mhc", "pseudo_seq"]).set_index("mhc")["pseudo_seq"].to_dict()

    # Read in BA data splits
    all_splits = []
    for cv_split in range(5):
        data = pd.read_csv(netmhc_data_folder / f'c00{cv_split}_ba', delim_whitespace=True, names=["peptide", "affinity", "mhc_name"])
        data["cv_split"] = cv_split
        all_splits.append(data)
    data = pd.concat(all_splits, axis=0)

    # Add in pseudo sequence to data
    data["mhc_psuedo_seq"] = data["mhc_name"].map(mhc_to_pseudo_seq)

    # Reorder columns and save
    data = data[["peptide", "mhc_psuedo_seq", "affinity", "mhc_name", "cv_split"]]
    if not os.path.exists("data"):
        os.makedirs("data")
    data.to_csv("data/IEDB_regression_data.csv", index=False)


if __name__ == "__main__":
    reformat_netmhc_data(Path("NetMHCpan_train"))
