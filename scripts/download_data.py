import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd


def download_data(download_url: str, download_dir: Path) -> Path:
    """Downloads data, extracts it, and returns the path to the extracted folder"""
    tmp_download_fn = tempfile.NamedTemporaryFile(suffix=".tar.gz").name
    urllib.request.urlretrieve(download_url, tmp_download_fn)
    with tarfile.open(tmp_download_fn, "r:gz") as tar:
        tar.extractall(download_dir)
        extracted_dir_name = tar.getnames()[0]
    return download_dir / extracted_dir_name


def reformat_downloaded_ba_data(
    netmhc_data_folder: Path, output_path: Path, mhc_to_pseudo_seq: dict
):
    """Reformats BA data from NetMHCpan download into one CSV"""

    # Read in BA data splits
    all_ba_splits = []
    for cv_split in range(5):
        ba_data = pd.read_csv(
            netmhc_data_folder / f"c00{cv_split}_ba",
            delim_whitespace=True,
            names=["peptide", "affinity", "mhc_name"],
        )
        ba_data["cv_split"] = cv_split
        all_ba_splits.append(ba_data)
    ba_data = pd.concat(all_ba_splits, axis=0)

    # Add in pseudo sequence to data
    ba_data["mhc_pseudo_seq"] = ba_data["mhc_name"].map(mhc_to_pseudo_seq)

    # Reorder columns and save
    ba_data = ba_data[["peptide", "mhc_pseudo_seq", "affinity", "mhc_name", "cv_split"]]
    ba_data.to_csv(output_path, index=False)


def reformat_downloaded_el_data(
    netmhc_data_folder: Path, output_path: Path, mhc_to_pseudo_seq: dict
):
    """Reformats el data from NetMHCpan download into one CSV"""
    # Read in BA data splits
    all_el_splits = []
    for cv_split in range(5):
        el_data = pd.read_csv(
            netmhc_data_folder / f"c00{cv_split}_el",
            delim_whitespace=True,
            names=["peptide", "presented", "cell_line"],
        )
        el_data["cv_split"] = cv_split
        all_el_splits.append(el_data)
    el_data = pd.concat(all_el_splits, axis=0)

    # Convert cell line to list of posisble alleles
    cell_line_to_allele = (
        pd.read_csv(
            netmhc_data_folder / "allelelist",
            delim_whitespace=True,
            names=["cell_line", "mhc_name"],
        )
        .set_index("cell_line")["mhc_name"]
        .to_dict()
    )
    el_data["allele"] = el_data["cell_line"].map(cell_line_to_allele)

    # Convert allele list to pseudo sequence list
    def allele_list_to_pseudo_seq_list(allele_list):
        return ",".join(
            [mhc_to_pseudo_seq[allele] for allele in allele_list.split(",")]
        )

    el_data["mhc_pseudo_seq"] = el_data["allele"].map(allele_list_to_pseudo_seq_list)
    el_data["n_possible_alleles"] = el_data["allele"].apply(lambda x: len(x.split(",")))

    # Reorder columns and save
    el_data = el_data[
        [
            "peptide",
            "mhc_pseudo_seq",
            "presented",
            "cell_line",
            "allele",
            "n_possible_alleles",
            "cv_split",
        ]
    ]
    el_data.to_csv(output_path, index=False)

def pp_classifciation_data(load_path: str, save_path: str):
        """Filters classification dataset of 13 million points to only include peptide-allele sequences 
            with single allele data (excludes multi-allele data)""" 
        df = pd.read_csv(load_path)
        df = df[df.n_possible_alleles == 1][['peptide','presented','mhc_pseudo_seq', 'cell_line', 'cv_split']]
        
        # Rename columns to be standardized to those in the regression data
        df.rename(columns={'presented': 'affinity', 'cell_line': 'mhc_name'}, inplace=True)
        df.to_csv(save_path, index=False) 

def map_mhc_to_pseudo_seq(MHC_pseudo_file: Path):
    """Create a dictionary mapping MHC name to pseudo sequence"""
    return (
        pd.read_csv(MHC_pseudo_file, delim_whitespace=True, names=["mhc", "pseudo_seq"])
        .set_index("mhc")["pseudo_seq"]
        .to_dict()
    )


if __name__ == "__main__":
    download_url = "https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/NetMHCpan_train.tar.gz"
    data_dir = Path("../data")

    print("Downloading data...")
    netmhc_data_folder = download_data(download_url=download_url, download_dir=data_dir)
    mhc_to_pseudo_seq = map_mhc_to_pseudo_seq(netmhc_data_folder / "MHC_pseudo.dat")

    print("Reformatting el data...")
    reformat_downloaded_el_data(
        netmhc_data_folder,
        output_path=data_dir / "IEDB_classification_data.csv",
        mhc_to_pseudo_seq=mhc_to_pseudo_seq,
    )

    print("Reformatting data...")
    reformat_downloaded_ba_data(
        netmhc_data_folder,
        output_path=data_dir / "IEDB_regression_data.csv",
        mhc_to_pseudo_seq=mhc_to_pseudo_seq,
    )

    print("Filtering and saving SA classification dataset...")
    pp_classifciation_data(load_path=data_dir / "IEDB_classification_data.csv",
                            save_path=data_dir / "IEDB_classification_data_SA.csv")

    print("Done!")
