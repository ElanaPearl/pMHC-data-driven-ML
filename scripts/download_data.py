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


def reformat_downloaded_data(netmhc_data_folder: Path, output_path: Path):
    """Reformats BA data from NetMHCpan download into one CSV"""

    # Load in MHC to pseudo sequence mapping
    mhc_to_pseudo_seq = (
        pd.read_csv(
            netmhc_data_folder / "MHC_pseudo.dat",
            delim_whitespace=True,
            names=["mhc", "pseudo_seq"],
        )
        .set_index("mhc")["pseudo_seq"]
        .to_dict()
    )

    # Read in BA data splits
    all_splits = []
    for cv_split in range(5):
        data = pd.read_csv(
            netmhc_data_folder / f"c00{cv_split}_ba",
            delim_whitespace=True,
            names=["peptide", "affinity", "mhc_name"],
        )
        data["cv_split"] = cv_split
        all_splits.append(data)
    data = pd.concat(all_splits, axis=0)

    # Add in pseudo sequence to data
    data["mhc_psuedo_seq"] = data["mhc_name"].map(mhc_to_pseudo_seq)

    # Reorder columns and save
    data = data[["peptide", "mhc_psuedo_seq", "affinity", "mhc_name", "cv_split"]]
    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    download_url = "https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/NetMHCpan_train.tar.gz"
    data_dir = Path("../data")
    output_path = data_dir / "IEDB_regression_data.csv"

    print("Downloading data...")
    netmhc_data_folder = download_data(download_url, download_dir=data_dir)

    print("Reformatting data...")
    reformat_downloaded_data(netmhc_data_folder, output_path=output_path)

    print("Done!")
