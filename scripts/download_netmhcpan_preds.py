import os
from pathlib import Path

import numpy as np
import pandas as pd


def standardize_allele_name(allele_name):
    if not allele_name.startswith("HLA-"):
        return allele_name
    if "*" not in allele_name:
        allele_name = allele_name[:5] + "*" + allele_name[5:]
    if ":" not in allele_name:
        allele_name = allele_name[:8] + ":" + allele_name[8:]
    return allele_name


def get_netmhcpan_predictions(
    peptide_seq: str, hla_name_str: "str"
) -> pd.DataFrame:
    """Returns a df of netmhcpan predictions for one peptide sequence and list of HLA alleles"""
    print(hla_name_str)
    p_lens = ",".join([str(len(peptide_seq))] * (hla_name_str.count(",") + 1))
    method_str = f"method=netmhcpan_el-4.1&sequence_text={peptide_seq}&allele={hla_name_str}&length={p_lens}"

    print(
        f"curl --data '{method_str}' http://tools-cluster-interface.iedb.org/tools_api/mhci/"
    )

    result = os.popen(
        f"curl --data '{method_str}' http://tools-cluster-interface.iedb.org/tools_api/mhci/"
    ).read()

    if "invalid" in result:
        print(
            f"curl --data '{method_str}' http://tools-cluster-interface.iedb.org/tools_api/mhci/"
        )
        print(result)
        raise Exception("Invalid")

    peptide_df = pd.DataFrame(
        [line.split("\t") for line in result.rstrip().split("\n")][1:],
        columns=[
            line.split("\t")[0]
            for line in result.rstrip().split("\n")[0:1][0].split("\t")
        ],
    )

    return peptide_df[
        ["allele", "peptide", "score", "percentile_rank"]
    ].rename(
        {"score": "score_el", "percentile_rank": "percentile_rank_el"},
        axis=1,
    )


def get_batch_of_preds(peptide_seq_batch, hla_strs_batch, save_path: Path):
    """Returns a df of netmhcpan predictions for a batch of peptide sequences and list of HLA alleles"""
    assert len(peptide_seq_batch) == len(hla_strs_batch)

    peptide_dfs = []
    for peptide_seq, hla_strs in zip(peptide_seq_batch, hla_strs_batch):
        peptide_dfs.append(get_netmhcpan_predictions(peptide_seq, hla_strs))
    pd.concat(peptide_dfs, axis=0).to_csv(save_path)


def combine_downloaded_files(
    download_dir: Path,
    output_path: Path = Path("netmhcpan_preds.csv"),
):
    files = [os.path.join(download_dir, fn) for fn in os.listdir(download_dir)]

    df = pd.concat([pd.read_csv(f) for f in files])
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # read in Eluted-Ligand data
    print("Reading data")
    el_ma_actives = pd.read_csv(
        "../data/IEDB_classification_data.csv", nrows=50
    ).query("n_possible_alleles > 1 and presented == 1")

    # standardize HLA names
    el_ma_actives["allele"] = el_ma_actives["allele"].apply(
        lambda x: ",".join(
            [standardize_allele_name(allele) for allele in x.split(",")]
        )
    )
    save_dir = Path("netmhc_preds/")
    intermediate_cache_dir = save_dir / "intermediate_cache"
    if not intermediate_cache_dir.exists():
        intermediate_cache_dir.mkdir(exist_ok=True)

    # create lists of
    chunked_dfs = np.array_split(el_ma_actives, 10)

    print(f"Getting predictions for {len(el_ma_actives)} peptides")

    for i, chunk in enumerate(chunked_dfs):
        cache_path = intermediate_cache_dir / f"{i}.csv"
        try:
            if not os.path.exists(cache_path):
                get_batch_of_preds(
                    peptide_seq_batch=chunk["peptide"],
                    hla_strs_batch=chunk["allele"],
                    save_path=cache_path,
                )
        except Exception as e:
            print(f"Error {e} with chunk {i}")

    print("combining")
    combine_downloaded_files(
        download_dir=intermediate_cache_dir,
        output_path=save_dir / "netmhcpan_preds.csv",
    )
    print("Done")
