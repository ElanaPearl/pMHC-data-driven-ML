import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def convert_web_result_to_df(result: str) -> pd.DataFrame:
    """Converts the result from the web to a dataframe"""
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


def get_netmhcpan_predictions(
    peptide_seq: str, hla_name_str: "str"
) -> pd.DataFrame:
    """Returns a df of netmhcpan predictions for one peptide sequence and list of HLA alleles"""
    p_lens = ",".join([str(len(peptide_seq))] * (hla_name_str.count(",") + 1))
    method_str = f"method=netmhcpan_el-4.1&sequence_text={peptide_seq}&allele={hla_name_str}&length={p_lens}"

    result = os.popen(
        f"curl --data '{method_str}' http://tools-cluster-interface.iedb.org/tools_api/mhci/"
    ).read()

    retry_attempts = 0
    need_to_retry = True
    while need_to_retry and retry_attempts < 10:
        try:
            res_df = convert_web_result_to_df(result)
            return res_df
        except Exception as e:
            retry_attempts += 1
            time.sleep(2)
            result = os.popen(
                f"curl --data '{method_str}' http://tools-cluster-interface.iedb.org/tools_api/mhci/"
            ).read()

    return None


def get_batch_of_preds(
    peptide_seq_batch: list[str], hla_strs_batch: list[str], save_path: str
):
    """Returns a df of netmhcpan predictions for a batch of peptide sequences and list of HLA alleles"""
    assert len(peptide_seq_batch) == len(hla_strs_batch)

    peptide_dfs = []
    top_predictions = []
    failed_combos = []
    with ThreadPoolExecutor() as executor:
        future_to_prediction = {
            executor.submit(
                get_netmhcpan_predictions, peptide_seq, hla_strs
            ): (
                peptide_seq,
                hla_strs,
            )
            for peptide_seq, hla_strs in zip(peptide_seq_batch, hla_strs_batch)
        }
        for future in as_completed(future_to_prediction):
            peptide_seq, hla_strs = future_to_prediction[future]
            try:
                data = future.result()
                if data is None:
                    failed_combos.append((peptide_seq, hla_strs))
                else:
                    peptide_dfs.append(data)
                    # get the row of the dataframe with the highest value in the el_rank column
                    top_predictions.append(
                        data.iloc[data["score_el"].astype("float").idxmax()]
                    )
            except Exception as exc:
                print(
                    f"An exception occurred with peptide {peptide_seq} and hla_strs {hla_strs}: {exc}"
                )

    # Save all predictions to save_path
    if len(peptide_dfs) > 0:
        pd.concat(peptide_dfs, axis=0).to_csv(save_path)

    # Save top predictions to save_path
    if len(top_predictions) > 0:
        pd.concat(top_predictions, axis=1).T.to_csv(
            save_path.replace(".csv", "_top.csv")
        )

    # Save failed combos to save_path
    if len(failed_combos) > 0:
        pd.DataFrame(failed_combos, columns=["peptide", "allele"]).to_csv(
            save_path.replace(".csv", "_failed.csv")
        )


def batch_processing(
    chunked_dfs: list[pd.DataFrame], intermediate_cache_dir: str
):
    for i, chunk in enumerate(chunked_dfs):
        cache_path = os.path.join(intermediate_cache_dir, f"{i}.csv")
        if not os.path.exists(cache_path):
            get_batch_of_preds(
                peptide_seq_batch=chunk["peptide"],
                hla_strs_batch=chunk["allele"],
                save_path=cache_path,
            )


def combine_downloaded_files(
    download_dir: Path,
    output_path: Path = Path("netmhcpan_preds.csv"),
):
    files = [os.path.join(download_dir, fn) for fn in os.listdir(download_dir)]

    df = pd.concat([pd.read_csv(f) for f in files])
    df.to_csv(output_path, index=False)


def get_preds_on_MA_data(
    ma_data_path: Path, save_dir: Path = Path("netmhc_preds/")
):
    # read in Eluted-Ligand data
    print("Reading data")
    el_ma_actives = pd.read_csv(ma_data_path, nrows=50).query(
        "n_possible_alleles > 1 and presented == 1"
    )

    # standardize HLA names
    el_ma_actives["allele"] = el_ma_actives["allele"].apply(
        lambda x: ",".join(
            [standardize_allele_name(allele) for allele in x.split(",")]
        )
    )
    intermediate_cache_dir = save_dir / "intermediate_cache"
    if not intermediate_cache_dir.exists():
        intermediate_cache_dir.mkdir(exist_ok=True, parents=True)

    # create chunked lists of the dataset
    chunked_dfs = np.array_split(el_ma_actives, 10)

    print(f"Getting predictions for {len(el_ma_actives)} peptides")
    s = time.time()
    batch_processing(chunked_dfs, intermediate_cache_dir)
    print(f"Done in {time.time() - s:.2f} seconds")

    print("combining")
    combine_downloaded_files(
        download_dir=intermediate_cache_dir,
        output_path=save_dir / "netmhcpan_preds.csv",
    )
    print("Done")


if __name__ == "__main__":
    get_preds_on_MA_data(
        ma_data_path=Path("../data/IEDB_classification_data.csv"),
        save_dir=Path("netmhc_preds_50/"),
    )
