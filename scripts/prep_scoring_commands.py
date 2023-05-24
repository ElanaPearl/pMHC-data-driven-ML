import os
from pathlib import Path

import pandas as pd


def write_peptide_and_allele_fastas_for_cell_line(
    df, name, save_dir=Path("fastas_to_score")
):
    with open(save_dir / f"{name}_peptides.fasta", "w") as f:
        for pep in df["peptide"].unique():
            f.write(f">{pep}\n{pep}\n")

    return save_dir / f"{name}_peptides.fasta", df["allele"].iloc[0]


def standardize_allele_name(allele_name):
    if not allele_name.startswith("HLA-"):
        return allele_name
    if "*" not in allele_name:
        allele_name = allele_name[:5] + "*" + allele_name[5:]
    if ":" not in allele_name:
        allele_name = allele_name[:8] + ":" + allele_name[8:]
    return allele_name


def get_data_to_decovolute(multi_allele_path: Path):
    df = pd.read_csv(multi_allele_path)
    df = df.query("n_possible_alleles > 1 and presented == 1")
    df["peptide_len"] = df["peptide"].str.len()

    df["allele"] = df["allele"].apply(
        lambda x: ",".join(
            [standardize_allele_name(allele) for allele in x.split(",")]
        )
    )

    return df


def main(
    multi_allele_path: pd.DataFrame,
    intermediate_save_dir: Path = Path("fastas_to_score"),
    path_to_predict_file: Path = Path("../../mhc_i/src/predict_binding.py"),
    output_dir: Path = Path("netmhcpan_el_results"),
    n_cell_lines: int = 10,
):
    print("Getting data to deconvolute"")
    el_ma_actives = get_data_to_decovolute(multi_allele_path)

    if not os.path.exists(intermediate_save_dir):
        os.mkdir(intermediate_save_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cell_lines =  el_ma_actives["cell_line"].value_counts().index.tolist()
    cell_lines = cell_lines[:min(n_cell_lines, len(cell_lines))]

    print("Writing fastas to score")
    commands_to_run = []
    for cell_line in cell_lines:
        cell_line_df = el_ma_actives.query(f"cell_line == '{cell_line}'")
        for peptide_len in cell_line_df["peptide_len"].unique():
            (
                peptide_fasta,
                allele_list,
            ) = write_peptide_and_allele_fastas_for_cell_line(
                df=cell_line_df.query(f"peptide_len == {peptide_len}"),
                name=f"{cell_line}_{peptide_len}",
                save_dir=intermediate_save_dir,
            )
            peptide_lens = ",".join([peptide_len] * len(allele_list))
            output_path = output_dir / f"{cell_line}_{peptide_len}.tsv"
            commands_to_run.append(
                f"{path_to_predict_file} netmhcpan_el {allele_list} {peptide_lens} {peptide_fasta} > {output_path}"
            )

    print("Writing commands to run")
    with open("commands_to_run.sh", "w") as f:
        f.write(";\n".join(commands_to_run))


if __name__ == "__main__":
    main(multi_allele_path="../data/IEDB_classification_data.csv")
