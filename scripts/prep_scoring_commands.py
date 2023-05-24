from pathlib import Path

import pandas as pd


def write_peptide_and_allele_fastas_for_cell_line(
    df, name, save_dir=Path("fastas_to_score")
):
    with open(save_dir / f"{name}_peptides.fasta", "w") as f:
        for pep in df["peptide"].unique():
            f.write(f">{pep}\n{pep}\n")

    return save_dir / f"{name}_peptides.fasta", df["allele"].iloc[0]


def main(multi_allele_path: pd.DataFrame):
    el_ma_actives = pd.read_csv(multi_allele_path).query(
        "n_possible_alleles > 1 and presented == 1"
    )
    el_ma_actives["peptide_len"] = el_ma_actives["peptide"].str.len()

    top_10_cell_lines = (
        el_ma_actives["cell_line"].value_counts().head(10).index.tolist()
    )

    cell_line_df = el_ma_actives.query(
        f"cell_line == '{top_10_cell_lines[0]}'"
    )

    commands_to_run = []
    for cell_line in top_10_cell_lines:
        cell_line_df = el_ma_actives.query(f"cell_line == '{cell_line}'")
        for peptide_len in cell_line_df["peptide_len"].unique():
            (
                peptide_fasta,
                allele_list,
            ) = write_peptide_and_allele_fastas_for_cell_line(
                df=cell_line_df.query(f"peptide_len == {peptide_len}"),
                name=f"{cell_line}_{peptide_len}",
            )
            commands_to_run.append(
                f"./src/predict_binding.py netmhcpan_el {allele_list} {peptide_len} {peptide_fasta}"
            )

    with open("commands_to_run.sh", "w") as f:
        f.write("\n".join(commands_to_run))


if __name__ == "__main__":
    main(multi_allele_path="../data/IEDB_classification_data.csv")
