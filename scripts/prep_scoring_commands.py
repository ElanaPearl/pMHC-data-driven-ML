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


def main(multi_allele_path: pd.DataFrame):
    el_ma_actives = pd.read_csv(multi_allele_path).query(
        "n_possible_alleles > 1 and presented == 1"
    )
    el_ma_actives["peptide_len"] = el_ma_actives["peptide"].str.len()

    el_ma_actives["allele"] = el_ma_actives["allele"].apply(
        lambda x: ",".join(
            [standardize_allele_name(allele) for allele in x.split(",")]
        )
    )

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
                f"./src/predict_binding.py netmhcpan_el {allele_list} {','.join([peptide_len]*len(allele_list))} {peptide_fasta}"
            )

    with open("commands_to_run.sh", "w") as f:
        f.write("\n".join(commands_to_run))


if __name__ == "__main__":
    main(multi_allele_path="../data/IEDB_classification_data.csv")
