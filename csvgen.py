import pandas as pd
from pathlib import Path
import os
import argparse


def process_architecture(arch):
    results = {}

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            is_full_dataset = (
                f"runs_eurosat_{arch}" in dir_name and "split" not in dir_name
            )
            if dir_name.startswith(f"runs_eurosat_{arch}") and (
                "_split" in dir_name or is_full_dataset
            ):
                split = "100" if is_full_dataset else dir_name.split("_split")[-1]
                arch_dir = os.path.join(root, dir_name)

                for sub_root, sub_dirs, _ in os.walk(arch_dir):
                    for sub_dir in sub_dirs:
                        csv_file_path = os.path.join(
                            sub_root, sub_dir, "best_test_metrics.csv"
                        )
                        if os.path.exists(csv_file_path):
                            df = pd.read_csv(csv_file_path)
                            micro_f1_score = df[df["Measurement Type"] == "micro"][
                                "F1-Score"
                            ].values[0]
                            if sub_dir not in results:
                                results[sub_dir] = {}
                            results[sub_dir][split] = micro_f1_score

    if results:
        results_df = pd.DataFrame.from_dict(results, orient="index").sort_index()
        results_df.columns = results_df.columns.map(int)
        results_df = results_df.sort_index(axis=1)
        output_csv_path = os.path.join(base_dir, f"{arch}_results.csv")
        results_df.to_csv(output_csv_path, header=True, index_label="Folder")
        print(f"CSV file for {arch} has been saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the directory containing the runs",
        default="/users/benjrise/sharedscratch2/torch-migration/runs/eurosat",
    )
    args = parser.parse_args()
    base_dir = Path(args.path)
    archs = ["resnet50", "vit", "densenet121"]
    for arch in archs:
        process_architecture(arch)
