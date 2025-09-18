import glob
import os
import subprocess
import pandas as pd


def main(
    output_dir: str,
    gue_dir: str = "./GUE",
    pretrain_file="./dnabert_2_pretrain/dev.txt"
):
    for dataset_path in glob.glob(f"{gue_dir}/*/*"):
        dataset_name = ".".join(dataset_path.split("/")[-2:])
        output_path = f"{output_dir}/{dataset_name}.txt"

        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing {dataset_name}...")

        metric = "mcc"
        if "covid" in dataset_name.lower():
            metric = "f1"

        df = pd.read_csv(f"{dataset_path}/train.csv")
        if "sequence" not in df.columns:
            print(f"Skipping {dataset_name}, no 'sequence' column found.")
            continue

        cmd = [
            "python", "Train.py", "-dataset_folder", dataset_path,
            "-pretrain_file", pretrain_file,
            "--include_prev_context", "{False}",
            "--gamma", "{0.1, 0.33, 0.5, 0.75, 1, 3, 5}",
            "--nb_train_iterations", "{1, 3, 5, 7, 10}",
            "--ratio_pretrain_train", "{0}",
            "--handle_n_setting", "{remove}",
            "--ensemble_type", "{entropy}",
            "--num_threads", "{32}",
            # "--validation_metric", metric,
            "--test_metric", metric,
            # "--augmentation_factors", "{0, 0.5, 1, 2}",
            # "--shuffle_preserve_kmer", "3",
            "--max_depth", "{4, 6}"
        ]

        try:
            res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(f"Command output: {e.output.decode('utf-8')}")
            raise e
        
        with open(output_path, "w") as f:
            f.write(res)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run training on multiple datasets.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files.")
    parser.add_argument("--gue_dir", type=str, default="./GUE", help="Directory containing GUE datasets.")
    parser.add_argument("--pretrain_file", type=str, default="./dnabert_2_pretrain/dev.txt", help="Path to pretraining file.")

    args = parser.parse_args()
    
    main(args.output_dir, args.gue_dir, args.pretrain_file)
    print("Training completed for all datasets.")
