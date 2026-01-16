import tempfile
from lz78 import Sequence, CharacterMap, LZ78SPA, spa_from_file
from lz78 import BackgroundPriors, KmerMultinomial, Sequences
import numpy as np
from os import makedirs
import time
import pandas as pd
import random
import argparse
import math
import itertools
from memory_profiler import profile
import os
from sklearn.metrics import average_precision_score, roc_auc_score
from dataclasses import dataclass, field
import argparse
import gc
import shutil
import multiprocessing
import subprocess
from sklearn.metrics import matthews_corrcoef, f1_score


ALPHABET = "ACGT"

# ============================ external counters ============================

def check_bin(path_or_name):
    if os.path.isabs(path_or_name):
        return path_or_name if os.path.exists(path_or_name) else None
    return shutil.which(path_or_name)

def acgt_fraction(s: str) -> float:
    if not s: return 0.0
    ok = sum(ch in ALPHABET for ch in s)
    return ok / len(s)


def write_class_fastas(df_train, outdir):
    class_to_path = {}
    for c, sub in df_train.groupby("label"):
        p = os.path.join(outdir, f"class_{int(c)}.fa")
        with open(p, "w") as fh:
            for i, seq in enumerate(sub["sequence"].values):
                fh.write(f">{c}_{i}\n{seq}\n")
        class_to_path[int(c)] = p
    return class_to_path

# Jellyfish (canonical via -C; contiguous k-mers only)
def count_with_jellyfish(class_fastas, k, threads=8, hashsize="200M", canonical=True,
                         jellyfish_bin="jellyfish", tdir=None):
    jf = check_bin(jellyfish_bin)
    if not jf:
        raise RuntimeError("jellyfish binary not found. Set --jellyfish_bin.")
    files = []
    for c, fa in class_fastas.items():
        db = os.path.join(tdir, f"class_{c}.jf")
        cmd = [jf, "count", "-m", str(k), "-s", str(hashsize), "-t", str(threads), "-o", db]
        if canonical: cmd.append("-C")
        cmd.append(fa)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Dump kmer\tcount
        tsv = os.path.join(tdir, f"class_{c}.tsv")
        with open(tsv, "w") as outfh:
            subprocess.run([jf, "dump", "-c", db], check=True, stdout=outfh, stderr=subprocess.PIPE)
        files.append(tsv)
    return files


@dataclass
class HyperparameterSweep:
    alphas: list[float]
    ks: list[int]
    feature_modes: list[str]


def train_spa_oneIter(
    X_train: list[Sequence],
    y_train: list[int],
    spa: list[LZ78SPA]
):
    for seq, label in zip(X_train, y_train):
        spa[label].reset_state()
        spa[label].train_on_block(seq)


def test_seq(
    df: pd.DataFrame,
    model: KmerMultinomial,
    n_threads=32
):
    # for every test seq, run it through all spas
    # classification = label associated with lowest loss spa
    # check classification against ground truth
    # compute metric (of all test runs)

    labels = df["label"].values
    data_seq = Sequences(df["sequence"].to_list())

    log_losses = np.array(model.class_log_loss_parallel(data_seq, num_threads=n_threads)).T
    classes = np.argmin(log_losses, axis=0)

    results = {}
    results["accuracy"] = float((classes == labels).sum() / len(labels))
    results["mcc"] = matthews_corrcoef(labels, classes)
    results["f1"] = f1_score(labels, classes, average='weighted')

    if log_losses.shape[0] == 2:
        scores = log_losses[0, :] - log_losses[1, :]  # higher => more likely class 1
        results["auprc"] = average_precision_score(labels, scores)
        results["auroc"] = roc_auc_score(labels, scores)
    return results


def bitstr_to_array(col):
    return np.array([list(map(int, list(s))) for s in col], dtype=np.uint8)


def get_df_for_label(df, label_idx):
    df["label"] = bitstr_to_array(df["label"].values)[:, label_idx]
    # remove Ns from each sequence
    df["sequence"] = df["sequence"].str.replace("N", "", regex=False)
    return df


class MultiLabelClassifier:
    def __init__(
        self, n_labels: int,
        data_dir: str,
        hyperparams: HyperparameterSweep,
        spa_dir: str,
        canonical: bool = True,
        val_metric: str = "auroc",
        inf_threads: int = 8,
        job_threads: int = 8,
        jf_hashsize: str = "200M",
        jellyfish_bin: str = "jellyfish",
    ):
        self.n_labels = n_labels
        self.train_csv = f"{data_dir}/train.csv"
        self.val_csv = f"{data_dir}/dev.csv"
        self.test_csv = f"{data_dir}/test.csv"

        os.makedirs(spa_dir, exist_ok=True)
        self.model_paths = [f"{spa_dir}/label_{i}_kmer.bin" for i in range(n_labels)]
        self.hyperparams = hyperparams
        self.inf_threads = inf_threads
        self.job_threads = job_threads
        self.val_metric = val_metric

        self.use_canonical = canonical
        self.jf_hashsize = jf_hashsize
        self.jellyfish_bin = jellyfish_bin


    def _train_one_label(
        self, label_idx: int
    ):
        print(f"[Training label {label_idx}]")
        train_data = get_df_for_label(
            pd.read_csv(self.train_csv, dtype={"label": str}),
            label_idx
        )
        validation_data = get_df_for_label(
            pd.read_csv(self.val_csv, dtype={"label": str}),
            label_idx
        )

        best_val_metric = 0

        for k in self.hyperparams.ks: 
            with tempfile.TemporaryDirectory() as tmp:
                class_fastas = write_class_fastas(train_data, tmp)
                files = count_with_jellyfish(
                    class_fastas, k,
                    threads=self.inf_threads, hashsize=self.jf_hashsize,
                    canonical=self.use_canonical, jellyfish_bin=self.jellyfish_bin,
                    tdir=tmp
                )

                for alpha, fmode in itertools.product(self.hyperparams.alphas, self.hyperparams.feature_modes):
                    model = KmerMultinomial(
                        k=k, alpha=alpha, feature_mode=fmode,
                        canonical=self.use_canonical,
                        background_priors=BackgroundPriors(self.use_canonical)
                    )
                    model.fit_from_files(files)

                    val_metric_value = test_seq(
                        validation_data, model,
                        n_threads=self.inf_threads,
                    )[self.val_metric]

                    if val_metric_value > best_val_metric or best_val_metric == 0:
                        best_val_metric = val_metric_value
                        spa_path = self.model_paths[label_idx]
                        with open(spa_path, "wb") as f:
                            f.write(model.to_bytes())

                    print(f"Label {label_idx} | k {k} | alpha {alpha} | fmode {fmode} | {self.val_metric}: {val_metric_value:.4f}", flush=True)


    def _test_one_label(
        self, label_idx: int
    ):
        print(f"[Testing label {label_idx}]")
        test_data = get_df_for_label(
            pd.read_csv(self.test_csv, dtype={"label": str}),
            label_idx
        )

        spa_path = self.model_paths[label_idx]
        try:
            spa = KmerMultinomial.from_file(spa_path)
        except Exception as e:
            print(f"Error loading spa for label {label_idx} from {spa_path}: {e}")
            spa = LZ78SPA(alphabet_size=4)  # Dummy spa to avoid crash

        results = test_seq(
            test_data, spa,
            n_threads=self.inf_threads,
        )
        return results


    def train_all_labels(self):
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(self.job_threads) as pool:
            pool.map(self._train_one_label, range(self.n_labels))
    
    
    def test_all_labels(self):
        all_results = {}
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(self.job_threads) as pool:
            all_results_list = pool.map(self._test_one_label, range(self.n_labels))
        for i, res in enumerate(all_results_list):
            all_results[f"label_{i}"] = res
        return all_results


@profile
def main():
    #Parse all arguments
    parser = argparse.ArgumentParser(description="Script for training and testing SPA model")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing train.csv, dev.csv, test.csv")
    parser.add_argument("--n_labels", type=int, required=True,
                        help="Number of labels")
    parser.add_argument("--spa_dir", type=str, required=True,
                        help="Directory to save SPA models")
    parser.add_argument("--val_metric", type=str, default="auroc", choices=["accuracy", "mcc", "f1", "auroc", "auprc"],
                        help="Metric to use for model selection on validation set")
    parser.add_argument("--inf_threads", type=int, default=8,
                        help="Number of threads to use for inference")
    parser.add_argument("--job_threads", type=int, default=8,
                        help="Number of parallel jobs to run for training/testing labels")
    parser.add_argument("--alphas", type=float, nargs='+', default=[0.1, 0.5, 1, 3, 5],
                        help="List of alpha values to try")
    parser.add_argument("--ks", type=int, nargs='+', default=[12],
                        help="List of k-mer sizes to try")
    parser.add_argument("--feature_modes", type=str, nargs='+', default=["count"],
                        help="List of feature modes to try")
    parser.add_argument("--canonical", action="store_true",
                        help="Use canonical k-mers")
    parser.add_argument("--jf_hashsize", type=str, default="200M",
                        help="Jellyfish hash size")
    parser.add_argument("--jellyfish_bin", type=str, default="jellyfish",
                        help="Path to jellyfish binary")
    parser.add_argument("--just_test", action="store_true",
                        help="If set, only run tests without training")
    args = parser.parse_args()

    hyperparams = HyperparameterSweep(
        alphas=args.alphas,
        ks=args.ks,
        feature_modes=args.feature_modes,
    )

    classifier = MultiLabelClassifier(
        n_labels=args.n_labels,
        data_dir=args.data_dir,
        hyperparams=hyperparams,
        spa_dir=args.spa_dir,
        val_metric=args.val_metric,
        inf_threads=args.inf_threads,
        job_threads=args.job_threads,
        canonical=args.canonical,
        jf_hashsize=args.jf_hashsize,
        jellyfish_bin=args.jellyfish_bin,
    )
    if not args.just_test:
        print("TRAINING-----", flush=True)
        train_start = time.perf_counter()
        classifier.train_all_labels()
        train_end = time.perf_counter()
        print(f"Training time: {train_end - train_start:.2f} seconds")

    print("TESTING-----", flush=True)
    test_start = time.perf_counter()
    all_results = classifier.test_all_labels()
    test_end = time.perf_counter()
    print(f"Testing time: {test_end - test_start:.2f} seconds")
    print("All test results:", flush=True)
    for label, res in all_results.items():
        print(f"{label}: {res}")
    
    # also compute the average
    avg_results = {}
    for key in ["accuracy", "mcc", "f1", "auroc", "auprc"]: # don't average the "per-" metrics
        vals = [res[key] for res in all_results.values() if key in res]
        if vals:
            avg_results[key] = float(np.mean(vals))
    print(f"\n\nAverage results across labels: {avg_results}", flush=True)

if __name__ == "__main__":
    main()