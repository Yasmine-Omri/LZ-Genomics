from lz78 import Sequence, CharacterMap, LZ78SPA, spa_from_file
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
from ushuffle import shuffle, Shuffler
from dataclasses import dataclass, field
import argparse
import gc
import multiprocessing
from sklearn.metrics import matthews_corrcoef, f1_score


DEFAULT_BACKSHIFT_LEN = 20
DEFAULT_ENSEMBLE_N = 10


@dataclass
class HyperparameterSweep:
    gamma: list[float]
    nb_train_iterations: list[int]
    max_depth: list[int]


def train_spa_oneIter(
    X_train: list[Sequence],
    y_train: list[int],
    spa: list[LZ78SPA]
):
    for seq, label in zip(X_train, y_train):
        spa[label].reset_state()
        spa[label].train_on_block(seq)


def compute_scores_matrix(X_val: list[Sequence], spas: list[LZ78SPA], n_threads=32):
    """
    Returns scores with shape (N, K), where higher means more likely class.
    We use negative avg_log_loss as a score.
    """
    class_losses = []
    for spa in spas:
        # list of dicts -> extract avg_log_loss for each seq
        try:
            class_losses.append([r["avg_log_loss"] for r in spa.compute_test_loss_parallel(X_val, num_threads=n_threads)])
        except Exception as e:
            print(f"Error computing test loss for spa: {e}")
            class_losses.append([float("inf")] * len(X_val))  # Assign high loss if error occurs
    losses = np.vstack(class_losses)         # (K, N)
    scores = (-losses).T                     # (N, K)
    return scores


def test_seq(
    X_val: list[Sequence],
    y_val: list[int],
    spas: list[LZ78SPA],
    n_threads=32
):
    # for every test seq, run it through all spas
    # classification = label associated with lowest loss spa
    # check classification against ground truth
    # compute metric (of all test runs)

    labels = y_val
    data_seq = X_val
    log_losses = np.zeros((len(spas), len(data_seq)))
    for i in range(len(spas)):
        try:
            log_losses[i, :] = [res["avg_log_loss"] for res in spas[i].compute_test_loss_parallel(data_seq, num_threads=n_threads)]
        except Exception as e:
            print(f"Error computing log losses for spa {i}: {e}")
            log_losses[i, :] = np.full(len(data_seq), np.inf)  # Assign high loss if error occurs
    classes = np.argmin(log_losses, axis=0)

    results = {}
    results["accuracy"] = float((classes == labels).sum() / len(labels))
    results["mcc"] = matthews_corrcoef(labels, classes)
    results["f1"] = f1_score(labels, classes, average='weighted')

    if len(spas) == 2:
        scores = log_losses[0, :] - log_losses[1, :]  # higher => more likely class 1
        results["auprc"] = average_precision_score(labels, scores)
        results["auroc"] = roc_auc_score(labels, scores)
    return results


def bitstr_to_array(col):
    return np.array([list(map(int, list(s))) for s in col], dtype=np.uint8)


class MultiLabelClassifier:
    def __init__(
        self, n_labels: int,
        data_dir: str,
        hyperparams: HyperparameterSweep,
        spa_dir: str,
        val_metric: str = "auroc",
        inf_threads: int = 8,
        job_threads: int = 8,
    ):
        self.n_labels = n_labels
        self.train_csv = f"{data_dir}/train.csv"
        self.val_csv = f"{data_dir}/dev.csv"
        self.test_csv = f"{data_dir}/test.csv"

        os.makedirs(spa_dir, exist_ok=True)
        self.model_paths = [(
            f"{spa_dir}/label_{i}_spa_0.bin",
            f"{spa_dir}/label_{i}_spa_1.bin"
        ) for i in range(n_labels)]
        self.hyperparams = hyperparams
        self.inf_threads = inf_threads
        self.job_threads = job_threads
        self.val_metric = val_metric


    def _train_one_label(
        self, label_idx: int
    ):
        print(f"[Training label {label_idx}]")
        train_data = pd.read_csv(self.train_csv, dtype={"label": str})
        validation_data = pd.read_csv(self.val_csv, dtype={"label": str})

        # remove Ns from each sequence
        train_data["sequence"] = train_data["sequence"].str.replace("N", "")
        validation_data["sequence"] = validation_data["sequence"].str.replace("N", "")

        X_train = [
            Sequence(s, charmap=CharacterMap("ACGT")) for s in train_data["sequence"]
        ]
        y_train = [x[label_idx] for x in bitstr_to_array(train_data["label"])]
        X_val = [
            Sequence(s, charmap=CharacterMap("ACGT")) for s in validation_data["sequence"]
        ]
        y_val = [x[label_idx] for x in bitstr_to_array(validation_data["label"])]

        del train_data
        del validation_data
        gc.collect()

        best_val_metric = 0

        for max_depth in self.hyperparams.max_depth: 
            spa0 = LZ78SPA(
                alphabet_size=4,
                compute_training_loss=False,
                max_depth=int(max_depth) if max_depth else None
            )
            spa1 = LZ78SPA(
                alphabet_size=4,
                compute_training_loss=False,
                max_depth=int(max_depth) if max_depth else None
            )

            backshift_len = DEFAULT_BACKSHIFT_LEN
            ensemble_n = DEFAULT_ENSEMBLE_N

            if max_depth and backshift_len > max_depth // 3:
                backshift_len = max(1, max_depth // 3)
                ensemble_n = min(ensemble_n, max(1, int(backshift_len * 0.6)))
                print(f"Adjusted backshift_len to {backshift_len} and ensemble_n to {ensemble_n} due to max_depth {max_depth}", flush=True)

            for sp in (spa0, spa1):
                sp.set_inference_config(
                    lb=1e-5,
                    ensemble_type="entropy",
                    ensemble_n=ensemble_n,
                    backshift_parsing=True,
                    backshift_ctx_len=backshift_len,
                    backshift_break_at_phrase=True
                )
            
            iterated_times = 0
            for nb_iter in self.hyperparams.nb_train_iterations:
                for _ in range(nb_iter - iterated_times):
                    train_spa_oneIter(X_train, y_train, [spa0, spa1])
                iterated_times = nb_iter

                val_metric_and_gamma = []
                for gamma in self.hyperparams.gamma:
                    spa0.set_inference_config(gamma=gamma)
                    spa1.set_inference_config(gamma=gamma)
                    val_metric_value = test_seq(
                        X_val, y_val,
                        [spa0, spa1],
                        n_threads=self.inf_threads,
                    )[self.val_metric]

                    val_metric_and_gamma.append((val_metric_value, gamma))
                    print(f"Label {label_idx} | max_depth {max_depth} | iter {nb_iter} | gamma {gamma} | {self.val_metric}: {val_metric_value:.4f}", flush=True)
                best_gamma, best_val_metric_value = max(val_metric_and_gamma, key=lambda x: x[0])
                spa0.set_inference_config(gamma=best_gamma)
                spa1.set_inference_config(gamma=best_gamma)
                if best_val_metric_value > best_val_metric or best_val_metric == 0:
                    best_val_metric = best_val_metric_value
                    spa0_path, spa1_path = self.model_paths[label_idx]
                    with open(spa0_path, "wb") as f:
                        f.write(spa0.to_bytes())
                    with open(spa1_path, "wb") as f:
                        f.write(spa1.to_bytes())


    def _test_one_label(
        self, label_idx: int
    ):
        print(f"[Testing label {label_idx}]")
        test_data = pd.read_csv(self.test_csv, dtype={"label": str})
        # Remove Ns from each sequence
        test_data["sequence"] = test_data["sequence"].str.replace("N", "", regex=False)

        X_test = [
            Sequence(s, charmap=CharacterMap("ACGT")) for s in test_data["sequence"]
        ]
        y_test = [x[label_idx] for x in bitstr_to_array(test_data["label"])]

        del test_data
        gc.collect()

        spa0_path, spa1_path = self.model_paths[label_idx]
        try:
            spa0 = spa_from_file(spa0_path)
        except Exception as e:
            print(f"Error loading spa0 for label {label_idx} from {spa0_path}: {e}")
            spa0 = LZ78SPA(alphabet_size=4)  # Dummy spa to avoid crash
        try:
            spa1 = spa_from_file(spa1_path)
        except Exception as e:
            print(f"Error loading spa1 for label {label_idx} from {spa1_path}: {e}")
            spa1 = LZ78SPA(alphabet_size=4)  # Dummy spa to avoid crash

        results = test_seq(
            X_test, y_test,
            [spa0, spa1],
            n_threads=self.inf_threads,
        )
        # print(f"Label {label_idx} test results: {results}")
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
    parser.add_argument("--gamma", type=float, nargs='+', default=[0.1, 0.5, 1, 3, 5],
                        help="List of gamma values to try")
    parser.add_argument("--nb_train_iterations", type=int, nargs='+', default=[1],
                        help="List of number of training iterations to try")
    parser.add_argument("--max_depth", type=int, nargs='+', default=[12],
                        help="List of max_depth values to try")
    parser.add_argument("--just_test", action="store_true",
                        help="If set, only run tests without training")
    args = parser.parse_args()

    hyperparams = HyperparameterSweep(
        gamma=args.gamma,
        nb_train_iterations=args.nb_train_iterations,
        max_depth=args.max_depth
    )

    classifier = MultiLabelClassifier(
        n_labels=args.n_labels,
        data_dir=args.data_dir,
        hyperparams=hyperparams,
        spa_dir=args.spa_dir,
        val_metric=args.val_metric,
        inf_threads=args.inf_threads,
        job_threads=args.job_threads,
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