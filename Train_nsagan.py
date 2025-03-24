from lz78 import Sequence, LZ78SPA, CharacterMap, spa_from_file
import pandas as pd
import numpy as np
from multiprocessing import Pool, Value, Lock
from sys import stdout, stderr
import os
from tqdm import tqdm
from transformers import HfArgumentParser
from dataclasses import dataclass, field

@dataclass
class Args:
    dataset_dir: str = field()
    out_dir: str = field()

GAMMAS = [1/3, 1/2, 1, 1.5, 2, 2.5, 3]
EPOCHS = 10
ENSEMBLE_TYPES = ["depth", "entropy"]
ALPHABET_SIZE = 4
N_THREADS = 64

def classify(data: pd.DataFrame, spas: list[LZ78SPA], n_threads=32):
    labels = data["label"]
    data = [Sequence(seq, charmap=CharacterMap("ACGT")) for seq in data["sequence"]]
    log_losses = np.zeros((len(spas), len(data)))
    for i in range(len(spas)):
        log_losses[i, :] = [res["avg_log_loss"] for res in spas[i].compute_test_loss_parallel(data, num_threads=n_threads)]
    classes = np.argmin(log_losses, axis=0)
    return (classes == labels).sum() / len(labels)


def train_spa_oneIter(data: pd.DataFrame, spas: list[LZ78SPA]):
    grouped_data = data.groupby("label")["sequence"]
    for (label, data) in grouped_data:
        for seq in data:
            spas[label].train_on_block(Sequence(seq, charmap=CharacterMap("ACGT")))


def main(dataset: str, out_dir: str):
    print("="*60, file=stderr)
    print(f"Dataset: {dataset}", file=stderr)
    print("="*60, file=stderr)
    train_path = f"{dataset}/train.csv"
    val_path = f"{dataset}/dev.csv"
    test_path = f"{dataset}/test.csv"

    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    unique_labels = sorted(train_data['label'].unique())
    n = len(unique_labels)

    for i in train_data.index:
        train_data.loc[i, "sequence"] = "".join([x for x in train_data.loc[i, "sequence"] if x in "ACGT"])
    for i in validation_data.index:
        validation_data.loc[i, "sequence"] =  "".join([x for x in validation_data.loc[i, "sequence"] if x in "ACGT"])
    for i in test_data.index:
        test_data.loc[i, "sequence"] =  "".join([x for x in test_data.loc[i, "sequence"] if x in "ACGT"])

    spas = [LZ78SPA(alphabet_size=4, gamma=1, compute_training_loss=False) for _ in range(n)]
    for i in range(n):
        spas[i].set_inference_config(
            lb=1e-5,
            ensemble_type="depth",
            ensemble_n=10,
            backshift_parsing=True,
            backshift_ctx_len=20,
            backshift_break_at_phrase=True
        )

    best_gamma = 0
    best_epoch = 0
    best_ensemble = None
    best_acc = 0

    print("Training...", file=stderr)
    for epoch in tqdm(range(EPOCHS), file=stderr):
        stdout.flush()
        train_spa_oneIter(train_data, spas)

        print("Saving...", file=stderr)
        os.makedirs(f"{out_dir}/epoch{epoch}", exist_ok=True)
        for i in range(n):
            spas[i].to_file(f"{out_dir}/epoch{epoch}/spa{i}.bin")    

        best_gamma_epoch = 0
        best_acc_epoch = 0
        best_ensemble_epoch = None

        print("Validating...", file=stderr)
        for gamma in GAMMAS:
            for ensemble in ENSEMBLE_TYPES:
                for i in range(n):
                    spas[i].set_inference_config(gamma=gamma, ensemble_type=ensemble)
            
                acc = classify(validation_data, spas, N_THREADS)
                print(f"GAMMA={gamma}, ENSEMBLE={ensemble}, VAL_ACC={acc}")

                if acc > best_acc_epoch:
                    best_acc_epoch = acc
                    best_gamma_epoch = gamma
                    best_ensemble_epoch = ensemble

        print(f"BEST_GAMMA_EPOCH={best_gamma_epoch}, BEST_ENSEMBLE_EPOCH={best_ensemble_epoch}, BEST_VAL_ACC_EPOCH={best_acc_epoch}")

        if best_acc_epoch > best_acc:
            best_acc = best_acc_epoch
            best_gamma = best_gamma_epoch
            best_epoch = epoch
            best_ensemble = best_ensemble_epoch

    print("Testing...", file=stderr)
    for i in range(n):
        spas[i] = spa_from_file(f"{out_dir}/epoch{best_epoch}/spa{i}.bin")
        spas[i].set_inference_config(gamma=best_gamma, ensemble_type=best_ensemble)

    print(f"\nBEST_EPOCH={best_epoch}, BEST_GAMMA={best_gamma}, BEST_ENSEMBLE={best_ensemble}")
    acc = classify(test_data, spas, N_THREADS)
    print(f"TEST_ACC={acc}")

if __name__ == "__main__":
    args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
    main(args.dataset_dir, args.out_dir)