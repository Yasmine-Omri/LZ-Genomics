'''
- Loads one master CSV with folds (fold, seq, label). The seq is either the 128bp window or the window with += 256bp context (flanks)
- performs training and 10-fold CV using the AUPRC metric
- this training and 10-fold CV is wrapped in a training sweep

'''

from lz78 import Sequence, CharacterMap, LZ78SPA
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
#from ushuffle import shuffle, Shuffler
from sklearn.metrics import average_precision_score  # UPDATED

# Training hyperparameters
INCLUDE_PREV_CONTEXT = None
GAMMA = None
NB_TRAIN_ITERATIONS = None
HANDLE_N_SETTING = None 
RATIO_PRETRAIN_TRAIN = None
ENSEMBLE_TYPE = None
NUM_THREADS = None
MAX_DEPTHS = None

def parse_set(input_str):
    input_str = input_str.strip("{}")
    parsed_set = set()
    for item in input_str.split(","):
        item = item.strip()
        try:
            parsed_set.add(float(item))
        except ValueError:
            parsed_set.add(item)
    return parsed_set

def parse_bool(input_str):
    input_str = input_str.strip("{}")
    items = input_str.split(",")
    return {item.strip() == "True" for item in items}

# Pretraining SPA
def pretrain_spa(seq, spa: list[LZ78SPA], nb_pretrain_symbols):
    global INCLUDE_PREV_CONTEXT
    if nb_pretrain_symbols == 0:
        return
    elements = seq.splitlines()
    len_pretrain_seq = len(elements[0])
    nb_pretrain_seqs = math.ceil(nb_pretrain_symbols / len_pretrain_seq)
    selected_elements = elements[:nb_pretrain_seqs]
    selected_elements[-1] = selected_elements[-1][0: nb_pretrain_symbols % len_pretrain_seq]
    for element in selected_elements:
        encoded_seq = Sequence(element, charmap=CharacterMap("ACGT"))
        seq_len = len(element)
        if (seq_len == 0):
            return
        for index in range(len(spa)):
            if not INCLUDE_PREV_CONTEXT:
                spa[index].reset_state()
            spa[index].train_on_block(encoded_seq)

def train_spa_oneIter(data, spa):
    global INCLUDE_PREV_CONTEXT
    for row in data.itertuples():
        seq = row[1]
        label = row[2]
        encoded_seq = Sequence(seq, charmap=CharacterMap("ACGT"))
        if not INCLUDE_PREV_CONTEXT:
            spa[label].reset_state()
        spa[label].train_on_block(encoded_seq)

def train_spa(data, spa, iterations):
    global INCLUDE_PREV_CONTEXT
    for i in range(iterations):
        for row in data.itertuples():
            seq = row[1]
            label = row[2]
            encoded_seq = Sequence(seq, charmap=CharacterMap("ACGT"))
            if not INCLUDE_PREV_CONTEXT:
                spa[label].reset_state()
            spa[label].train_on_block(encoded_seq)

def test_seq(data: pd.DataFrame, spas: list[LZ78SPA], metric, n_threads=32):
    labels = data["label"].to_numpy()
    seqs = [Sequence(seq, charmap=CharacterMap("ACGT")) for seq in data["sequence"]]
    log_losses = np.zeros((len(spas), len(seqs)))
    for i in range(len(spas)):
        log_losses[i, :] = [res["avg_log_loss"] for res in spas[i].compute_test_loss_parallel(seqs, num_threads=n_threads)]
    classes = np.argmin(log_losses, axis=0)

    if metric == "accuracy":
        return (classes == labels).sum() / len(labels)
    if metric == "mcc":
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(labels, classes)
    if metric == "f1":
        from sklearn.metrics import f1_score
        return f1_score(labels, classes, average='weighted')
    if metric == "auprc":  # UPDATED
        # Treat lower log-loss = higher confidence for that class
        scores = -log_losses[1, :]  # use SPA1 score for positive class
        return average_precision_score(labels, scores)
    else:
        raise ValueError("Invalid metric specified.")

def process_sequence(sequence, setting="remove", n=10):
    if setting == "remove":
        return ["".join(char for char in sequence if char in "ACGT")]
    elif setting == "random":
        return ["".join(random.choice("ACGT") if char not in "ACGT" else char for char in sequence)]
    elif setting == "expand":
        expanded = []
        for _ in range(n):
            new_seq = "".join(random.choice("ACGT") if char not in "ACGT" else char for char in sequence)
            expanded.append(new_seq)
        return expanded
    else:
        raise ValueError("Setting must be 'remove', 'random', or 'expand'.")

def handle_N(data, setting="remove"):
    new_data = []
    for _, row in data.iterrows():
        sequence, label = row['sequence'], row['label']
        processed = process_sequence(sequence, setting)
        for seq in processed:
            new_data.append({"sequence": seq, "label": label})
    return pd.DataFrame(new_data)

@profile
def main(dataset_file, pretrain_file, metric):
    global INCLUDE_PREV_CONTEXT, GAMMA, NB_TRAIN_ITERATIONS
    global HANDLE_N_SETTING, RATIO_PRETRAIN_TRAIN, ENSEMBLE_TYPE
    global NUM_THREADS, MAX_DEPTH

    global include_prev_contexts
    global gammas 
    global nb_train_iterations 
    global handle_N_settings 
    global ratio_pretrain_train
    global ensemble_type
    global num_threads
    global augmentation_factors
    global preserve_kmer
    global max_depth


    #UPDATED: read single CSV with fold column
    full_data = pd.read_csv(dataset_file) #added the str , dtype={'sequence': str}
    folds = sorted(full_data['fold'].unique())

    with open(pretrain_file, 'r') as file:
        pretrain_data = file.read()

    results_df = pd.DataFrame()

    print("-----TRAINING")
    print("---HYPERPARAM SWEEP WITH 10-FOLD CV")

    train_start_time = time.perf_counter()  # UPDATED: profiling start
   
    for include_prev_context, handle_N_setting, ratio, aug_factor, max_depth in itertools.product(
        include_prev_contexts, handle_N_settings, ratio_pretrain_train, augmentation_factors, max_depth
    ):

        INCLUDE_PREV_CONTEXT = include_prev_context
        HANDLE_N_SETTING = handle_N_setting
        RATIO_PRETRAIN_TRAIN = ratio
        ENSEMBLE_TYPE = ensemble_type
        NUM_THREADS = num_threads
        MAX_DEPTH = max_depth

        for heldout in folds:  # 10-fold CV
            train_data = full_data[full_data['fold'] != heldout].copy()
            val_data   = full_data[full_data['fold'] == heldout].copy()
            train_data = handle_N(train_data, setting=HANDLE_N_SETTING)
            val_data   = handle_N(val_data, setting=HANDLE_N_SETTING)

            spa = [LZ78SPA(alphabet_size=4, compute_training_loss=False,
                        max_depth=int(MAX_DEPTH) if MAX_DEPTH else None) for _ in [0,1]]

            nb_train_symbols = len(train_data) * train_data["sequence"].str.len().iloc[0]
            nb_pretrain_symbols = math.ceil(RATIO_PRETRAIN_TRAIN * nb_train_symbols)
            pretrain_spa(pretrain_data, spa, nb_pretrain_symbols)

            iterated_times = 0
            for nb_iterations in sorted(nb_train_iterations):
                # Train further up to this nb_iterations milestone
                for _ in range(nb_iterations - iterated_times):
                    train_spa_oneIter(train_data, spa)
                iterated_times = nb_iterations

                for gamma in gammas:
                    for ensemble in ENSEMBLE_TYPE:
                        for i in range(len(spa)):
                            spa[i].set_inference_config(gamma=gamma, ensemble_type=ensemble)

                        inference_start_time = time.perf_counter()
                        val_score = test_seq(val_data, spa, metric=metric, n_threads=NUM_THREADS)
                        inference_end_time = time.perf_counter()
                        inference_duration = inference_end_time - inference_start_time

                        # Print *every hyperparam combo, every fold*
                        print(f"Fold={heldout}, Iter={nb_iterations}, Gamma={gamma}, "
                            f"Ensemble={ensemble}, Context={INCLUDE_PREV_CONTEXT}, "
                            f"Ratio={RATIO_PRETRAIN_TRAIN}, MaxDepth={MAX_DEPTH}, "
                            f"AUPRC={val_score:.4f}", flush=True)

                        # Save one row per combo per fold
                        current = pd.DataFrame([{
                            "FOLD": heldout,
                            "INCLUDE_PREV_CONTEXT": INCLUDE_PREV_CONTEXT,
                            "GAMMA": gamma,
                            "NB_TRAIN_ITERATIONS": nb_iterations,
                            "HANDLE_N_SETTING": HANDLE_N_SETTING,
                            "RATIO_PRETRAIN_TRAIN": RATIO_PRETRAIN_TRAIN,
                            "ENSEMBLE_TYPE": ensemble,
                            "MAX_DEPTH": MAX_DEPTH if MAX_DEPTH else 0,
                            "NUM_THREADS": NUM_THREADS,
                            "VALIDATION METRIC": val_score
                        }])
                        results_df = pd.concat([results_df, current], ignore_index=True)

    # Now average across folds per hyperparam combo
    avg_results = results_df.groupby([
        "INCLUDE_PREV_CONTEXT", "GAMMA", "NB_TRAIN_ITERATIONS", "HANDLE_N_SETTING",
        "RATIO_PRETRAIN_TRAIN", "ENSEMBLE_TYPE", "MAX_DEPTH", "NUM_THREADS"
    ])["VALIDATION METRIC"].mean().reset_index()

    print("avg results", avg_results)
    best_row = avg_results.loc[avg_results['VALIDATION METRIC'].idxmax()]
    print("Best hyperparameters:", best_row.to_dict())


    train_end_time = time.perf_counter()  # UPDATED: profiling end
    train_duration = train_end_time - train_start_time

    print("Best hyperparameters:", best_row.to_dict())
    #UPDATED: Save only best SPAs
    # (Retrain with full dataset and save)
    best_params = best_row.to_dict()
    spa = [LZ78SPA(alphabet_size=4, gamma=best_params["GAMMA"],
                   compute_training_loss=False,
                   max_depth=int(best_params["MAX_DEPTH"])) for _ in [0,1]]
    # UPDATED: save best SPAs with dataset-based filename
    dataset_base = os.path.splitext(os.path.basename(dataset_file))[0]
    for i, sp in enumerate(spa):
        spa_bytes = bytearray(sp.to_bytes())
        makedirs("best_spas", exist_ok=True)
        path = os.path.join("best_spas", f"{dataset_base}_best_label{i}.bin")
        with open(path, 'wb') as f:
            f.write(spa_bytes)
    print(f"Saved best SPAs to {dataset_base}_best_label*.bin")

    print("-----TIME PROFILING+")
    print(f"Number of training symbols: {nb_train_symbols}")
    #print(f"Length of one training sequence: {len(train_data.iloc[0, 1])}")
    print(f"Length of one training sequence: {train_data['sequence'].str.len().iloc[0]}")

    print(f"Total training time: {train_duration:.3f} seconds")
    print(f"Number of test sequences: {len(val_data)}")
    #print(f"Length of test sequence: {len(len_val_data.iloc[0, 1])}")
    print(f"Length of test sequence: {val_data['sequence'].str.len().iloc[0]}")

    print(f"Total inference time: {inference_duration:.3f} seconds")
    #print(f"Inference time/symbol: {inference_duration/(len(len_val_data) * len(len_val_data.iloc[0, 1]))} seconds")
    print(f"Inference time/symbol: {inference_duration/(len(val_data) * val_data['sequence'].str.len().iloc[0])} seconds")

    print("-----MEMORY REPORT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_file", type=str, required=True,
                        help="CSV with fold,sequence,label")
    parser.add_argument("-pretrain_file", type=str, required=True,
                        help="Path to pretraining file")
    parser.add_argument("--include_prev_context", type=str, required=True)
    parser.add_argument("--gamma", type=str, required=True)
    parser.add_argument("--nb_train_iterations", type=str, required=True)
    parser.add_argument("--handle_n_setting", type=str, required=False, default="{remove}")
    parser.add_argument("--ratio_pretrain_train", type=str, required=True)
    parser.add_argument("--ensemble_type", type=str, required=True)
    parser.add_argument("--num_threads", type=str, required=True)
    parser.add_argument("--augmentation_factors", type=str, required=False, default="{0}")
    parser.add_argument("--shuffle_preserve_kmer", type=int, default=2)
    parser.add_argument("--max_depth", type=str, required=False, default="{}")
    parser.add_argument("--metric", type=str, default="auprc",  # UPDATED default
                        choices=["accuracy","mcc","f1","auprc"])
    args = parser.parse_args()

    include_prev_contexts = parse_bool(args.include_prev_context)
    gammas = parse_set(args.gamma)
    nb_train_iterations = {int(x) for x in parse_set(args.nb_train_iterations)}
    #handle_N_settings = {args.handle_n_setting}
    handle_N_settings = {"remove"} #harcoding remove
    ratio_pretrain_train = parse_set(args.ratio_pretrain_train)
    ensemble_type = parse_set(args.ensemble_type)
    augmentation_factors = parse_set(args.augmentation_factors)
    preserve_kmer = args.shuffle_preserve_kmer
    num_threads = int(list(parse_set(args.num_threads))[0])
    max_depth = [None] + [int(x) for x in list(parse_set(args.max_depth))]

    main(args.dataset_file, args.pretrain_file, args.metric)
