'''
SAME SCRIPT FROM GITHUB REBUTTAL 

This script is used to run the pre-train, train, validate, test framework for the LZ78-based classifier for a given dataset.
The framework is highly configurable and outputs a detailed report including accuracy numbers and time/memory profiling.

INPUTS:
- Labeled dataset path
- Unlabeled data for the optional pre-training phase
- Hyperparameter values to consider for the hyperparameter sweep

OUTPUTS:
- Detailed printed report including: 
    * VALIDATION METRIC for each combination of hyperparameters tested
    * Hyperparameter Combination producing the highest VALIDATION METRIC
    * Test metric (on test dataset) of the best SPAs
    * Depth of the trees corresponding to the best SPAs
    * Computational metrics
- Best SPAs (highest VALIDATION METRIC) saved as .bin files to be used for inference or further analysis.

EXAMPLE USAGE:

python Train.py -dataset_folder "$DATASET_FOLDER" -pretrain_file "$PRETRAIN_FILE" --include_prev_context "{False}" --gamma "{0.1, 0.33, 0.5, 0.75, 1, 3, 5}" --nb_train_iterations "{1, 3, 5, 7, 10}" --ratio_pretrain_train "{0}"\ --handle_n_setting "{remove}" --ensemble_type "{entropy}" --num_threads "{64}" > "$OUTPUT_DIR/$OUTPUT_FILE"
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
from ushuffle import shuffle, Shuffler

# NEW (minimal): for per-label parallelization and AUROC
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import multiprocessing

# Training hyperparameters:
# The hyperparameters defining the hyperparameter search space are passed as command line arguments in the USAGE command specified above
# Alternatively, the hyperparameter values to consider could be defined below.
INCLUDE_PREV_CONTEXT = None
GAMMA = None
NB_TRAIN_ITERATIONS = None
HANDLE_N_SETTING = None 
RATIO_PRETRAIN_TRAIN = None # nb of pretrained sequences / nb train sequences
ENSEMBLE_TYPE = None
NUM_THREADS = None
MAX_DEPTHS = None

import argparse


def parse_set(input_str):
    """
    Converts a comma-separated string into a Python set.
    Example: "{0.1, 0.5, 1.0}" -> {0.1, 0.5, 1.0}
    """
    input_str = input_str.strip("{}")  # Remove the surrounding braces
    parsed_set = set()
    
    # Iterate over the split string and try to convert each item to float if possible
    for item in input_str.split(","):
        item = item.strip()  # Remove any extra spaces
        try:
            # Try converting to float and add to set
            parsed_set.add(float(item))
        except ValueError:
            # If conversion fails, add the item as is (assuming it's not a float)
            parsed_set.add(item)
    
    return parsed_set


def parse_bool(input_str):
    # Strip the braces and split the string into components
    input_str = input_str.strip("{}")
    items = input_str.split(",")
    # Trim and convert to appropriate types
    return {item.strip() == "True" for item in items}

#Pretraining the spa routine
def pretrain_spa(seq, spa: list[LZ78SPA], nb_pretrain_symbols):
    global INCLUDE_PREV_CONTEXT
    if nb_pretrain_symbols == 0:
        return
    # Split the sequence into individual elements by newline
    elements = seq.splitlines()
    
    # Determine the number of elements to use based on the specified percentage
    # num_elements = int(len(elements) * (percentage / 100))
    len_pretrain_seq = len(elements[0]) if elements else 0
    if len_pretrain_seq == 0:
        return
    nb_pretrain_seqs = math.ceil(nb_pretrain_symbols / len_pretrain_seq) if len_pretrain_seq else 0

    selected_elements = elements[:nb_pretrain_seqs] if nb_pretrain_seqs > 0 else []
    if selected_elements:
        selected_elements[-1] = selected_elements[-1][0: nb_pretrain_symbols % len_pretrain_seq]

    for element in selected_elements:
        # Encode each individual element
        encoded_seq = Sequence(element, charmap=CharacterMap("ACGT"))
        seq_len = len(element)

        if (seq_len == 0):
            print("Warning: seq_len = 0 during pretraining. Exited pretrain.")
            return
        # Compute log-loss for each label's SPA
        for index in range(len(spa)):
            if not INCLUDE_PREV_CONTEXT:
                spa[index].reset_state()
            spa[index].train_on_block(encoded_seq)


def train_spa_oneIter(data, spa):
    global INCLUDE_PREV_CONTEXT
    logloss_per_label = [[] for _ in range(len(spa))]  # List of lists for log-losses per label index
    for row in data.itertuples():
        seq = row[1]
        label = row[2]
        
        # Encode sequence
        encoded_seq = Sequence(seq, charmap=CharacterMap("ACGT"))
        seq_len = len(seq)
        # Compute log-loss for the respective label's SPA
        if not INCLUDE_PREV_CONTEXT:
            spa[label].reset_state()
        spa[label].train_on_block(encoded_seq)
        
    return logloss_per_label

def train_spa(data, spa, iterations):
    global INCLUDE_PREV_CONTEXT
    logloss_per_label = [[] for _ in range(len(spa))]  # List of lists for log-losses per label index
    for i in range(iterations):
        for row in data.itertuples():
            seq = row[1]
            label = row[2]
            
            # Encode sequence
            encoded_seq = Sequence(seq, charmap=CharacterMap("ACGT"))
            seq_len = len(seq)
            # Compute log-loss for the respective label's SPA
            if not INCLUDE_PREV_CONTEXT:
                spa[label].reset_state()
            spa[label].train_on_block(encoded_seq)
            
    return logloss_per_label

# NEW (minimal): AUROC support for binary (2 SPA) evaluation; keep multiclass path intact
def test_seq (data: pd.DataFrame, spas: list[LZ78SPA], metric, n_threads=32):
    # for every test seq,
    # run it through all spas
    # classification = label associated with lowest loss spa
    # check classification against ground truth
    # compute metric (of all test runs)
    labels = data["label"].to_numpy()
    seqs = [Sequence(seq, charmap=CharacterMap("ACGT")) for seq in data["sequence"]]

    if metric == "auroc":
        assert len(spas) == 2, "For AUROC we expect exactly 2 SPAs (class 0 and class 1)."
        loss0 = np.array([res["avg_log_loss"] for res in spas[0].compute_test_loss_parallel(seqs, num_threads=n_threads)])
        loss1 = np.array([res["avg_log_loss"] for res in spas[1].compute_test_loss_parallel(seqs, num_threads=n_threads)])
        scores = loss0 - loss1  # higher => more likely class 1
        return roc_auc_score(labels, scores)

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
    else:
        raise ValueError("Invalid metric specified. Choose from 'accuracy', 'mcc', 'f1', or 'auroc'.")


#Processes a sequence for its placeholders
def process_sequence(sequence, setting="remove", n=10):
    if setting == "remove":
        # Remove all characters that are not A, C, G, or T
        return ["".join(char for char in sequence if char in "ACGT")]
    
    elif setting == "random":
        # Replace each character that is not A, C, G, or T with a random nucleotide (A, C, G, or T)
        return ["".join(random.choice("ACGT") if char not in "ACGT" else char for char in sequence)]
    
    elif setting == "expand":
        # Generate 'n' sequences by replacing non-ACGT characters with random nucleotides
        expanded_sequences = []
        for _ in range(n):
            new_sequence = "".join(random.choice("ACGT") if char not in "ACGT" else char for char in sequence)
            expanded_sequences.append(new_sequence)
        return expanded_sequences
    else:
        raise ValueError("Setting must be 'remove', 'random', or 'expand'.")

#Handles the placeholder "N"s in a sequence
def handle_N(data, setting="remove"):
    new_data = []
    for _, row in data.iterrows():
        sequence, label = row['sequence'], row['label']
        processed_sequences = process_sequence(sequence, setting)
        
        for proc_seq in processed_sequences:
            new_data.append({"sequence": proc_seq, "label": label})

    return pd.DataFrame(new_data)

# NEW (minimal): exact product constraint -> choose (label_jobs, threads_per_job)
def choose_label_jobs_and_threads(total_threads: int, num_labels: int):
    """
    Returns (label_jobs, threads_per_job) such that:
      - label_jobs * threads_per_job == total_threads
      - 1 <= label_jobs <= num_labels
      - threads_per_job >= 1
    Picks the largest feasible label_jobs (more concurrency across labels).
    """
    assert total_threads >= 1
    divisors = sorted({d for d in range(1, total_threads + 1) if total_threads % d == 0})
    cap = max(1, min(num_labels, total_threads))
    feas = [d for d in divisors if d <= cap]
    label_jobs = feas[-1] if feas else 1
    threads_per_job = total_threads // label_jobs
    return label_jobs, threads_per_job

# NEW (minimal): per-label sweep using existing train helpers; returns best params per label
def run_sweep_for_label(
    label_idx: int,
    X_train: list[str], y_train_k: np.ndarray,
    X_val:   list[str], y_val_k:   np.ndarray,
    pretrain_text: str,
    include_prev_contexts, gammas, nb_train_iterations,
    handle_N_settings, ratio_pretrain_train,
    ensemble_type, augmentation_factors, preserve_kmer,
    max_depth_list, alphabet_size: int,
    threads_per_job: int,
    metric: str = "auroc"
):
    """
    Returns (label_idx, best_val_auc, best_params_dict)
    """
    global INCLUDE_PREV_CONTEXT

    # Build lightweight DataFrames expected by helpers
    train_df_base = pd.DataFrame({"sequence": X_train, "label": y_train_k}, copy=False)
    val_df_base   = pd.DataFrame({"sequence": X_val,   "label": y_val_k},   copy=False)

    best_val = -1.0
    best_params = None

    # Precompute sizes for pretraining budget
    seq_len = len(X_train[0]) if len(X_train) else 0
    nb_train_symbols_total = len(X_train) * seq_len

    print(f"[Label {label_idx}] sweep start")

    for include_prev_context, handle_N_setting, ratio, aug_factor, max_depth in itertools.product(
        include_prev_contexts, handle_N_settings, ratio_pretrain_train, augmentation_factors, max_depth_list
    ):
        INCLUDE_PREV_CONTEXT = include_prev_context  # used by pretrain/train helpers

        # Materialize per-setting data (N handling + augmentation)
        train_df = handle_N(train_df_base, setting=handle_N_setting)
        val_df   = handle_N(val_df_base,   setting=handle_N_setting)

        # Augment: shuffle ONLY positives (label==1), add as NEGATIVE (label=0)
        pos_train = train_df[train_df['label'] == 1]
        new_negs = []
        for s in pos_train["sequence"]:
            s_bytes = s.lower().encode('utf-8')
            shuffler = Shuffler(s_bytes, preserve_kmer)
            if aug_factor < 1:
                if random.random() < aug_factor:
                    new_negs.append([shuffler.shuffle().decode('utf-8').upper(), 0])
                continue
            for _ in range(int(aug_factor)):
                new_negs.append([shuffler.shuffle().decode('utf-8').upper(), 0])
        if new_negs:
            train_df = pd.concat([train_df, pd.DataFrame(new_negs, columns=['sequence','label'])], ignore_index=True)

        # Two SPAs (class 0 and class 1)
        spa0 = LZ78SPA(alphabet_size=alphabet_size, compute_training_loss=False, max_depth=int(max_depth) if max_depth else None)
        spa1 = LZ78SPA(alphabet_size=alphabet_size, compute_training_loss=False, max_depth=int(max_depth) if max_depth else None)
        for sp in (spa0, spa1):
            sp.set_inference_config(
                lb=1e-5,
                ensemble_type="entropy",
                ensemble_n=10,
                backshift_parsing=True,
                backshift_ctx_len=20,
                backshift_break_at_phrase=True
            )

        # Pretrain
        nb_pretrain_symbols = math.ceil(ratio * nb_train_symbols_total) if seq_len else 0
        pretrain_spa(pretrain_text, [spa0, spa1], nb_pretrain_symbols)

        # Train (iterative)
        iterated_times = 0
        train_one_iter_start_time = time.perf_counter()
        for nb_iter in nb_train_iterations:
            for _ in range(nb_iter - iterated_times):
                train_spa_oneIter(train_df, [spa0, spa1])
            iterated_times = nb_iter

            # Validate over gamma x ensemble
            for gamma in gammas:
                for ensemble in ensemble_type:
                    spa0.set_inference_config(gamma=gamma, ensemble_type=ensemble)
                    spa1.set_inference_config(gamma=gamma, ensemble_type=ensemble)
                    val_metric = test_seq(val_df, [spa0, spa1], metric=metric, n_threads=threads_per_job)

                    train_one_iter_end_time = time.perf_counter()
                    train_one_iter_duration = train_one_iter_end_time - train_one_iter_start_time

                    # Mirror your print style
                    print(f"[Label {label_idx}] {nb_iter}, {aug_factor}, {gamma}, {include_prev_context}, {handle_N_setting}, {ratio}, {ensemble}, {max_depth}, {threads_per_job}, {train_one_iter_duration:.3f}, {(val_metric * 100):.2f}", flush=True)

                    if val_metric > best_val:
                        best_val = val_metric
                        best_params = {
                            "label_idx": label_idx,
                            "INCLUDE_PREV_CONTEXT": include_prev_context,
                            "GAMMA": float(gamma),
                            "NB_TRAIN_ITERATIONS": int(nb_iter),
                            "HANDLE_N_SETTING": handle_N_setting,
                            "RATIO_PRETRAIN_TRAIN": float(ratio),
                            "ENSEMBLE_TYPE": ensemble,
                            "MAX_DEPTH": int(max_depth) if max_depth else 0,
                            "THREADS_PER_JOB": threads_per_job,
                            "AUGMENTATION_FACTOR": float(aug_factor),
                        }

    print(f"[Label {label_idx}] best val AUROC = {best_val:.4f}")
    return label_idx, best_val, best_params

# NEW (minimal): retrain with best params per label, compute test AUROC, and save two SPAs
def retrain_and_test_for_label(
    label_idx: int, best_params: dict,
    X_train: list[str], y_train_k: np.ndarray,
    X_test:  list[str], y_test_k:  np.ndarray,
    pretrain_text: str, alphabet_size: int,
    threads_per_job: int, dataset_folder: str
):
    global INCLUDE_PREV_CONTEXT

    INCLUDE_PREV_CONTEXT = best_params["INCLUDE_PREV_CONTEXT"]
    gamma = best_params["GAMMA"]
    nb_iter = best_params["NB_TRAIN_ITERATIONS"]
    handle_N_setting = best_params["HANDLE_N_SETTING"]
    ratio = best_params["RATIO_PRETRAIN_TRAIN"]
    ensemble = best_params["ENSEMBLE_TYPE"]
    max_depth = best_params["MAX_DEPTH"] if best_params["MAX_DEPTH"] != 0 else None

    # Data views
    train_df_base = pd.DataFrame({"sequence": X_train, "label": y_train_k}, copy=False)
    test_df_base  = pd.DataFrame({"sequence": X_test,  "label": y_test_k},  copy=False)

    train_df = handle_N(train_df_base, setting=handle_N_setting)
    test_df  = handle_N(test_df_base,  setting=handle_N_setting)

    # Two SPAs
    spa0 = LZ78SPA(alphabet_size=alphabet_size, gamma=gamma, compute_training_loss=False, max_depth=max_depth)
    spa1 = LZ78SPA(alphabet_size=alphabet_size, gamma=gamma, compute_training_loss=False, max_depth=max_depth)
    for sp in (spa0, spa1):
        sp.set_inference_config(
            lb=1e-5,
            ensemble_type=ensemble,
            ensemble_n=10,
            backshift_parsing=True,
            backshift_ctx_len=20,
            backshift_break_at_phrase=True
        )

    # Pretrain
    seq_len = len(X_train[0]) if len(X_train) else 0
    nb_train_symbols_total = len(X_train) * seq_len
    nb_pretrain_symbols = math.ceil(ratio * nb_train_symbols_total) if seq_len else 0
    pretrain_spa(pretrain_text, [spa0, spa1], nb_pretrain_symbols)

    # Train
    for _ in range(nb_iter):
        train_spa_oneIter(train_df, [spa0, spa1])

    # Test AUROC
    test_auc = test_seq(test_df, [spa0, spa1], metric="auroc", n_threads=threads_per_job)

    # Save SPAs
    makedirs("best_spas", exist_ok=True)
    base_name = dataset_folder.split("data/", 1)[-1].replace("/", "_")
    b0 = bytearray(spa0.to_bytes()); b1 = bytearray(spa1.to_bytes())
    with open(os.path.join("best_spas", f"{base_name}_label{label_idx}_class0_short.bin"), "wb") as f:
        f.write(b0)
    with open(os.path.join("best_spas", f"{base_name}_label{label_idx}_class1_short.bin"), "wb") as f:
        f.write(b1)
    print(f"[Label {label_idx}] saved SPAs: class0 {len(b0)/(1024*1024):.2f} MB, class1 {len(b1)/(1024*1024):.2f} MB")

    return label_idx, float(test_auc)

@profile
def main(dataset_folder, pretrain_file, metric):
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
    global NUM_THREADS

    read_data_in_time = time.perf_counter()
    
    # Read train, val, test data 
    train_path = f"{dataset_folder}/train.csv"
    val_path   = f"{dataset_folder}/dev.csv"
    test_path  = f"{dataset_folder}/test.csv"
    
    # the labels are bitstrings, e.g., "01001" for 5 labels
    train_data = pd.read_csv(train_path, dtype={"label": str})
    validation_data = pd.read_csv(val_path, dtype={"label": str})
    test_data = pd.read_csv(test_path, dtype={"label": str})
    
    ALPHABET_SIZE = 4

    # ---- NEW (minimal): parse bitstring labels into matrices; sequences into lists
    def bitstr_to_array(col):
        return np.array([list(map(int, list(s))) for s in col], dtype=np.uint8)

    Y_train = bitstr_to_array(train_data["label"])
    Y_val   = bitstr_to_array(validation_data["label"])
    Y_test  = bitstr_to_array(test_data["label"])

    X_train = train_data["sequence"].tolist()
    X_val   = validation_data["sequence"].tolist()
    X_test  = test_data["sequence"].tolist()

    num_labels = Y_train.shape[1]

    # Pretrain text
    with open(pretrain_file, 'r') as file:
        pretrain_data = file.read()

    # Parallelism: choose processes × threads so product == NUM_THREADS
    label_jobs, threads_per_job = choose_label_jobs_and_threads(NUM_THREADS, num_labels)
    print(f"[Parallelism] num_labels={num_labels}, label_jobs={label_jobs}, threads_per_job={threads_per_job} (product={label_jobs*threads_per_job})")

    # ---- TRAINING / VALIDATION SWEEPS (per label, in parallel) ----
    print("-----TRAINING")
    print("---SEARCH FOR BEST SPA(s)")
    print(f"nb_iterations, aug_factor, gamma, include_prev_context, handle_N_setting, ratio, ensemble_type, max_depth, threads_per_job, time taken, {metric}", flush=True)
    train_start_time = time.perf_counter()

    results = Parallel(n_jobs=label_jobs, backend="multiprocessing")(
        delayed(run_sweep_for_label)(
            k,
            X_train, Y_train[:, k],
            X_val,   Y_val[:,   k],
            pretrain_data,
            include_prev_contexts, gammas, nb_train_iterations,
            handle_N_settings, ratio_pretrain_train,
            ensemble_type, augmentation_factors, preserve_kmer,
            max_depth, ALPHABET_SIZE,
            threads_per_job,
            metric="auroc"  # Histone uses AUROC for selection
        )
        for k in range(num_labels)
    )

    # Collect best per label
    best_params_per_label = {}
    val_auc_per_label = np.zeros(num_labels, dtype=float)
    for k, best_val, best_params in results:
        best_params_per_label[k] = best_params
        val_auc_per_label[k] = best_val

    print("---BEST VALIDATION AUROC PER LABEL---")
    for k in range(num_labels):
        print(f"label {k:02d}: val_AUROC = {val_auc_per_label[k]:.4f}")

    # ---- RETRAIN BEST PER LABEL + TEST (parallel) ----
    print("-----TESTING")
    test_results = Parallel(n_jobs=label_jobs, backend="multiprocessing")(
        delayed(retrain_and_test_for_label)(
            k, best_params_per_label[k],
            X_train, Y_train[:, k].astype(int),
            X_test,  Y_test[:,  k].astype(int),
            pretrain_data, ALPHABET_SIZE,
            threads_per_job, dataset_folder
        )
        for k in range(num_labels)
    )

    test_auc_per_label = np.zeros(num_labels, dtype=float)
    for k, auc in test_results:
        test_auc_per_label[k] = auc

    mean_test_auc = float(test_auc_per_label.mean())
    print("---TEST AUROC PER LABEL---")
    for k in range(num_labels):
        print(f"label {k:02d}: test_AUROC = {test_auc_per_label[k]:.4f}")
    print(f"MEAN TEST AUROC (macro over {num_labels} labels): {mean_test_auc:.4f}")

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time

    #Output all measured times
    print("-----TIME PROFILING+")
    print(f"Read train + val + test data time: {(train_start_time - read_data_in_time): .5f}")
    # For reference (training symbols depends on chosen settings per-label; print base number)
    print(f"Number of base training sequences: {len(X_train)}")
    print(f"Length of one training sequence: {len(X_train[0]) if len(X_train) else 0}")
    print(f"Total training (sweeps + retrain) time: {train_duration:.3f} seconds")

    #Output memory report, which is automatically printed at the end of the run
    print("-----MEMORY REPORT")

if __name__ == "__main__":

    #Parse all arguments
    parser = argparse.ArgumentParser(description="Script for training and testing SPA model")

    parser.add_argument("-dataset_folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("-pretrain_file", type=str, required=True, help="Path to the pretraining file")
    parser.add_argument("--include_prev_context", type=str, required=True,
                        help="Set of values for INCLUDE_PREV_CONTEXT, e.g., '{True, False}'")
    parser.add_argument("--gamma", type=str, required=True,
                        help="Set of values for GAMMA, e.g., '{0.1, 0.5, 1.0}'")
    parser.add_argument("--nb_train_iterations", type=str, required=True,
                        help="Set of values for NB_TRAIN_ITERATIONS, e.g., '{1, 3, 5}'")
    parser.add_argument("--handle_n_setting", type=str, required=False,
                        help="Set of values for HANDLE_N_SETTING, e.g., '{remove, expand}'")
    parser.add_argument("--ratio_pretrain_train", type=str, required=True,
                        help="Set of values for RATIO_PRETRAIN_TRAIN, e.g., '{0.0, 0.1, 0.25}'")
    parser.add_argument("--ensemble_type", type=str, required=True,
                        help="Set of values for ENSEMBLE_TYPE e.g., '{depth,entropy}'")
    parser.add_argument("--num_threads", type=str, required=True,
                        help="Total threads budget. Script will choose processes×threads_per_job so the product equals this.")
    parser.add_argument("--validation_metric", type=str, default="accuracy",
                        choices=["accuracy", "mcc", "f1", "auroc"],   # NEW: add 'auroc'
                        help="Metric to use for validation")
    parser.add_argument("--augmentation_factors", type=str, required=False, default="{0}",
                        help=("Set of augmentation factors for adding shuffled versions of the positive "
                        "sequences to the negative training examples, e.g., '{0, 0.5, 1}'"
                        ))
    parser.add_argument("--shuffle_preserve_kmer", type=int, default=2,
                        help="Preserve k-mer frequncies when shuffling sequences")
    parser.add_argument("--max_depth", type=str, required=False, default="{}",
                        help="Set of max depths for the LZ78 tree, e.g., {4, 8, 12}., tried in addition to not limiting the depth. Defaults to empty ")
    args = parser.parse_args()

    # Convert string inputs to Python sets
    include_prev_contexts = parse_bool(args.include_prev_context)
    gammas = parse_set(args.gamma)
    nb_train_iterations = parse_set(args.nb_train_iterations)
    nb_train_iterations = {int(x) for x in nb_train_iterations}

    handle_N_settings = {"remove"}  # keep as before

    ratio_pretrain_train = parse_set(args.ratio_pretrain_train)
    ensemble_type = parse_set(args.ensemble_type)

    augmentation_factors = parse_set(args.augmentation_factors)
    preserve_kmer = args.shuffle_preserve_kmer
    # print(f"Preserving k-mer frequencies: {preserve_kmer}")

    num_threads = parse_set(args.num_threads)
    num_threads = int(list(num_threads)[0])
    NUM_THREADS = num_threads  # NEW: treat as total threads budget

    #max_depth = [None] + [int(x) for x in list(parse_set(args.max_depth))]
    max_depth = [int(x) for x in list(parse_set(args.max_depth))]


    main(args.dataset_folder, args.pretrain_file, args.validation_metric)
