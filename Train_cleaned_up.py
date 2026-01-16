'''
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
from sklearn.metrics import average_precision_score, roc_auc_score
from ushuffle import shuffle, Shuffler
from dataclasses import dataclass, field
import argparse
from sklearn.metrics import matthews_corrcoef, f1_score


ALPHABET_SIZE = 4
ENSEMBLE_N = 10
BACKSHIFT_CTX_LEN = 20

@dataclass
class LZ78TrainConfig:
    include_prev_context: bool
    handle_N_setting: str
    ratio_pretrain_train: float
    max_depth: int = field(default=None)


@dataclass
class HyperparameterSweep:
    include_prev_context: list[bool]
    gamma: list[float]
    nb_train_iterations: list[int]
    handle_N_setting: list[str]
    ratio_pretrain_train: list[float]
    ensemble_type: list[str]
    max_depth: list[int]
    augmentation_factor: list[float]
    preserve_kmer: int = 2


RCMAP = str.maketrans("ACGTN", "TGCAN")
def revcomp(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(RCMAP)[::-1]


def augment_revcomp(train_data: pd.DataFrame, ratio: float=0.5, seed=42) -> pd.DataFrame:
    if ratio == 0:
        return train_data
    for c in train_data["label"].unique():
        class_data: pd.Series = train_data[train_data['label'] == c]
        sampled_seqs = class_data.sample(frac=ratio, replace=False, random_state=seed)["sequence"]
        # Create reverse complement sequences
        revcomp_seqs = sampled_seqs.apply(revcomp)
        # Append to the training data
        train_data = pd.concat([train_data, pd.DataFrame({"sequence": revcomp_seqs, "label": c})])
    return train_data


#Pretraining the spa routine
def pretrain_spa(
    seq,
    spa: list[LZ78SPA],
    nb_pretrain_symbols,
    config: LZ78TrainConfig
):
    if nb_pretrain_symbols == 0:
        return
    # Split the sequence into individual elements by newline
    elements = seq.splitlines()
    
    # Determine the number of elements to use based on the specified percentage
    # num_elements = int(len(elements) * (percentage / 100))
    len_pretrain_seq = len(elements[0])
    nb_pretrain_seqs = math.ceil(nb_pretrain_symbols / len_pretrain_seq)

    selected_elements = elements[:nb_pretrain_seqs]
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
            if not config.include_prev_context:
                spa[index].reset_state()
            spa[index].train_on_block(encoded_seq)


def train_spa_oneIter(data, spa: list[LZ78SPA], config: LZ78TrainConfig):
    for row in data.itertuples():
        seq, label = row[1], row[2]
        
        # Encode sequence
        encoded_seq = Sequence(seq, charmap=CharacterMap("ACGT"))
        if not config.include_prev_context:
            spa[label].reset_state()
        spa[label].train_on_block(encoded_seq)


def train_spa(data, spa: list[LZ78SPA], iterations: int, config: LZ78TrainConfig):
    for i in range(iterations):
        for row in data.itertuples():
            seq = row[1]
            label = row[2]
            
            # Encode sequence
            encoded_seq = Sequence(seq, charmap=CharacterMap("ACGT"))
            if not config.include_prev_context:
                spa[label].reset_state()
            spa[label].train_on_block(encoded_seq)
            

def test_seq(data: pd.DataFrame, spas: list[LZ78SPA], compute_auc=False, n_threads=32):
    # for every test seq, run it through all spas
    # classification = label associated with lowest loss spa
    # check classification against ground truth
    # compute metric (of all test runs)

    labels = data["label"]
    data_seq = [Sequence(seq, charmap=CharacterMap("ACGT")) for seq in data["sequence"]]
    log_losses = np.zeros((len(spas), len(data_seq)))
    for i in range(len(spas)):
        log_losses[i, :] = [res["avg_log_loss"] for res in spas[i].compute_test_loss_parallel(data_seq, num_threads=n_threads)]
    classes = np.argmin(log_losses, axis=0)

    results = {}
    results["accuracy"] = (classes == labels).sum() / len(labels)
    results["mcc"] = matthews_corrcoef(labels, classes)
    results["f1"] = f1_score(labels, classes, average='weighted')

    if compute_auc:
        scores = compute_scores_matrix(data, spas, n_threads)   # (N, K)
        per_auroc, per_auprc, macro_auroc, macro_auprc = ovr_auroc_auprc(labels, scores)
        results.update({
            "macro_auroc": macro_auroc,
            "macro_auprc": macro_auprc,
            "per_auroc": per_auroc,
            "per_auprc": per_auprc
        })
    return results


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


def augment_shuffle(train_data, aug_factor, preserve_kmer=2):
    if aug_factor == 0:
        return train_data
    pos_train = train_data[train_data['label'] != 0]
    new_negatives_train = []
    for sample in pos_train["sequence"]:
        sample = sample.lower().encode('utf-8')
        shuffler = Shuffler(sample, preserve_kmer)
        if aug_factor < 1:
            if random.random() < aug_factor:
                # print("bye", shuffler.shuffle())
                new_negatives_train.append([shuffler.shuffle().decode('utf-8').upper(), 0])
            continue
        for _ in range(int(aug_factor)):
            new_negatives_train.append([shuffler.shuffle().decode('utf-8').upper(), 0])
    train_data = pd.concat(
        [train_data, pd.DataFrame(new_negatives_train, columns=['sequence', 'label'])],
        ignore_index=True
    )
    return train_data


def compute_scores_matrix(data_df: pd.DataFrame, spas: list[LZ78SPA], n_threads=32):
    """
    Returns scores with shape (N, K), where higher means more likely class.
    We use negative avg_log_loss as a score.
    """
    seqs = [Sequence(s, charmap=CharacterMap("ACGT")) for s in data_df["sequence"]]
    class_losses = []
    for spa in spas:
        # list of dicts -> extract avg_log_loss for each seq
        class_losses.append([r["avg_log_loss"] for r in spa.compute_test_loss_parallel(seqs, num_threads=n_threads)])
    losses = np.vstack(class_losses)         # (K, N)
    scores = (-losses).T                     # (N, K)
    return scores


def ovr_auroc_auprc(y_true: np.ndarray, scores: np.ndarray):
    """
    One-vs-rest AUROC & AUPRC per class + macro means.
    y_true: (N,) with int labels 0..K-1
    scores: (N, K) with higher=more likely
    """
    y_true = np.asarray(y_true)
    K = scores.shape[1]
    per_cls_auroc, per_cls_auprc = [], []
    for c in range(K):
        y_bin = (y_true == c).astype(int)
        s = scores[:, c]
        # AUROC can error if only one class present; guard and return NaN
        try:
            auroc = roc_auc_score(y_bin, s)
        except ValueError:
            auroc = float("nan")
        auprc = average_precision_score(y_bin, s)
        per_cls_auroc.append(auroc)
        per_cls_auprc.append(auprc)
    macro_auroc = float(np.nanmean(per_cls_auroc))
    macro_auprc = float(np.nanmean(per_cls_auprc))
    return per_cls_auroc, per_cls_auprc, macro_auroc, macro_auprc


@profile
def main(
    dataset_folder: str,
    pretrain_file: str,
    val_metric: str,
    test_metric: str,
    hyperparams: HyperparameterSweep,
    num_threads: int = 32,
    revcomp_augment_factor: float = 0.0,
    target_class_ratios: list[float] = [],
    no_ensemble: bool = False
):
    read_data_in_time = time.perf_counter()
    
    # Read train, val, test data 
    train_path = f"{dataset_folder}/train.csv"
    val_path = f"{dataset_folder}/dev.csv"
    test_path = f"{dataset_folder}/test.csv"

    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(val_path)

    n_classes = len(train_data['label'].unique())

    if len(target_class_ratios) == 0: # no balancing
        target_class_ratios = [1.0 / n_classes] * n_classes
        balance_classes = False
    elif target_class_ratios == [0]: # code for default balancing
        target_class_ratios = [1.0 / n_classes] * n_classes
        balance_classes = True
    else:
        balance_classes = True

    target_class_ratios = [r / sum(target_class_ratios) for r in target_class_ratios]
     # check that class_ratios has one entry per class and sums to 1.0
    assert len(target_class_ratios) == n_classes and abs(sum(target_class_ratios) - 1.0) < 1e-6, \
        "class_ratios must have one entry per class and sum to 1.0"
    if balance_classes:
        print(f"Using class ratios: {target_class_ratios}", flush=True)

    raw_class_ratios = train_data['label'].value_counts(normalize=True).sort_index().tolist()
    if balance_classes:
        print(f"Raw class ratios: {raw_class_ratios}", flush=True)

    # Downsample classes to meet target ratios
    # raw * sample = target so sample propto target/raw
    sample_weights = [t / r if r > 0 else 0.0 for r, t in zip(raw_class_ratios, target_class_ratios)]
    # we want to max weight to be 1.0
    max_weight = max(sample_weights)
    sample_weights = [w / max_weight for w in sample_weights]
    if balance_classes:
        print(f"Sampling weights: {sample_weights}", flush=True)
    
    unique_labels = train_data['label'].unique()
    
    with open(pretrain_file, 'r') as file:
        pretrain_data = file.read()
    
    # Train all SPAs using all possible combinations of hyperparams
    # Test all on validation set, return best SPA
    results_df = pd.DataFrame(columns=[
        "INCLUDE_PREV_CONTEXT", "GAMMA", "NB_TRAIN_ITERATIONS", "HANDLE_N_SETTING",
        "RATIO_PRETRAIN_TRAIN", "ENSEMBLE_TYPE", "MAX_DEPTH", "NUM_THREADS",
        "TRAINING_TIME", "VALIDATION METRIC", "AUGMENTATION_FACTOR",
    ])

    print("-----TRAINING")
    print("---SEARCH FOR BEST SPA(s)")
    print(", ".join(results_df.columns), flush=True)
    train_start_time = time.perf_counter()

    assert hyperparams.handle_N_setting == ["remove"], "Only 'remove' setting is currently supported for handle_N_setting."
     # Preprocess training and validation data to handle 'N's
    train_data = augment_revcomp(handle_N(train_data, setting="remove"), ratio=revcomp_augment_factor)
    validation_data = handle_N(validation_data, setting="remove")

    for include_prev_context, handle_N_setting, ratio_pretrain_train, aug_factor, max_depth in itertools.product(
        hyperparams.include_prev_context,
        hyperparams.handle_N_setting,
        hyperparams.ratio_pretrain_train,
        hyperparams.augmentation_factor,
        hyperparams.max_depth
    ):
        train_config = LZ78TrainConfig(
            include_prev_context=include_prev_context,
            handle_N_setting=handle_N_setting,
            ratio_pretrain_train=ratio_pretrain_train,
            max_depth=max_depth
        )

        nb_train_seqs = len(train_data)
        seq_len = len(train_data.iloc[0, 0])
        nb_train_symbols = nb_train_seqs * seq_len
        
        # Create list of spas based on number of labels: (spa_0 and spa_1 for labels 0, 1)
        spa = [
            LZ78SPA(
                alphabet_size=ALPHABET_SIZE,
                compute_training_loss=False,
                max_depth=train_config.max_depth
            ) for _ in unique_labels
        ]
        for i in range(len(unique_labels)):
            spa[i].set_inference_config(
                lb=1e-5,
                ensemble_type="entropy",
                ensemble_n=ENSEMBLE_N,
                backshift_parsing=True,
                backshift_ctx_len=BACKSHIFT_CTX_LEN,
                backshift_break_at_phrase=True
            )
            if no_ensemble:
                spa[i].set_inference_config(
                    lb=0,
                    ensemble_n=1,
                    backshift_parsing=False,
                )

        #Pretrain spas
        nb_pretrain_symbols = math.ceil(ratio_pretrain_train * nb_train_symbols)
        pretrain_spa(pretrain_data, spa, nb_pretrain_symbols, train_config) 

        #Train spas iteratively with more data while testing the other hyperparameters
        iterated_times = 0
        for nb_iterations in hyperparams.nb_train_iterations:
            train_one_iter_start_time = time.perf_counter()
            for _ in range(nb_iterations - iterated_times):
                # downsample training data according to class ratios
                pre_balance_data = train_data.copy()
                if balance_classes:
                    sampled_dfs = []
                    for label in unique_labels:
                        class_df = train_data[train_data['label'] == label]
                        weight = sample_weights[label]
                        sampled_class_df = class_df.sample(frac=weight, replace=False, random_state=42)
                        sampled_dfs.append(sampled_class_df)
                    train_data = pd.concat(sampled_dfs, ignore_index=True)

                train_data = augment_shuffle(train_data, aug_factor, hyperparams.preserve_kmer)

                if balance_classes: #check class ratios after downsampling
                    new_class_ratios = train_data['label'].value_counts(normalize=True).sort_index().tolist()
                    print(f"New class ratios: {new_class_ratios}", flush=True)
                    assert np.allclose(new_class_ratios, target_class_ratios, atol=0.05)
                train_spa_oneIter(train_data, spa, train_config)

                train_data = pre_balance_data  # reset train_data to pre-balance state for next iteration
            
            iterated_times = nb_iterations
            for gamma in hyperparams.gamma:
                for ensemble in hyperparams.ensemble_type:
                # Test on validation test to assess this combination of hyperparams
                    for index in range(len(spa)):
                        spa[index].set_inference_config(gamma=gamma, ensemble_type=ensemble)
                    val_metric_value = test_seq(
                        validation_data, spa, n_threads=num_threads, compute_auc=(val_metric in ["auroc", "auprc"])
                    )["macro_" + val_metric if val_metric in ["auroc", "auprc"] else val_metric]
                    train_one_iter_end_time = time.perf_counter()
                    train_one_iter_duration = train_one_iter_end_time - train_one_iter_start_time
                
                    current_result = pd.DataFrame([{
                        "INCLUDE_PREV_CONTEXT": include_prev_context,
                        "GAMMA": gamma,
                        "NB_TRAIN_ITERATIONS": nb_iterations,
                        "HANDLE_N_SETTING": handle_N_setting,
                        "RATIO_PRETRAIN_TRAIN": ratio_pretrain_train,
                        "ENSEMBLE_TYPE": ensemble,
                        "MAX_DEPTH": max_depth if max_depth else 0,
                        "NUM_THREADS": num_threads,
                        "TRAINING_TIME": train_one_iter_duration, 
                        "VALIDATION METRIC": val_metric_value,
                        "AUGMENTATION_FACTOR": aug_factor,
                    }])

                # Concatenate the current result with results_df
                results_df = results_df.dropna(axis=1, how='all')
                current_result = current_result.dropna(axis=1, how='all')

                results_df = pd.concat([results_df, current_result], ignore_index=True)

                row = current_result.iloc[0].tolist()
                # make VALIDATION_METRIC 3 decimal places
                row[results_df.columns.get_loc("VALIDATION METRIC")] = f"{row[results_df.columns.get_loc('VALIDATION METRIC')]*100:.2f}"
                print(", ".join(map(str, row)), flush=True)
    
    # Find the best hyperparameter combination based on the highest validation metric
    print("---BEST SPA(s) FOUND")
    best_row = results_df.loc[results_df['VALIDATION METRIC'].idxmax()]
    best_params = best_row.to_dict()
    print("Best hyperparameters:", best_params)

    best_train_config = LZ78TrainConfig(
        include_prev_context=best_params["INCLUDE_PREV_CONTEXT"],
        handle_N_setting=best_params["HANDLE_N_SETTING"],
        ratio_pretrain_train=best_params["RATIO_PRETRAIN_TRAIN"],
        max_depth=best_params["MAX_DEPTH"] if best_params["MAX_DEPTH"] != 0 else None
    )

    best_gamma = best_params["GAMMA"]
    best_ensemble_type = best_params["ENSEMBLE_TYPE"]
    best_nb_train_iter = int(best_params["NB_TRAIN_ITERATIONS"])

    # Retrain our best SPAs and use that to test on test data 
    spa = [LZ78SPA(
        alphabet_size=ALPHABET_SIZE,
        gamma=best_gamma,
        compute_training_loss=False,
        max_depth=best_train_config.max_depth
    ) for _ in unique_labels]
    for i in range(len(unique_labels)):
        spa[i].set_inference_config(
            lb=1e-5,
            ensemble_type=best_ensemble_type,
            ensemble_n=ENSEMBLE_N,
            backshift_parsing=True,
            backshift_ctx_len=BACKSHIFT_CTX_LEN,
            backshift_break_at_phrase=True
        )
        if no_ensemble:
            spa[i].set_inference_config(
                lb=0,
                ensemble_n=1,
                backshift_parsing=False
            )

    train_data = handle_N(train_data, setting=best_train_config.handle_N_setting)
    nb_train_seqs = len(train_data)
    seq_len = len(train_data.iloc[0, 0])
    nb_train_symbols = nb_train_seqs * seq_len
    nb_pretrain_symbols = math.ceil(best_train_config.ratio_pretrain_train * nb_train_symbols)

    pretrain_spa(pretrain_data, spa, nb_pretrain_symbols, config=best_train_config) 
    train_spa(train_data, spa, iterations=best_nb_train_iter, config=best_train_config)

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time

    
    
    # Final test
    print("-----TESTING")
    read_test_data_start_time = time.perf_counter()
    test_data = pd.read_csv(test_path)

    inference_start_time = time.perf_counter()

    test_data = handle_N(test_data)
    test_metric_values = test_seq(
        test_data, spa, n_threads=num_threads, compute_auc=(test_metric in ["auroc", "auprc"])
    )
    test_metric_processed = "macro_auroc" if test_metric == "auroc" else "macro_auprc" if test_metric == "auprc" else test_metric
    test_metric_value = test_metric_values[test_metric_processed]

    inference_end_time = time.perf_counter()
    print(f"Final metric ({test_metric}) with best hyperparameters: {(test_metric_value*100):.2f}")

    # print all other metrics too
    for metric in test_metric_values:
        if metric == test_metric_processed:
            continue
        print(f"Final metric ({metric}) with best hyperparameters: {test_metric_values[metric]}")

        
    inference_duration = inference_end_time - inference_start_time

    #Save all spas
    label = 0
    for sp in spa:
        spa_bytes = bytearray(sp.to_bytes())
        print(f"Mem in MB: {len(spa_bytes) / (1024 * 1024):.2f}", flush=True)
        makedirs("best_spas", exist_ok=True)
        # Extract the part after 'GUE/' and replace slashes with underscores
        # binary_file_name = dataset_folder.split("GUE/", 1)[-1].replace("/", "_")
        binary_file_name = "_".join(dataset_folder.split("/")[-3:])
        
        # Create the full path for the binary file
        binary_file_path = os.path.join("best_spas/", f"{binary_file_name}_{label}.bin")
        label += 1
        # Save the binary file
        with open(binary_file_path, 'wb') as file:
            file.write(spa_bytes)
    
    #Output all measured times
    print("-----TIME PROFILING+")
    print(f"Read train + val data time: {(train_start_time - read_data_in_time): .5f}")
    print(f"Number of training symbols: {nb_train_symbols}")
    print(f"Length of one training sequence: {len(train_data.iloc[0, 0])}")
    print(f"Total training time: {train_duration:.3f} seconds")
    
    print(f"Number of test sequences: {len(test_data)}")
    print(f"Length of test sequence: {len(test_data.iloc[0, 0])}")
    print(f"Read test data time: {(inference_start_time - read_test_data_start_time): .5f}")
    print(f"Total inference time: {inference_duration:.3f} seconds")
    print(f"Inference time/symbol: {inference_duration/(len(test_data) * len(test_data.iloc[0, 0]))} seconds")

    #Output memory report, which is automatically printed at the end of the run
    print("-----MEMORY REPORT")

if __name__ == "__main__":

    #Parse all arguments
    parser = argparse.ArgumentParser(description="Script for training and testing SPA model")

    parser.add_argument("-dataset_folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("-pretrain_file", type=str, required=True, help="Path to the pretraining file")
    parser.add_argument("--include_prev_context", type=str, nargs='+', required=True,
                        help="Set of values for INCLUDE_PREV_CONTEXT, e.g., 'True False'")
    parser.add_argument("--gamma", type=float, nargs='+', required=True,
                        help="Set of values for GAMMA, e.g., '0.1 0.33 0.5 0.75 1 3 5'")
    parser.add_argument("--nb_train_iterations", type=int, nargs='+', required=True,
                        help="Set of values for NB_TRAIN_ITERATIONS, e.g., '1 3 5'")
    parser.add_argument("--handle_n_setting", type=str, nargs='+', required=False, default=["remove"],
                        help="Set of values for HANDLE_N_SETTING, e.g., 'remove expand'")
    parser.add_argument("--ratio_pretrain_train", type=float, nargs='+', required=True,
                        help="Set of values for RATIO_PRETRAIN_TRAIN, e.g., '0.0 0.1 0.25'")
    parser.add_argument("--ensemble_type", type=str, nargs='+', required=True,
                        help="Set of values for ENSEMBLE_TYPE e.g., 'depth entropy'")
    parser.add_argument("--num_threads", type=int, required=True,
                        help="Number of threads to compute on in parallel'")
    parser.add_argument("--validation_metric", type=str, default="accuracy",
                        choices=["accuracy", "mcc", "f1", "auroc", "auprc"],
                        help="Metric to use for validation, default is 'accuracy'")
    parser.add_argument("--test_metric", type=str, default="accuracy",
                        choices=["accuracy", "mcc", "f1", "auroc", "auprc"],
                        help="Metric to use for validation, default is 'accuracy'")
    parser.add_argument("--augmentation_factors", type=float, nargs='+', required=False, default=[0],
                        help=("Set of augmentation factors for adding shuffled versions of the positive "
                        "sequences to the negative training examples, e.g., '0 0.5 1'"
                        ))
    parser.add_argument("--shuffle_preserve_kmer", type=int, default=2,
                        help="Preserve k-mer frequncies when shuffling sequences")
    parser.add_argument("--max_depth", type=int, nargs='+', required=False, default=[],
                        help="Set of max depths for the LZ78 tree, e.g., '4 8 12', tried in addition to not limiting the depth. Defaults to empty ")
    parser.add_argument("--revcomp_augment_factor", type=float, default=0.0, 
                        help="Ratio of reverse-complement sequences to add to the training data. Default is 0 (no augmentation).")
    parser.add_argument("--class_ratios", type=float, nargs='+', required=False, default=[],
                        help=("Set of class ratios for datasets with imbalanced classes, e.g., "
                        "'1 10' means each epoch will consist of 1 negative and 10 positive samples. "
                        "These need not sum to 1.0. Defaults to empty (no class weighting)."))
    parser.add_argument("--no_ensemble", action='store_true',
                        help="If set, do not use ensemble methods during inference.")
    args = parser.parse_args()

    include_prev_context = [s.lower() == 'true' for s in args.include_prev_context]

    hyperparams = HyperparameterSweep(
        include_prev_context=include_prev_context,
        gamma=args.gamma,
        nb_train_iterations=args.nb_train_iterations,
        handle_N_setting=args.handle_n_setting,
        ratio_pretrain_train=args.ratio_pretrain_train,
        ensemble_type=args.ensemble_type,
        augmentation_factor=args.augmentation_factors,
        max_depth=[None] + args.max_depth,
        preserve_kmer=args.shuffle_preserve_kmer
    )

    print("Parsed hyperparameters:", hyperparams, flush=True)

    main(
        dataset_folder=args.dataset_folder,
        pretrain_file=args.pretrain_file,
        val_metric=args.validation_metric,
        test_metric=args.test_metric,
        hyperparams=hyperparams,
        num_threads=args.num_threads,
        revcomp_augment_factor=args.revcomp_augment_factor,
        target_class_ratios=args.class_ratios,
        no_ensemble=args.no_ensemble
    )

