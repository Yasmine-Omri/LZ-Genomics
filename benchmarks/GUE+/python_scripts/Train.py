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

def test_seq (data: pd.DataFrame, spas: list[LZ78SPA], metric, n_threads=32):
    # for every test seq,
    # run it through all spas
    # classification = label associated with lowest loss spa
    # check classification against ground truth
    # compute metric (of all test runs)
    labels = data["label"]
    data = [Sequence(seq, charmap=CharacterMap("ACGT")) for seq in data["sequence"]]
    log_losses = np.zeros((len(spas), len(data)))
    for i in range(len(spas)):
        log_losses[i, :] = [res["avg_log_loss"] for res in spas[i].compute_test_loss_parallel(data, num_threads=n_threads)]
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
        raise ValueError("Invalid metric specified. Choose from 'accuracy', 'mcc', or 'f1'.")


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

@profile
def main(dataset_folder, pretrain_file, metric):
    global INCLUDE_PREV_CONTEXT
    global GAMMA
    global NB_TRAIN_ITERATIONS 
    global HANDLE_N_SETTING 
    global RATIO_PRETRAIN_TRAIN 
    global ENSEMBLE_TYPE 
    global NUM_THREADS
    
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

    read_data_in_time = time.perf_counter()
    
    # Read train, val, test data 
    train_path = f"{dataset_folder}/train.csv"
    val_path = f"{dataset_folder}/dev.csv"
    test_path = f"{dataset_folder}/test.csv"
    

    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(val_path)
    
    ALPHABET_SIZE = 4
    unique_labels = train_data['label'].unique()
    
    with open(pretrain_file, 'r') as file:
        pretrain_data = file.read()
    
    # Train all SPAs using all possible combinations of hyperparams
    # Test all on validation set, return best SPA
    results_df = pd.DataFrame(columns=[
    "INCLUDE_PREV_CONTEXT", "GAMMA", "NB_TRAIN_ITERATIONS", 
    "HANDLE_N_SETTING", "RATIO_PRETRAIN_TRAIN", "ENSEMBLE_TYPE", "NUM_THREADS", "VALIDATION METRIC"
    ])

    print("-----TRAINING")
    print("---SEARCH FOR BEST SPA(s)")
    print(f"nb_iterations, aug_factor, gamma, include_prev_context, handle_N_setting, ratio, ensemble_type, max_depth, num_threads, time taken, {metric}", flush=True)
    train_start_time = time.perf_counter()
    for include_prev_context, handle_N_setting, ratio, aug_factor, max_depth in itertools.product(
        include_prev_contexts, handle_N_settings, ratio_pretrain_train, augmentation_factors, max_depth
    ):  
        INCLUDE_PREV_CONTEXT = include_prev_context
        GAMMA = gammas
        NB_TRAIN_ITERATIONS = 0
        HANDLE_N_SETTING = handle_N_setting
        RATIO_PRETRAIN_TRAIN = ratio 
        ENSEMBLE_TYPE = ensemble_type
        NUM_THREADS = num_threads
        MAX_DEPTH = max_depth

        old_train_data = train_data.copy()

        train_data = handle_N(train_data, setting=HANDLE_N_SETTING)
        validation_data = handle_N(validation_data, setting=HANDLE_N_SETTING)

        # Augment training data with shuffled positive sequences
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

        nb_train_seqs = len(train_data)
        seq_len = len(train_data.iloc[0, 0])
        nb_train_symbols = nb_train_seqs * seq_len
        
        # Create list of spas based on number of labels: (spa_0 and spa_1 for labels 0, 1)
        spa = [LZ78SPA(alphabet_size=ALPHABET_SIZE, compute_training_loss=False, max_depth=int(MAX_DEPTH) if MAX_DEPTH else None) for _ in unique_labels]
        for i in range(len(unique_labels)):
            spa[i].set_inference_config(
                lb=1e-5,
                ensemble_type="entropy",
                ensemble_n=10,
                backshift_parsing=True,
                backshift_ctx_len=20,
                backshift_break_at_phrase=True
            )

        #Pretrain spas
        nb_pretrain_symbols = math.ceil(RATIO_PRETRAIN_TRAIN * nb_train_symbols)
        pretrain_spa(pretrain_data, spa, nb_pretrain_symbols) 

        #Train spas iteratively with more data while testing the other hyperparameters
        iterated_times = 0
        for nb_iterations in nb_train_iterations:
            train_one_iter_start_time = time.perf_counter()
            for _ in range(nb_iterations - iterated_times):
                train_spa_oneIter(train_data, spa)
            
            iterated_times = nb_iterations
            for gamma in gammas:
                for ensemble in ENSEMBLE_TYPE:
                # Test on validation test to assess this combination of hyperparams
                    for index in range(len(spa)):
                        spa[index].set_inference_config(gamma=gamma, ensemble_type=ensemble)
                    val_metric = test_seq(validation_data, spa, metric=metric, n_threads=NUM_THREADS)
                    train_one_iter_end_time = time.perf_counter()
                    train_one_iter_duration = train_one_iter_end_time - train_one_iter_start_time
                    print(f"{nb_iterations}, {aug_factor}, {gamma}, {include_prev_context}, {handle_N_setting}, {ratio}, {ensemble}, {max_depth}, {NUM_THREADS}, {train_one_iter_duration:.3f}, {(val_metric * 100):.2f}", flush=True)

                
                    current_result = pd.DataFrame([{
                        "INCLUDE_PREV_CONTEXT": INCLUDE_PREV_CONTEXT,
                        "GAMMA": gamma,
                        "NB_TRAIN_ITERATIONS": nb_iterations,
                        "HANDLE_N_SETTING": HANDLE_N_SETTING,
                        "RATIO_PRETRAIN_TRAIN": RATIO_PRETRAIN_TRAIN,
                        "ENSEMBLE_TYPE": ensemble,
                        "MAX_DEPTH": MAX_DEPTH if MAX_DEPTH else 0,
                        "NUM_THREADS": NUM_THREADS,
                        "TRAINING_TIME": train_one_iter_duration, 
                        "VALIDATION METRIC": val_metric,
                        "AUGMENTATION_FACTOR": aug_factor,
                    }])

                # Concatenate the current result with results_df
                results_df = results_df.dropna(axis=1, how='all')
                current_result = current_result.dropna(axis=1, how='all')

                results_df = pd.concat([results_df, current_result], ignore_index=True)
        train_data = old_train_data.copy()  # Reset train_data to original state after each hyperparameter combination
    
    # Find the best hyperparameter combination based on the highest validation metric
    print("---BEST SPA(s) FOUND")
    best_row = results_df.loc[results_df['VALIDATION METRIC'].idxmax()]
    best_params = best_row.to_dict()
    print("Best hyperparameters:", best_params)

    INCLUDE_PREV_CONTEXT = best_params["INCLUDE_PREV_CONTEXT"]
    GAMMA = best_params["GAMMA"]
    NB_TRAIN_ITERATIONS = int(best_params["NB_TRAIN_ITERATIONS"])
    HANDLE_N_SETTING = best_params["HANDLE_N_SETTING"]
    RATIO_PRETRAIN_TRAIN = best_params["RATIO_PRETRAIN_TRAIN"]
    ENSEMBLE_TYPE = best_params["ENSEMBLE_TYPE"]
    NUM_THREADS = best_params["NUM_THREADS"]
    MAX_DEPTH = int(best_params["MAX_DEPTH"])
    if MAX_DEPTH == 0:
        MAX_DEPTH = None

    # Retrain our best SPAs and use that to test on test data 
    spa = [LZ78SPA(alphabet_size=ALPHABET_SIZE, gamma= GAMMA, compute_training_loss=False, max_depth=MAX_DEPTH) for _ in unique_labels]
    for i in range(len(unique_labels)):
        spa[i].set_inference_config(
            lb=1e-5,
            ensemble_type= ENSEMBLE_TYPE,
            ensemble_n=10,
            backshift_parsing=True,
            backshift_ctx_len=20,
            backshift_break_at_phrase=True
        )

    train_data = handle_N(train_data, setting=HANDLE_N_SETTING)
    nb_train_seqs = len(train_data)
    seq_len = len(train_data.iloc[0, 0])
    nb_train_symbols = nb_train_seqs * seq_len
    nb_pretrain_symbols = math.ceil(RATIO_PRETRAIN_TRAIN * nb_train_symbols)

    pretrain_spa(pretrain_data, spa, nb_pretrain_symbols) 
    train_spa(train_data, spa, iterations=NB_TRAIN_ITERATIONS)

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time

    
    
    # Final test
    print("-----TESTING")
    read_test_data_start_time = time.perf_counter()
    test_data = pd.read_csv(test_path)

    inference_start_time = time.perf_counter()

    test_data = handle_N(test_data)
    test_metric = test_seq(test_data, spa, metric=metric, n_threads=NUM_THREADS)

    inference_end_time = time.perf_counter()
    print(f"Final metric with best hyperparameters: {(test_metric*100):.2f}")
    
        
    inference_duration = inference_end_time - inference_start_time

    #Save all spas
    label = 0
    for sp in spa:
        spa_bytes = bytearray(sp.to_bytes())
        print(f"Mem in MB: {len(spa_bytes) / (1024 * 1024):.2f}", flush=True)
        makedirs("best_spas", exist_ok=True)
        # Extract the part after 'GUE/' and replace slashes with underscores
        binary_file_name = dataset_folder.split("GUE/", 1)[-1].replace("/", "_")
        
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
                        help="Number of threads to compute on in parallel'")
    parser.add_argument("--validation_metric", type=str, default="accuracy",
                        choices=["accuracy", "mcc", "f1"],
                        help="Metric to use for validation, default is 'accuracy'")
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

    handle_N_settings = {"remove"}

    ratio_pretrain_train = parse_set(args.ratio_pretrain_train)

    ensemble_type = parse_set(args.ensemble_type)

    augmentation_factors = parse_set(args.augmentation_factors)
    preserve_kmer = args.shuffle_preserve_kmer
    print(f"Preserving k-mer frequencies: {preserve_kmer}")

    num_threads = parse_set(args.num_threads)
    num_threads = int(list(num_threads)[0])

    max_depth = [None] + [int(x) for x in list(parse_set(args.max_depth))]

    main(args.dataset_folder, args.pretrain_file, args.validation_metric)

    