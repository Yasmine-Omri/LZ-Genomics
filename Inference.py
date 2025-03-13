'''
Uses trained SPAs to perform inference on a test dataset and report test accuracy.
The script can be easily modified to perform inference on a single sequence.

EXAMPLE USAGE:

python3 Inference.py --dataset Trained_SPAs/mouse_0 --dataset_test_csv GUE/mouse/0/test.csv --nb_classes 2

'''


from lz78 import Sequence, LZ78Encoder, CharacterMap, BlockLZ78Encoder, LZ78SPA
from lz78 import encoded_sequence_from_bytes, spa_from_bytes
import numpy as np
import lorem
import requests
from sys import stdout
from os import makedirs

import argparse



def test_seq (data, spa):
    nb_correct = 0
    nb_test_total = 0
    for row in data.itertuples():
        seq = row[1]
        correct_label = row[2]
        encoded_seq = Sequence(seq, charmap=CharacterMap("ACGT"))
        seq_len = len(seq)
        nb_test_total += 1
        spa_logloss = []
        for index in range(len(spa)):
            spa_logloss.append(spa[index].compute_test_loss(encoded_seq, include_prev_context=False) / seq_len)
        
        predicted_label = spa_logloss.index(min(spa_logloss))
        
        if predicted_label == correct_label:
            nb_correct += 1 

    accuracy = nb_correct / nb_test_total 
    return accuracy


# load spa from bytes
def main():

    parser = argparse.ArgumentParser(description="Process dataset and parameters.")
    
    # Define command-line arguments
    parser.add_argument("--dataset", type=str, required=True, help="Path to the .bin dataset file")
    parser.add_argument("--dataset_test_csv", type=str, required=True, help="Path to the test CSV file")
    parser.add_argument("--nb_classes", type=int, required=True, help="Number of classes")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Access arguments
    dataset = args.dataset
    dataset_test_csv = args.dataset_test_csv
    nb_classes = args.nb_classes

    
    spa = []
    for label in range(nb_classes):
        spa_bin_file = f"{dataset}_{label}.bin"
        with open(spa_bin_file, 'rb') as file:
            encoded_bytes = file.read()
        spa.append(spa_from_bytes(encoded_bytes))


    # Load only the specific row (e.g., row index 0)
    import pandas as pd


    test_path = dataset_test_csv
    test_data = pd.read_csv(test_path)

    test_accuracy = test_seq(test_data, spa)
    print("Test accuracy", test_accuracy)


if __name__ == "__main__":
    main()
