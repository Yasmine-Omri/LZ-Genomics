'''
Uses trained SPAs to perform inference on a test dataset and report test accuracy.
The script can be easily modified to perform inference on a single sequence.

EXAMPLE USAGE:

python3 Inference.py --dataset Trained_SPAs/mouse_0 --dataset_test_csv GUE/mouse/0/test.csv --nb_classes 2

'''


from lz78 import Sequence, LZ78Encoder, CharacterMap, BlockLZ78Encoder, LZ78Classifier
from lz78 import encoded_sequence_from_bytes, spa_from_bytes, classifier_from_files
import numpy as np
import lorem
import requests
from sys import stdout
from os import makedirs
import pandas as pd
from tqdm import tqdm
import time

import argparse



def test_seq (data, spas: LZ78Classifier):
    nb_correct = 0
    nb_test_total = 0
    for row in tqdm(data.itertuples(), total=len(data)):
        nb_test_total += 1
        seq = row[1]
        correct_label = row[2]
        encoded_seq = Sequence(seq, charmap=CharacterMap("ACGT"))
        predicted_label = spas.classify(encoded_seq)
                
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

    spa = classifier_from_files([f"{dataset}_{i}.bin" for i in range(nb_classes)])

    # Load only the specific row (e.g., row index 0)

    test_path = dataset_test_csv
    test_data = pd.read_csv(test_path)
    for i in range(len(test_data)):
        test_data.loc[i, "sequence"] =  "".join([x for x in test_data.loc[i, "sequence"] if x in "ACGT"])

    start = time.perf_counter()
    test_accuracy = test_seq(test_data, spa)
    end = time.perf_counter()
    print("Test accuracy", test_accuracy)
    print("Total test time (seconds)", end - start)


if __name__ == "__main__":
    main()
