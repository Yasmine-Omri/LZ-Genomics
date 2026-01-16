#!/bin/bash

MOTIF_CSVS=$(ls motif_training_csv/*.csv | xargs -n 1 basename | sed 's/\.csv//')
MEME_FILE=meme/H12CORE_meme_format.meme
OUTPUT_DIR=motif_kmer_freqs_csv
# Loop through each dataset folder and its associated output file

MAX_PROC=64
n_proc=1
for MOTIF in $MOTIF_CSVS; do
    echo "Training motif: $MOTIF"
    # Call the Python script with the current dataset folder and output file
    python Train_motif_trees_kmer.py \
        --motif_name $MOTIF \
        --meme_path $MEME_FILE \
        --output_dir $OUTPUT_DIR \
        --alpha 0.5 --use_canonical \
        --threads 64 \
        --hashsize 200M &
    
    n_proc=$((n_proc + 1))
    if (( n_proc > MAX_PROC )); then
        n_proc=1
        wait
    fi
done

wait


echo "All tasks are complete."

