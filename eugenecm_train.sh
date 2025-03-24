#!/bin/bash

# Define the pretrain file and dataset folders with their associated output files
PRETRAIN_FILE="./dnabert_2_pretrain/dev.txt"
OUTPUT_DIR="./outputs_minimal"

   
# Define the dataset folders and their respective output files
declare -A DATASET_OUTPUTS=(
    #Mouse
    ["./GUE/mouse/0"]="mouse0.txt"
    ["./GUE/mouse/1"]="mouse1.txt"
    ["./GUE/mouse/2"]="mouse2.txt"
    ["./GUE/mouse/3"]="mouse3.txt"
    ["./GUE/mouse/4"]="mouse4.txt"

    #TF
    ["./GUE/tf/0"]="tf0.txt"
    ["./GUE/tf/1"]="tf1.txt"
    ["./GUE/tf/2"]="tf2.txt"
    ["./GUE/tf/3"]="tf3.txt"
    ["./GUE/tf/4"]="tf4.txt"

    #Splice
    ["./GUE/splice/reconstructed"]="splice.txt"

    #Prom
    ["./GUE/prom/prom_300_all"]="prom_300_all.txt"
    ["./GUE/prom/prom_300_notata"]="prom_300_notata.txt"
    ["./GUE/prom/prom_300_tata"]="prom_300_tata.txt"
    ["./GUE/prom/prom_core_all"]="prom_core_all.txt"
    ["./GUE/prom/prom_core_notata"]="prom_core_notata.txt"
    ["./GUE/prom/prom_core_tata"]="prom_core_tata.txt"

    #EMP
    ["./GUE/EMP/H3"]="H3.txt"
    ["./GUE/EMP/H3K4me1"]="H3K4me1.txt"
    ["./GUE/EMP/H3K4me2"]="H3K4me2.txt"
    ["./GUE/EMP/H3K4me3"]="H3K4me3.txt"
    ["./GUE/EMP/H3K9ac"]="H3K9ac.txt"
    ["./GUE/EMP/H3K14ac"]="H3K14ac.txt"
    ["./GUE/EMP/H3K36me3"]="H3K36me3.txt"
    ["./GUE/EMP/H3K79me3"]="H3K79me3.txt"
    ["./GUE/EMP/H4"]="H4.txt"
    ["./GUE/EMP/H4ac"]="H4ac.txt"

    #Covid
    ["./GUE/virus/covid"]="covid.txt"

)

# Loop through each dataset folder and its associated output file
for DATASET_FOLDER in "${!DATASET_OUTPUTS[@]}"; do
    OUTPUT_FILE="${DATASET_OUTPUTS[$DATASET_FOLDER]}"

    echo "python Train.py -dataset_folder \"$DATASET_FOLDER\" -pretrain_file \"$PRETRAIN_FILE\" --include_prev_context \"{False}\" \
    --gamma \"{0.1, 0.33, 0.5, 0.75, 1, 3, 5}\" --nb_train_iterations \"{1, 3, 5, 7, 10}\" --ratio_pretrain_train \"{0}\"\
    --handle_n_setting \"{remove}\" --ensemble_type \"{entropy}\" --num_threads \"{48}\" > \"$OUTPUT_DIR/$OUTPUT_FILE\""

    # Run the python script in the background
    #python3 explore.py -dataset_folder "$DATASET_FOLDER" -pretrain_file "$PRETRAIN_FILE" > "$OUTPUT_DIR/$OUTPUT_FILE" &
    python Train.py -dataset_folder "$DATASET_FOLDER" -pretrain_file "$PRETRAIN_FILE" --include_prev_context "{False}" \
    --gamma "{0.1, 0.33, 0.5, 0.75, 1, 3, 5}" --nb_train_iterations "{1, 3, 5, 7, 10}" --ratio_pretrain_train "{0}"\
    --handle_n_setting "{remove}" --ensemble_type "{entropy}" --num_threads "{64}" > "$OUTPUT_DIR/$OUTPUT_FILE"
done

# Wait for all background jobs to finish
wait
echo "All tasks are complete."
