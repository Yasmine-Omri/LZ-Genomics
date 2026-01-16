#!/bin/bash

# Define the pretrain file and dataset folders with their associated output files
PRETRAIN_FILE="./dnabert_2_pretrain/dev.txt"
OUTPUT_DIR="./Tang_Yoo_reports"

   
# Define the dataset folders and their respective output files
declare -A DATASET_OUTPUTS=(
    ["./LZ78_Data_Tang_Yoo/Arid3"]="Arid3.txt"
    ["./LZ78_Data_Tang_Yoo/ATF2"]="ATF2.txt"
    ["./LZ78_Data_Tang_Yoo/BACH1"]="BACH1.txt"
    ["./LZ78_Data_Tang_Yoo/CTCF"]="CTCF.txt"
    ["./LZ78_Data_Tang_Yoo/ELK1"]="ELK1.txt"
    ["./LZ78_Data_Tang_Yoo/GABPA"]="GABPA.txt"
    ["./LZ78_Data_Tang_Yoo/MAX"]="MAX.txt"
    ["./LZ78_Data_Tang_Yoo/REST"]="REST.txt"
    ["./LZ78_Data_Tang_Yoo/SRF"]="SRF.txt"
    ["./LZ78_Data_Tang_Yoo/ZNF24"]="ZNF24.txt"
)

# Loop through each dataset folder and its associated output file
for DATASET_FOLDER in "${!DATASET_OUTPUTS[@]}"; do
    OUTPUT_FILE="${DATASET_OUTPUTS[$DATASET_FOLDER]}"

    echo "python Train_cleaned_up.py -dataset_folder \"$DATASET_FOLDER\" -pretrain_file \"$PRETRAIN_FILE\" --include_prev_context False \
    --gamma 0.1 0.33 0.5 0.75 1 3 5 --nb_train_iterations 1 3 5 7 10 --ratio_pretrain_train 0 \
    --handle_n_setting remove --ensemble_type entropy --num_threads 64 --validation_metric auroc --test_metric auroc > \"$OUTPUT_DIR/$OUTPUT_FILE\""

    python Train_cleaned_up.py -dataset_folder "$DATASET_FOLDER" -pretrain_file "$PRETRAIN_FILE" --include_prev_context False \
    --gamma 0.1 0.33 0.5 0.75 1 3 5 --nb_train_iterations 1 3 5 7 10 --ratio_pretrain_train 0 \
    --handle_n_setting remove --ensemble_type entropy --num_threads 64 --validation_metric auroc --test_metric auroc > "$OUTPUT_DIR/$OUTPUT_FILE"
done

# Wait for all background jobs to finish
wait
echo "All tasks are complete."
