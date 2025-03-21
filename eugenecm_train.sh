#!/bin/bash

# Define the pretrain file and dataset folders with their associated output files
PRETRAIN_FILE="/Users/eugenemin/LZ-Genomics/dnabert_2_pretrain/dev.txt"
OUTPUT_DIR="/Users/eugenemin/LZ-Genomics/outputs"

   
# Define the dataset folders and their respective output files
declare -A DATASET_OUTPUTS=(
     #Mouse
    ["/Users/eugenemin/LZ-Genomics/GUE/mouse/0"]="mouse0.txt"
    # ["/home/users/yomri/Documents/EE376C/lz78_rust/GUE/mouse/1"]="mouse1.txt"
    # ["/home/users/yomri/Documents/EE376C/lz78_rust/GUE/mouse/2"]="mouse2.txt"
    # ["/home/users/yomri/Documents/EE376C/lz78_rust/GUE/mouse/3"]="mouse3.txt"
    # ["/home/users/yomri/Documents/EE376C/lz78_rust/GUE/mouse/4"]="mouse4.txt"
    #EMP
    # ["/Users/yasmineomri/Documents/EE376C/lz78_rust/GUE/mouse/0"]="mouse0.txt"
    # ["/Users/yasmineomri/Documents/EE376C/lz78_rust/GUE/EMP/H3"]="EMP_H3.txt"
    # ["/Users/yasmineomri/Documents/EE376C/lz78_rust/GUE/EMP/H3K4me1"]="EMP_H3K4me1.txt"
    # ["/Users/yasmineomri/Documents/EE376C/lz78_rust/GUE/EMP/H3K4me2"]="EMP_H3K4me2.txt"
    # ["/Users/yasmineomri/Documents/EE376C/lz78_rust/GUE/EMP/H3K4me3"]="EMP_H3K4me3.txt"
    # ["/Users/yasmineomri/Documents/EE376C/lz78_rust/GUE/EMP/H3K9ac"]="EMP_H3K9ac.txt"
    # ["/Users/yasmineomri/Documents/EE376C/lz78_rust/GUE/EMP/H3K14ac"]="EMP_H3K14ac.txt"
    # ["/Users/yasmineomri/Documents/EE376C/lz78_rust/GUE/EMP/H3K36me3"]="EMP_H3K36me3.txt"
    # ["/Users/yasmineomri/Documents/EE376C/lz78_rust/GUE/EMP/H3K79me3"]="EMP_H3K79me3.txt"
    # ["/Users/yasmineomri/Documents/EE376C/lz78_rust/GUE/EMP/H4"]="EMP_H4.txt"
    # ["/Users/yasmineomri/Documents/EE376C/lz78_rust/GUE/EMP/H4ac"]="EMP_H4ac.txt"
    #TODO ADD REST OF THE DATASETS (given long run times, it's better to split the datasets across all of us + multiple machines)
)

# Loop through each dataset folder and its associated output file
for DATASET_FOLDER in "${!DATASET_OUTPUTS[@]}"; do
    OUTPUT_FILE="${DATASET_OUTPUTS[$DATASET_FOLDER]}"

    # Run the python script in the background
    #python3 explore.py -dataset_folder "$DATASET_FOLDER" -pretrain_file "$PRETRAIN_FILE" > "$OUTPUT_DIR/$OUTPUT_FILE" &
    python Train.py -dataset_folder "$DATASET_FOLDER" -pretrain_file "$PRETRAIN_FILE" --include_prev_context "{True, False}" \
    --gamma "{0.1, 0.33, 0.5, 0.75, 1, 3, 5}" --nb_train_iterations "{1, 3, 5, 7, 10}" --ratio_pretrain_train "{0}"\
    --handle_n_setting "{remove}" --ensemble_type "{depth, entropy}" --num_threads "{1}" > "$OUTPUT_DIR/$OUTPUT_FILE" &
done

# Wait for all background jobs to finish
wait
echo "All tasks are complete."


