#!/bin/bash

# DIR=results/depth-sweep-4-6-acc-sweep
DIR=kmer_train_reports_GC_content_only
FILENAMES=(
    "H3.txt"
    "H3K14ac.txt"
    "H3K36me3.txt"
    "H3K4me1.txt"
    "H3K4me2.txt"
    "H3K4me3.txt"
    "H3K79me3.txt"
    "H3K9ac.txt"
    "H4.txt"
    "H4ac.txt"
    "covid.txt"
    "mouse0.txt"
    "mouse1.txt"
    "mouse2.txt"
    "mouse3.txt"
    "mouse4.txt"
    "prom_300_all.txt"
    "prom_300_notata.txt"
    "prom_300_tata.txt"
    "prom_core_all.txt"
    "prom_core_notata.txt"
    "prom_core_tata.txt"
    "splice.txt"
    "tf0.txt"
    "tf1.txt"
    "tf2.txt"
    "tf3.txt"
    "tf4.txt"
)

INSERT_SPACE_AFTER=(
    "H3K9ac.txt"
    "H4ac.txt"
    "covid.txt"
    "mouse4.txt"
    "prom_300_tata.txt"
    "prom_core_tata.txt"
    "splice.txt"
)

# FILENAMES=(
#     "EMP.H3.txt"
#     "EMP.H3K14ac.txt"
#     "EMP.H3K36me3.txt"
#     "EMP.H3K4me1.txt"
#     "EMP.H3K4me2.txt"
#     "EMP.H3K4me3.txt"
#     "EMP.H3K79me3.txt"
#     "EMP.H3K9ac.txt"
#     "EMP.H4.txt"
#     "EMP.H4ac.txt"
#     "virus.covid.txt"
#     "mouse.0.txt"
#     "mouse.1.txt"
#     "mouse.2.txt"
#     "mouse.3.txt"
#     "mouse.4.txt"
#     "prom.prom_300_all.txt"
#     "prom.prom_300_notata.txt"
#     "prom.prom_300_tata.txt"
#     "prom.prom_core_all.txt"
#     "prom.prom_core_notata.txt"
#     "prom.prom_core_tata.txt"
#     "splice.reconstructed.txt"
#     "tf.0.txt"
#     "tf.1.txt"
#     "tf.2.txt"
#     "tf.3.txt"
#     "tf.4.txt"
# )

# INSERT_SPACE_AFTER=(
#     "EMP.H3K9ac.txt"
#     "EMP.H4ac.txt"
#     "virus.covid.txt"
#     "mouse.4.txt"
#     "prom.prom_300_tata.txt"
#     "prom.prom_core_tata.txt"
#     "splice.reconstructed.txt"
# )

# for file in "${FILENAMES[@]}"; do
#     METRIC="mcc"
#     if [[ "$file" == "virus.covid.txt" ]]; then
#         METRIC="f1"
#     fi

#     filepath="$DIR/$file"
#     if [[ -f "$filepath" ]]; then
#         line=$(grep -m 1 "^Final metric" "$filepath")
#         if [[ -n "$line" ]]; then
#             metric=$(echo "$line" | grep -o "Final metric .*" | sed "s/Final metric ($METRIC) with best hyperparameters: //")
#             # echo "File: $file -> METRIC = $metric"

#             # times 100 and 2 decimal places
#             # metric=$(printf "%.2f" "$(echo "$metric * 100" | bc -l)")
#             echo "$metric"
#         fi
#         if [[ " ${INSERT_SPACE_AFTER[*]} " == *" $file "* ]]; then
#             echo ""
#         fi
#     else
#         echo "File: $file does not exist."
#     fi
# done

for file in "${FILENAMES[@]}"; do
    METRIC="MCC"
    if [[ "$file" == "covid.txt" ]]; then
        METRIC="F1"
    fi

    filepath="$DIR/$file"
    if [[ -f "$filepath" ]]; then
        line=$(grep -m 1 "^VERIFY $METRIC" "$filepath")
        if [[ -n "$line" ]]; then
            metric=$(echo "$line" | grep -o "VERIFY $METRIC: .*" | sed "s/VERIFY $METRIC: //")
            # echo "File: $file -> METRIC = $metric"

            # times 100 and 2 decimal places
            metric=$(printf "%.2f" "$(echo "$metric * 100" | bc -l)")
            echo "$metric"
        fi
        if [[ " ${INSERT_SPACE_AFTER[*]} " == *" $file "* ]]; then
            echo ""
        fi
    else
        echo "File: $file does not exist."
    fi
done

# # GET THE TRAINING TIME
# for file in "${FILENAMES[@]}"; do
#     filepath="$DIR/$file"
#     if [[ -f "$filepath" ]]; then
#         line=$(grep -m 1 "^Total inference time:" "$filepath")
#         if [[ -n "$line" ]]; then
#             metric=$(echo "$line" | grep -o "Total inference time: .*" | sed "s/Total inference time: //")
#             # remove seconds
#             metric=$(echo "$metric" | sed "s/ seconds//")

#             echo "$metric"
#         fi
#         if [[ " ${INSERT_SPACE_AFTER[*]} " == *" $file "* ]]; then
#             echo ""
#         fi
#     else
#         echo "File: $file does not exist."
#     fi
# done

# GET THE TRAINING MEMORY
# for file in "${FILENAMES[@]}"; do
#     filepath="$DIR/$file"
#     if [[ -f "$filepath" ]]; then
#         python3 get_peak_memory.py "$filepath"
        
#     else
#         echo "File: $file does not exist."
#     fi

#     if [[ " ${INSERT_SPACE_AFTER[*]} " == *" $file "* ]]; then
#         echo ""
#     fi
# done