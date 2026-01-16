#!/bin/bash
# run_all.sh

# List of datasets you want to train on
DATASETS=("human_nontata_promoters" "human_enhancers_cohn" "human_enhancers_ensembl" "human_ocr_ensembl" "human_ensembl_regulatory" "dummy_mouse_enhancers_ensembl" "drosophila_enhancers_stark" "demo_coding_vs_intergenomic_seqs" "demo_human_or_worm")


# Loop over datasets
for DATASET in "${DATASETS[@]}"; do
    echo ">>> Running dataset: $DATASET"
    python ../python_scripts/Train.py \
      -dataset_folder "../data/$DATASET" \
        -pretrain_file "../../../dnabert_2_pretrain/dev.txt" \
        --include_prev_context False \
        --gamma 0.1 0.33 0.5 0.75 1 3 5 \
        --nb_train_iterations 1 3 5 7 10 \
        --handle_n_setting remove \
        --ratio_pretrain_train 0 \
        --ensemble_type entropy \
        --num_threads 64 \
        --validation_metric accuracy \
        --max_depth 12 \
      > "../outputs/${DATASET}.txt"
done

