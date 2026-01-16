# task4 (binary classification)

DATASET_FOLDER="../data/task4"
PRETRAIN_FILE="../../../dnabert_2_pretrain/dev.txt"
OUTPUT_DIR="../outputs"
PYTHONPATH=$(pwd)

for cell_type in "H1ESC" "GM12878" "K562" "IMR90" "HEPG2"; do
    echo "Processing cell type: $cell_type"
    python ../../../Train_cleaned_up.py \
    -dataset_folder "$DATASET_FOLDER/$cell_type" \
    -pretrain_file "$PRETRAIN_FILE" \
    --max_depth 12 \
    --include_prev_context False \
    --gamma 0.5 0.75 1 3 5 7 9 \
    --nb_train_iterations 1 3 5 7 9 \
    --ratio_pretrain_train 0 \
    --handle_n_setting remove \
    --ensemble_type entropy \
    --revcomp_augment_factor 0.2 \
    --validation_metric auroc \
    --test_metric auroc \
    --class_ratios 1 1 \
    --num_threads 64 > "$OUTPUT_DIR/task4_$cell_type.txt"
done
