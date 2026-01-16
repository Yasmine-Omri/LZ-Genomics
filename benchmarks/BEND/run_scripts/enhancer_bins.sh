#!/bin/bash
# enhancer_bins training script

OUTPUT_DIR="../outputs"
DATASET="../data/enhancer_annotation/enhancer_bins.csv"
#DATASET="/Users/yasmineomri/Documents/EE376C/lz78_rust/repo/rebuttal_bench/BEND/scripts/enhancer_bins.csv"
PRETRAIN="../../../dnabert_2_pretrain/dev.txt"

python ../python_scripts/enhancer_bin_flanks.py \
  -dataset_file "$DATASET" \
  -pretrain_file "$PRETRAIN" \
  --include_prev_context "{False}" \
  --gamma "{0.1, 0.33, 0.5, 0.75, 1, 3, 5}" \
  --nb_train_iterations "{1, 3, 5, 7, 10}" \
  --handle_n_setting "{remove}" \
  --ratio_pretrain_train "{0}" \
  --ensemble_type "{entropy}" \
  --num_threads "{64}" \
  --metric "auprc" \
  --max_depth "{12}" \
  > "$OUTPUT_DIR/enhancer_bins.txt"
