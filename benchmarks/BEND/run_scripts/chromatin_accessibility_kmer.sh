#!/bin/bash

python ../python_scripts/multilabel_train_new_kmer.py \
  --data_dir "../data/chromatin_accessibility" \
  --n_labels 125 \
  --spa_dir /media/hdd1/chromatin_accessibility_kmer \
  --ks 6 9 12 \
  --alphas 0.5 1.0 \
  --feature_modes count \
  --inf_threads 16 \
  --job_threads 8 \
  --val_metric auroc \
  --canonical \
  > ../outputs/chromatin_accessibility_kmer.txt