#!/bin/bash

python ../python_scripts/multilabel_train_new.py \
  --data_dir "../data/histone_modification" \
  --n_labels 18 \
  --spa_dir /media/hdd1/histone_new \
  --gamma 0.1 1 3 5 7 \
  --nb_train_iterations 1 \
  --inf_threads 16 \
  --job_threads 8 \
  --val_metric auroc \
  --max_depth 12 \
  > ../outputs/histone_new_4.txt