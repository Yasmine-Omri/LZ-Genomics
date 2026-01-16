'''
check shuffling is off + check max depth
'''

python ../python_scripts/Train.py \
  -dataset_folder "../data/GUE_v2/fungi/species_20" \
  -pretrain_file "../../../dnabert_2_pretrain/dev.txt" \
  --include_prev_context "{False}" \
  --gamma "{0.1, 0.33, 0.5, 0.75, 1, 3, 5}" \
  --nb_train_iterations "{1, 3, 5, 7, 10}" \
  --handle_n_setting "{remove}" \
  --ratio_pretrain_train "{0}" \
  --ensemble_type "{entropy}" \
  --num_threads "{64}" \
  --validation_metric "mcc" \
  --augmentation_factors "{0}" \
  --shuffle_preserve_kmer 2 \
  --max_depth "{8, 16}" \
  > ../outputs/fungi.txt
