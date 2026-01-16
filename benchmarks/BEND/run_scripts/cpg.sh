

python ../python_scripts/train_multi_label.py \
  -dataset_folder "../data/cpg" \
  -pretrain_file "../../../dnabert_2_pretrain/dev.txt" \
  --include_prev_context "{False}" \
  --gamma "{0.1, 0.33, 0.5, 0.75, 1, 3, 5}" \
  --nb_train_iterations "{1, 3, 5, 7, 10}" \
  --handle_n_setting "{remove}" \
  --ratio_pretrain_train "{0}" \
  --ensemble_type "{entropy}" \
  --num_threads "{64}" \
  --validation_metric "auroc" \
  --max_depth "{12}" \
  > ../outputs/cpg.txt
