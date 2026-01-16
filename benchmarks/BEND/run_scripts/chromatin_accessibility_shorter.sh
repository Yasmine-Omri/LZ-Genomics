

python ../python_scripts/train_multi_label_short_chrom.py \
  -dataset_folder "../data/chromatin_accessibility" \
  -pretrain_file "../../../dnabert_2_pretrain/dev.txt" \
  --include_prev_context "{False}" \
  --gamma "{0.1, 1, 3, 5}" \
  --nb_train_iterations "{1, 5}" \
  --handle_n_setting "{remove}" \
  --ratio_pretrain_train "{0}" \
  --ensemble_type "{entropy}" \
  --num_threads "{64}" \
  --validation_metric "auroc" \
  --max_depth "{12}" \
  > ../outputs/chromatin_accessibility_shorter.txt
