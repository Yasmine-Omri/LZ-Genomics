'''
task1 (binary classification):

Note: 
- the max tree depth is set to 16 (feel free to update)
- the minimal sweep is used (including entropy for ensemble heuristic)
- The Ns are removed (however, note that they are more substantial here, over 800). Might want to think about deterministically replacing them?

'''

DATASET_FOLDER="../data/task1"
PRETRAIN_FILE="../../../dnabert_2_pretrain/dev.txt"
OUTPUT_DIR="../outputs"
PYTHONPATH=$(pwd)

python ../python_scripts/task1.py --max_depth 16 -dataset_folder "$DATASET_FOLDER" -pretrain_file "$PRETRAIN_FILE" --include_prev_context "{False}" --gamma "{0.1, 0.33, 0.5, 0.75, 1, 3, 5}" --nb_train_iterations "{1, 3, 5, 7, 10}" --ratio_pretrain_train "{0}" --handle_n_setting "{remove}" --ensemble_type "{entropy}" --num_threads "{64}" > "$OUTPUT_DIR/task1.txt"
