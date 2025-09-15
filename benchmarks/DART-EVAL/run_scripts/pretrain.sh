

PYTHONPATH=$(pwd)
PRETRAIN_FILE=<path to the train.txt (the 30GBS downloaded file. Should be in data/pretrain_data after running download command in README.md)
#PRETRAIN_FILE="/Users/yasmineomri/Documents/EE376C/lz78_rust/repo/fresh_repo/LZ-Genomics/benchmarks/DART-EVAL/data/pretrain_data/small_dev_debug.txt"
OUTPUT_DIR="../outputs"

python ../python_scripts/pretrain.py \
  --pretrain_file $PRETRAIN_FILE \
  --limit 5G \
  --max_depth 16 \
  --include_prev_context False \
  --handle_n_setting remove \
  --output_dir best_spas > $OUTPUT_DIR/pretrain_out.txt