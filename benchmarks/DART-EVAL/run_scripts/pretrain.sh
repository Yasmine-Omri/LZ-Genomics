

PYTHONPATH=$(pwd)
PRETRAIN_FILE=/home/nsagan/LZ-Genomics/benchmarks/DART-EVAL/data/pretrain_data/train.txt
#PRETRAIN_FILE="/Users/yasmineomri/Documents/EE376C/lz78_rust/repo/fresh_repo/LZ-Genomics/benchmarks/DART-EVAL/data/pretrain_data/small_dev_debug.txt"
OUTPUT_DIR="../outputs"

python ../python_scripts/pretrain.py \
  --pretrain_file $PRETRAIN_FILE \
  --limit 5G \
  --max_depth 12 \
  --include_prev_context False \
  --handle_n_setting remove \
  --output_dir best_spas > $OUTPUT_DIR/pretrain_out_12MD.txt