#/bin/bash

CUDA_VISIBLE_DEVICES=5,6,7  python finetune_grover.py \
    --dataset_folder GUE/splice/reconstructed \
    --output_dir results/grover_ft/splice/
