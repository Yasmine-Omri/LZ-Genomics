#/bin/bash

OUTPUT_DIR=grover_ft/model
TEXT_OUTPUT_DIR=grover_ft/text
mkdir -p $TEXT_OUTPUT_DIR
DEVICES=6,7

# # Finetue Grover on EMP tasks
# # H3  H3K14ac  H3K36me3  H3K4me1  H3K4me2  H3K4me3  H3K79me3  H3K9ac  H4  H4ac
# for DATASET in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
# do
#     CUDA_VISIBLE_DEVICES=$DEVICES  python finetune_grover.py \
#         --dataset_folder GUE/EMP/$DATASET \
#         --output_dir $OUTPUT_DIR/EMP/$DATASET > $TEXT_OUTPUT_DIR/EMP_$DATASET.txt
#     rm -rf $OUTPUT_DIR/EMP/$DATASET
# done

# # Mouse tasks
# for DATASET in 0 1 2 3 4
# do
#     CUDA_VISIBLE_DEVICES=$DEVICES  python finetune_grover.py \
#         --dataset_folder GUE/mouse/$DATASET \
#         --output_dir $OUTPUT_DIR/mouse/$DATASET > $TEXT_OUTPUT_DIR/mouse_$DATASET.txt
#     rm -rf $OUTPUT_DIR/mouse/$DATASET
# done

# # virus covid
CUDA_VISIBLE_DEVICES=$DEVICES  python finetune_grover.py \
    --dataset_folder GUE/virus/covid \
    --learning_rate 1e-5 \
    --output_dir $OUTPUT_DIR/virus/covid > $TEXT_OUTPUT_DIR/virus_covid.txt
rm -rf $OUTPUT_DIR/virus/covid

# # prom tasks
# for DATASET in prom_300_all prom_300_notata prom_300_tata prom_core_all prom_core_notata prom_core_tata
# do
#     CUDA_VISIBLE_DEVICES=$DEVICES  python finetune_grover.py \
#         --dataset_folder GUE/prom/$DATASET \
#         --output_dir $OUTPUT_DIR/prom/$DATASET > $TEXT_OUTPUT_DIR/prom_$DATASET.txt
#     rm -rf $OUTPUT_DIR/prom/$DATASET
# done

# # splice/reconstructed
# CUDA_VISIBLE_DEVICES=$DEVICES  python finetune_grover.py \
#     --dataset_folder GUE/splice/reconstructed \
#     --output_dir $OUTPUT_DIR/splice/reconstructed > $TEXT_OUTPUT_DIR/splice_reconstructed.txt
# rm -rf $OUTPUT_DIR/splice/reconstructed

# # tf
# for DATASET in 0 1 2 3 4
# do
#     CUDA_VISIBLE_DEVICES=$DEVICES  python finetune_grover.py \
#         --dataset_folder GUE/tf/$DATASET \
#         --output_dir $OUTPUT_DIR/tf/$DATASET > $TEXT_OUTPUT_DIR/tf_$DATASET.txt
#     rm -rf $OUTPUT_DIR/tf/$DATASET
# done