#!/bin/bash

DATA=data1 # cytokine
# DATA=data5 # antibody

python train.py \
--output_dir /mnt/$DATA/spatial/runs/triplet-gi \
--checkpoint_interval 100 \
--num_epochs 1000 \
--data_paths \
    /mnt/$DATA/spatial/data/colon/CD/A/tiles \
    /mnt/$DATA/spatial/data/colon/CD/B/tiles \
    /mnt/$DATA/spatial/data/colon/CD/C/tiles \
    /mnt/$DATA/spatial/data/colon/CD/D/tiles \
    /mnt/$DATA/spatial/data/colon/UC/A/tiles \
    /mnt/$DATA/spatial/data/colon/UC/B/tiles \
    /mnt/$DATA/spatial/data/colon/UC/C/tiles \
    /mnt/$DATA/spatial/data/colon/UC/D/tiles \
    /mnt/$DATA/spatial/data/colon/normal/A/tiles \
    /mnt/$DATA/spatial/data/colon/C.diff/A/tiles \
    /mnt/$DATA/spatial/data/colon/C.diff/B/tiles \
    /mnt/$DATA/spatial/data/colon/C.diff/C/tiles \
    /mnt/$DATA/spatial/data/stomach/normal/A/tiles \
    /mnt/$DATA/spatial/data/stomach/H.pylori/A/tiles \
    /mnt/$DATA/spatial/data/stomach/H.pylori/B/tiles \
    /mnt/$DATA/spatial/data/stomach/H.pylori/C/tiles \
--loader pathology/adj-tile-triplet \
--loader_params \
    augment=0 \
    in_slide_neg=0 \
    position_table=/mnt/$DATA/spatial/data/colon/CD/A/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/colon/CD/B/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/colon/CD/C/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/colon/CD/D/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/colon/UC/A/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/colon/UC/B/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/colon/UC/C/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/colon/UC/D/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/colon/normal/A/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/colon/C.diff/A/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/colon/C.diff/B/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/colon/C.diff/C/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/stomach/normal/A/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/stomach/H.pylori/A/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/stomach/H.pylori/B/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/$DATA/spatial/data/stomach/H.pylori/C/outs/spatial/tissue_positions_list.csv \
--model_arch triplet \
--model_params \
    backbone=resnet50 \
    projector_hidden_dim=2048 \
    output_dim=2048 \
--batch_size 768 \
--lr 0.05 \
--optimizer sgd \
--optimizer_params \
    momentum=0.9 \
    weight_decay=1e-4 \
--scheduler cosine \
--num_workers 16
