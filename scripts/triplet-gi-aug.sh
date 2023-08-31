#!/bin/bash

python src/train.py \
--output_dir /mnt/data5/spatial/runs/triplet-gi-aug \
--checkpoint_interval 10 \
--num_epochs 100 \
--data_paths \
    /mnt/data5/spatial/data/colon/CD/A/tiles \
    /mnt/data5/spatial/data/colon/CD/B/tiles \
    /mnt/data5/spatial/data/colon/CD/C/tiles \
    /mnt/data5/spatial/data/colon/CD/D/tiles \
    /mnt/data5/spatial/data/colon/UC/A/tiles \
    /mnt/data5/spatial/data/colon/UC/B/tiles \
    /mnt/data5/spatial/data/colon/UC/C/tiles \
    /mnt/data5/spatial/data/colon/UC/D/tiles \
    /mnt/data5/spatial/data/colon/normal/A/tiles \
    /mnt/data5/spatial/data/colon/C.diff/A/tiles \
    /mnt/data5/spatial/data/colon/C.diff/B/tiles \
    /mnt/data5/spatial/data/colon/C.diff/C/tiles \
    /mnt/data5/spatial/data/stomach/normal/A/tiles \
    /mnt/data5/spatial/data/stomach/H.pylori/A/tiles \
    /mnt/data5/spatial/data/stomach/H.pylori/B/tiles \
    /mnt/data5/spatial/data/stomach/H.pylori/C/tiles \
--loader pathology/adj-tile-triplet \
--loader_params \
    position_table=/mnt/data5/spatial/data/colon/CD/A/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/colon/CD/B/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/colon/CD/C/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/colon/CD/D/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/colon/UC/A/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/colon/UC/B/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/colon/UC/C/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/colon/UC/D/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/colon/normal/A/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/colon/C.diff/A/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/colon/C.diff/B/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/colon/C.diff/C/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/stomach/normal/A/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/stomach/H.pylori/A/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/stomach/H.pylori/B/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/stomach/H.pylori/C/outs/spatial/tissue_positions_list.csv \
    augment=1 \
--model_arch triplet \
--model_params \
    backbone=resnet50 \
    projector_hidden_dim=2048 \
    output_dim=2048 \
--batch_size 192 \
--lr 0.05 \
--optimizer sgd \
--optimizer_params \
    momentum=0.9 \
    weight_decay=1e-4 \
--scheduler cosine \
--num_workers 16
