#!/bin/bash

python src/train.py \
--output_dir /mnt/data5/spatial/runs/triplet-dlpfc \
--checkpoint_interval 100 \
--num_epochs 1000 \
--data_paths \
    /mnt/data5/spatial/data/dlpfc/donor1/151507/tiles \
    /mnt/data5/spatial/data/dlpfc/donor1/151508/tiles \
    /mnt/data5/spatial/data/dlpfc/donor1/151509/tiles \
    /mnt/data5/spatial/data/dlpfc/donor1/151510/tiles \
    /mnt/data5/spatial/data/dlpfc/donor2/151669/tiles \
    /mnt/data5/spatial/data/dlpfc/donor2/151670/tiles \
    /mnt/data5/spatial/data/dlpfc/donor2/151671/tiles \
    /mnt/data5/spatial/data/dlpfc/donor2/151672/tiles \
    /mnt/data5/spatial/data/dlpfc/donor3/151673/tiles \
    /mnt/data5/spatial/data/dlpfc/donor3/151674/tiles \
    /mnt/data5/spatial/data/dlpfc/donor3/151675/tiles \
    /mnt/data5/spatial/data/dlpfc/donor3/151676/tiles \
--loader pathology/adj-tile-triplet \
--loader_params \
    augment=1 \
    position_table=/mnt/data5/spatial/data/dlpfc/donor1/151507/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/dlpfc/donor1/151508/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/dlpfc/donor1/151509/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/dlpfc/donor1/151510/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/dlpfc/donor2/151669/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/dlpfc/donor2/151670/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/dlpfc/donor2/151671/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/dlpfc/donor2/151672/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/dlpfc/donor3/151673/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/dlpfc/donor3/151674/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/dlpfc/donor3/151675/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/data/dlpfc/donor3/151676/outs/spatial/tissue_positions_list.csv \
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
