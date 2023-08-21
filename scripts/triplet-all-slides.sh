#!/bin/bash

python src/train.py \
--output_dir /mnt/data5/spatial/runs/triplet-all-slides \
--checkpoint_interval 100 \
--data_paths \
    /mnt/data5/spatial/tiles/slide1/A1 \
    /mnt/data5/spatial/tiles/slide1/B1 \
    /mnt/data5/spatial/tiles/slide1/C1 \
    /mnt/data5/spatial/tiles/slide1/D1 \
    /mnt/data5/spatial/tiles/slide2/A1 \
    /mnt/data5/spatial/tiles/slide2/B1 \
    /mnt/data5/spatial/tiles/slide2/C1 \
    /mnt/data5/spatial/tiles/slide2/D1 \
    /mnt/data5/spatial/tiles/slide3/A1 \
    /mnt/data5/spatial/tiles/slide3/B1 \
    /mnt/data5/spatial/tiles/slide3/C1 \
    /mnt/data5/spatial/tiles/slide3/D1 \
    /mnt/data5/spatial/tiles/slide4/A1 \
    /mnt/data5/spatial/tiles/slide4/B1 \
    /mnt/data5/spatial/tiles/slide4/C1 \
    /mnt/data5/spatial/tiles/slide4/D1 \
--loader pathology/adj-tile-triplet \
--loader_params \
    position_table=/mnt/data5/spatial/count/slide1/A1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide1/B1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide1/C1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide1/D1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide2/A1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide2/B1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide2/C1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide2/D1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide3/A1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide3/B1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide3/C1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide3/D1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide4/A1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide4/B1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide4/C1/outs/spatial/tissue_positions_list.csv \
    position_table=/mnt/data5/spatial/count/slide4/D1/outs/spatial/tissue_positions_list.csv \
--model_arch triplet \
--model_params \
    backbone=resnet50 \
    projector_hidden_dim=2048 \
    output_dim=2048 \
--batch_size 192 \
--lr 0.05 \
--num_epochs 1000 \
--optimizer sgd \
--optimizer_params \
    momentum=0.9 \
    weight_decay=1e-4 \
--scheduler cosine \
--num_workers 16
