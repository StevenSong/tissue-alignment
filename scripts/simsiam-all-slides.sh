#!/bin/bash

python src/train.py \
--output_dir /mnt/data5/spatial/runs/simsiam-all-slides \
--checkpoint_interval 100 \
--data_paths \
/mnt/data5/spatial/tiles/slide1 \
/mnt/data5/spatial/tiles/slide2 \
/mnt/data5/spatial/tiles/slide3 \
/mnt/data5/spatial/tiles/slide4 \
--loader pathology/tile-simsiam \
--model_arch simsiam \
--model_params \
    backbone=resnet50 \
    projector_hidden_dim=2048 \
    predictor_hidden_dim=512 \
    output_dim=2048 \
--batch_size 256 \
--lr 0.05 \
--num_epochs 1000 \
--optimizer sgd \
--optimizer_params \
    momentum=0.9 \
    weight_decay=1e-4 \
--scheduler cosine \
--num_workers 16
