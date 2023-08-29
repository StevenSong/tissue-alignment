#!/bin/bash

python src/train.py \
--output_dir /mnt/data5/spatial/runs/simsiam-all-slides \
--checkpoint_interval 100 \
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
