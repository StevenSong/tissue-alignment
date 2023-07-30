#!/bin/bash

python train.py \
--output_dir /mnt/data5/spatial/delete-me \
--tile_dir \
/mnt/data5/spatial/tiles/slide1 \
--backbone resnet50 \
--projector_hidden_dim 2048 \
--predictor_hidden_dim 512 \
--output_dim 2048 \
--batch_size 256 \
--lr 0.05 \
--num_epochs 100 \
--optimizer sgd \
--momentum 0.9 \
--weight_decay 1e-4 \
--scheduler cosine \
--num_workers 16


# /mnt/data5/spatial/tiles/slide3 \
# /mnt/data5/spatial/tiles/slide4 \
# --eval_tile_dir /mnt/data5/spatial/tiles/slide2 \
