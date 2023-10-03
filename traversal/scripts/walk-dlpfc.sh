#!/bin/bash

DATA=data1
# DATA=data5

python walk.py \
--output-dir ~/figs/dlpfc-donor3-151673-norm-per-bucket \
--section dlpfc/donor3/151673 \
--data_root /mnt/$DATA/spatial/data \
--model triplet-dlpfc-0999
