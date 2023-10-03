#!/bin/bash

DATA=data1
# DATA=data5

# MODEL=dlpfc
# MODEL=dlpfc-augment
# MODEL=dlpfc-augment-slide-neg
# MODEL=dlpfc-slide-neg
# MODELS="dlpfc dlpfc-augment dlpfc-slide-neg dlpfc-augment-slide-neg"
MODELS="dlpfc-half-augment-slide-neg"

for MODEL in $MODELS; do
    python walk.py \
    --adjacency hex \
    --avg-expression path-clusters \
    --genes NEFH \
    --alignment-genes NEFH \
    --spot-frac 1 \
    --model triplet-$MODEL-0999 \
    --sections \
        dlpfc/donor3/151673 \
    --start_idxs \
        74 \
    --end_idxs \
        3547 \
    --data_root /mnt/$DATA/spatial/data \
    --output-dir ~/figs/$MODEL
done
