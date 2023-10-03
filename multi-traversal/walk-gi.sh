#!/bin/bash

DATA=data1
# DATA=data5

python walk.py \
--adjacency hex \
--avg-expression path-clusters \
--genes CD3D PTPRC EPCAM ACTA2 \
--alignment-genes EPCAM ACTA2 \
--spot-frac 1 \
--model triplet-gi-0999 \
--sections \
    colon/CD/A \
    colon/CD/B \
    colon/CD/C \
    colon/CD/D \
--data_root /mnt/$DATA/spatial/data \
--output-dir ~/figs/CD
