#!/bin/bash

DATA=data1
# DATA=data5



python walk.py \
--output-dir ~/figs/colon-UC-B \
--section colon/UC/B \
--data_root /mnt/$DATA/spatial/data \
--model triplet-gi-0999
