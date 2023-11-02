Run scripts from the root directory of this repo.

Code for training the encoder model are contained in the `learn` subfolder with training scripts in `learn/scripts`.

Code for traversal and alignment are contained in the `traversal` subfolder with example scripts in `traversal/scripts`.

All code for generating the figures presented in the paper are contained within the `paper` subfolder. To examine the methods used to generate a figure, check out it's corresponding notebook.

Colon and stomach datasets are available upon reasonable request to the corresponding author.

Data should be organized as follows:
```
.
├── colon (organ)
│   ├── CD (disease or patient)
│   │   ├── A (slide)
│   │   │   ├── A.tif
│   │   │   └── outs (spaceranger output)
│   │   │       ├── filtered_feature_bc_matrix.h5
│   │   │       └── spatial
│   │   │           ├── scalefactors_json.json
│   │   │           ├── tissue_hires_image.png
│   │   │           └── tissue_positions_list.csv
│   │   └── B
│   │       ├── B.tif
│   │       └── outs
│   │           └── [...]
│   └── UC
│       └── [...]
└── dlpfc
    └── [...]
```
