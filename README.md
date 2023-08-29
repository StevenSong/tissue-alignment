Run scripts from the root directory of this repo

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
└── stomach
    └── [...]
```
