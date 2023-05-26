# AbRFC
Code and data to reproduce the AbRFC model results and scores from "Machine Learning-Guided Antibody Engineering That Leverages Domain Knowledge To Overcome The Small Data Problem"

## Requirements:
conda or miniconda https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

python=3.8

pip install scikit-learn

pip install seaborn

pip install openpyxl

Code was tested on a 2022 Macbook Pro with an Apple M2 chip, 8GB Ram, running OS Ventura 13.13.1.  Python was run in a terminal using Rosetta 2 to allow the use of programs/libraries currently not adapted for Apple Silicon.

## Installation instructions
1. create a conda environment `conda create -n "AbRFC" python=3.8`
2. clone this repository and change directories after cloning: `cd AbRFC`
3. activate the conda environment `conda activate AbRFC`
4. install dependencies `pip install -r requirements.txt` 

## To Replicate results from the paper:
**All commands are run from the /path/to/AbRFC/ directory**
### Figure 2/S1
- AA Bias in training/validation sets (Figure 2A, 2B)
`python run.py aabias`
outputs: `outpus/Figure2AB.pdf`
- cross validation
`python run.py cv`
outputs: `outputs/FigureS1.pdf,outputs/Figure2C.pdf, outputs/Figure2D.pdf`

- validation benchmarking
`python run.py validation`
outputs: `outputs/Figure2E.pdf`

- score
`python run.py score`
outputs: `outputs/scores_CMAB.csv`, `outputs/scores_GMAB.csv`

