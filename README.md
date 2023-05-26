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


## To generate the results, run the following commands:

- cross validation

cd AbRFC

python3 run.py 'cv'

- score

cd AbRFC

python3 run.py 'score'

All results will be saved in the afficlass folder.
