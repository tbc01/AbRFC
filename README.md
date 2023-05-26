# AbRFC
Code and data to reproduce the AbRFC model results and scores from "Machine Learning-Guided Antibody Engineering That Leverages Domain Knowledge To Overcome The Small Data Problem"

## Requirements:

python=3.8

pip install scikit-learn

pip install seaborn

pip install openpyxl

## To generate the results, run the following commands:

- cross validation

cd AbRFC

python3 run.py 'cv'

- score

cd AbRFC

python3 run.py 'score'

All results will be saved in the afficlass folder.
