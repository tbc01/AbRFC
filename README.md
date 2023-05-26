# AbRFC
Code and data to reproduce the AbRFC model from "Machine Learning-Guided Intelligent Antibody Engineering for Rapid Response to SARS-COV-2"

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
