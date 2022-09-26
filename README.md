# Mobile-Facility

## Setup

1. Ensure that Conda, Git, and Git-LFS are properly set up.
2. Use `conda env create -f environment.yml` to create a new environment called `facility` which has all of the dependencies installed.
3. `conda activate facility` to activate the conda environment

## Getting Started

1. Look at BasicStart.ipynb for more information on how to load data and run various heuristics/algorithms
2. mobile/config.py contains data processing and loading of our datasets
3. mobile/heuristics.py contains our basic baselines, algorithms, and heuristics

## Experiments

1. All experiments are contained in FinalExp.ipynb under notebooks
2. Experiment scripts that produce the runs graphed in FinalExp.ipynb are in scripts