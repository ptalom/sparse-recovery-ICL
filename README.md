#### This repo contains the code of our paper : 
### How Are LLMs Sensitive to Coherence In-context? A Study on Sparse Recovery and Matrix Factorization

## Goals of the project
We test the performance of Transformers by changing our coherence controller τ ∈ [0, 1] :
- τ = 0 → for a full random gaussian measurement matrix (maximum incoherence)
- τ = 0.5 → 50% coherence and 50% incoherence in our measurement matrix
- τ = 1 → a maximum coherence measurement matrix


## Setup
```bash
git clone https://github.com/ptalom/sparse-recovery-ICL/
cd src
```

## Getting started
Check the `requirements.txt` file to configure the appropriate environment

## To change configurations
- For Sparse Recovery task, edit the yaml configuration file : `src/conf/compressed_sensing.yaml`
- For Matrix Factorization task, edit the yaml configuration file : `src/conf/matrix_factorization.yaml`


## Start an experiment
`train.py` takes as argument a configuration yaml from conf and trains the corresponding model. So you can try:

For sparse recovery training 
```bash
python train.py --config conf/compressed_sensing.yaml 
```
For matrix factorization
```bash
python train.py --config conf/matrix_factorization.yaml
```
The evaluation part will start automatically after the training.
`The eval.ipynb` notebook contains code to load our own pre-trained models and plot the pre-computed metrics


## Code organisation
- `train.py`  →  main script for training
- `models.py` → models defintion and baselines
- `eval.py` → evaluation script
- `tasks.py` → task definitions (sparse recovery, matrix factorization)
- `src/conf/` → configuration files (YAML) to run different experiments
- `samplers.py` → data generation scripts
- `sparse_recovery/compressed_sensing.py` (resp. `sparse_recovery/matrix_factorization.py`) → all functions for matrix creation, base Φ creation etc.
- `models/sparse_recovery` → directories containing checkpoints & results (`metrics.json`, generate after evaluation part)
- `models/matrix_factorization` → directories containing checkpoints & results (`metrics.json`, generate after evaluation part)

## Contributors
- Patrick C. Talom
- Pascal Jr Notsawo Tikeng

Inspired by the article : *"What Can Transformers Learn In-Context? A Case Study of Simple Function Classes" (Garg et al., 2022)"*