\# MAC-Unlearn: A Differentially Private Federated Edge Unlearning Framework to secure MAC De-randomization underlying 6G

This repository contains the implementation and analysis code for the paper.


\## Dataset

The Wi-Fi dataset used in this study is not publicly distributed.

Place `training\_ready\_wifi\_dataset.csv` in the project root before running experiments.



\## Setup


```bash

pip install -r requirements.txt


\## Data

Place training\_ready\_wifi\_dataset.csv in data/



\## Run Experiment:


\# Baselines

python experiments/run\_baselines.py --mlp-epochs 30



\# FedAvg / DP-FedAvg grid

python experiments/run\_fed\_grid.py --epsilons inf 8 4 2 --local-epochs 1 3 5



\# Unlearning (FedSF + SFU)

python experiments/run\_unlearning.py \\

         --data data/training\_ready\_wifi\_dataset.csv \\

         --forget-client 0 \\

         --retrain-epochs 12 \\

         --num-shards 4



\## Attacks (provide a trained model)

python experiments/run\_attacks.py --model-path results/models/fed\_model.pth

