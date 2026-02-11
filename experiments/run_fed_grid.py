"""
Run federated learning grid:
- Vary local_epochs and epsilon (DP strength)
- Save results table
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data import load_and_preprocess
from src.utils import set_seed, get_device, ensure_dir
from src.federated import fedavg_train
from src.evaluate import evaluate_torch


def partition_random(X, y, num_clients=10, seed=42):
    np.random.seed(seed)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    parts = np.array_split(idx, num_clients)
    return [p.astype(int) for p in parts]


def run_grid(args):
    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    print("Device:", device)

    (X_train, y_train), _, (X_test, y_test), _ = load_and_preprocess(args.data)

    client_parts = partition_random(X_train, y_train, num_clients=args.num_clients)

    results = []

    for local_ep in args.local_epochs:
        for eps in args.epsilons:
            print(f"\n=== local_epochs={local_ep}   epsilon={eps} ===")
            t0 = time.time()

            model = fedavg_train(
                X_train, y_train, client_parts, X_train.shape[1], device,
                global_rounds=args.rounds,
                clients_per_round=args.clients_per_round,
                local_epochs=local_ep,
                dp_epsilon=eps
            )

            metrics = evaluate_torch(model, X_test, y_test, device)
            duration = time.time() - t0

            row = {
                "Method": f"DP-FedAvg" if np.isfinite(eps) else "FedAvg",
                "local_epochs": local_ep,
                "epsilon": eps if np.isfinite(eps) else "inf",
                **metrics,
                "time_s": duration
            }
            results.append(row)

            print(f"Acc: {metrics['acc']:.4f}  F1: {metrics['f1']:.4f}  Time: {duration:.1f}s")

    # Save
    import pandas as pd
    df = pd.DataFrame(results)
    out_csv = Path(args.output_dir) / "fed_grid_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nGrid results saved to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training_ready_wifi_dataset.csv")
    parser.add_argument("--output-dir", default="results/fed_grid", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients-per-round", type=float, default=0.4)
    parser.add_argument("--local-epochs", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[float("inf"), 8.0, 4.0, 2.0])
    args = parser.parse_args()

    run_grid(args)