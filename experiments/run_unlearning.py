"""
Run unlearning experiments:
- FedSF (retrain excluding clients)
- SFU (sharded retraining)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_and_preprocess
from src.utils import set_seed, get_device, ensure_dir
from src.evaluate import evaluate_torch
from src.unlearning import fedsf_retrain_excluding, sfu_sharded_unlearning


def find_device_column(df):
    """
    Try to find a column that can be used to group by device (MAC address).
    Returns column name or None if not found.
    """
    possible_names = [
        'src_mac', 'source_mac', 'src_mac_enc', 'source_mac_enc',
        'mac', 'src_mac_address', 'device_id', 'device_mac'
    ]
    for col in possible_names:
        if col in df.columns:
            return col
    # fallback: look for any column containing 'mac'
    for col in df.columns:
        if 'mac' in col.lower():
            return col
    return None


def partition_by_mac(X, y, df, num_clients=10, seed=42):
    """
    Partition training data by source MAC address (device identity).
    Clients = unique devices, balanced as much as possible.
    """
    device_col = find_device_column(df)
    if device_col is None:
        print("Warning: No MAC/device column found. Falling back to random partitioning.")
        np.random.seed(seed)
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        parts = np.array_split(idx, num_clients)
        return [p.astype(int) for p in parts]

    print(f"Using column '{device_col}' for device-based partitioning")

    # Create DataFrame with indices
    df_idx = pd.DataFrame({
        'idx': np.arange(len(y)),
        'device': df[device_col].values
    })

    groups = df_idx.groupby('device')['idx'].apply(list).to_dict()
    device_indices = list(groups.values())

    num_devices = len(device_indices)

    client_parts = [[] for _ in range(num_clients)]

    for i, indices in enumerate(device_indices):
        client_parts[i % num_clients].extend(indices)

    client_parts = [np.array(part).astype(int) for part in client_parts]

    sizes = [len(p) for p in client_parts]
    print(f"Client sizes (by device grouping): {sizes}")
    print(f"Min: {min(sizes)}, Max: {max(sizes)}, Avg: {np.mean(sizes):.1f}")

    return client_parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training_ready_wifi_dataset.csv", type=str)
    parser.add_argument("--output-dir", default="results/unlearning", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--forget-client", type=int, default=0,
                        help="Client ID to forget (index in client_parts)")
    parser.add_argument("--retrain-epochs", type=int, default=10)
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    print("Device:", device)

    # Load full original DataFrame to access MAC column
    df = pd.read_csv(args.data)
    (X_train, y_train), _, (X_test, y_test), _ = load_and_preprocess(args.data)
    client_parts = partition_by_mac(X_train, y_train, df.iloc[:len(X_train)], 
                                   num_clients=args.num_clients, seed=args.seed)

    results = {}

    # FedSF: retrain excluding the forgotten client
    print(f"\n=== FedSF — forgetting client {args.forget_client} ===")
    t0 = time.time()
    model_fedsf = fedsf_retrain_excluding(
        X_train, y_train, client_parts, X_train.shape[1], device,
        forget_clients=[args.forget_client],
        retrain_rounds=args.retrain_epochs
    )
    metrics_fedsf = evaluate_torch(model_fedsf, X_test, y_test, device)
    results["FedSF"] = {**metrics_fedsf, "time": time.time() - t0}
    print(f"Acc: {metrics_fedsf['acc']:.4f}  F1: {metrics_fedsf['f1']:.4f}")

    # SFU: sharded unlearning
    print(f"\n=== SFU — forgetting client {args.forget_client} ===")
    t0 = time.time()
    shards = sfu_sharded_unlearning(
        X_train, y_train, client_parts, X_train.shape[1], device,
        forget_clients=[args.forget_client],
        num_shards=args.num_shards,
        retrain_epochs=args.retrain_epochs
    )

    # SFU inference: average logits across shards
    def sfu_predict(X):
        with torch.no_grad():
            logits_sum = None
            for m in shards:
                m.eval()
                l = m(torch.from_numpy(X).float().to(device))
                if logits_sum is None:
                    logits_sum = l
                else:
                    logits_sum += l
            avg_logits = logits_sum / len(shards)
            return torch.argmax(avg_logits, dim=1).cpu().numpy()

    preds_sfu = sfu_predict(X_test)
    acc_sfu = accuracy_score(y_test, preds_sfu)
    f1_sfu = f1_score(y_test, preds_sfu, zero_division=0)
    results["SFU"] = {"acc": acc_sfu, "f1": f1_sfu, "time": time.time() - t0}
    print(f"Acc: {acc_sfu:.4f}  F1: {f1_sfu:.4f}")

    # Save results
    out_path = Path(args.output_dir) / "unlearning_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()