"""
Run centralized baselines:
- Logistic Regression
- Centralized MLP
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_and_preprocess
from src.models import TorchMLP
from src.utils import set_seed, get_device, ensure_dir
from src.evaluate import evaluate_torch


def train_logreg(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    return acc, f1


def train_centralized_mlp(X_train, y_train, X_val, y_val, X_test, y_test, device, epochs=20):
    model = TorchMLP(input_dim=X_train.shape[1]).to(device)

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long()
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    best_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_torch(model, X_val, y_val, device)
        print(f"Epoch {epoch+1:2d}  Val acc: {val_metrics['acc']:.4f}  Val f1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    test_metrics = evaluate_torch(model, X_test, y_test, device)
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training_ready_wifi_dataset.csv", type=str)
    parser.add_argument("--output-dir", default="results/baselines", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp-epochs", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    print("Device:", device)

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = load_and_preprocess(args.data)

    results = {}

    # Logistic Regression
    print("\n=== Logistic Regression ===")
    t0 = time.time()
    acc_lr, f1_lr = train_logreg(X_train, y_train, X_test, y_test)
    results["LogReg"] = {"acc": acc_lr, "f1": f1_lr, "time": time.time() - t0}
    print(f"Acc: {acc_lr:.4f}  F1: {f1_lr:.4f}")

    # Centralized MLP
    print("\n=== Centralized MLP ===")
    t0 = time.time()
    mlp_metrics = train_centralized_mlp(
        X_train, y_train, X_val, y_val, X_test, y_test,
        device=device, epochs=args.mlp_epochs
    )
    results["MLP-central"] = {**mlp_metrics, "time": time.time() - t0}
    print(f"Acc: {mlp_metrics['acc']:.4f}  F1: {mlp_metrics['f1']:.4f}")

    # Save results
    out_path = Path(args.output_dir) / "baselines_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()