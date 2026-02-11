"""
Run attack evaluations:
- Membership Inference Attack (max confidence)
- Gradient Inversion Attack (tabular)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import load_and_preprocess
from src.utils import set_seed, get_device, ensure_dir
from src.attacks import membership_inference_maxconf, gradient_inversion_attack_tabular


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training_ready_wifi_dataset.csv")
    parser.add_argument("--model-path", default=None, help="Path to saved model state dict")
    parser.add_argument("--output-dir", default="results/attacks", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mia-samples", type=int, default=200)
    parser.add_argument("--gi-samples", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    print("Device:", device)

    _, _, (X_test, y_test), _ = load_and_preprocess(args.data)

    # Load model (you need to provide a trained model path)
    if not args.model_path:
        print("Error: --model-path is required (e.g. path to .pth file)")
        return

    from src.models import TorchMLP
    model = TorchMLP(input_dim=X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    results = {}

    # Membership Inference Attack
    print("\n=== Membership Inference Attack (max confidence) ===")
    # For simplicity, treat train as member, test as non-member
    _, _, (X_train, y_train), _ = load_and_preprocess(args.data)
    mia_auc, fpr, tpr = membership_inference_maxconf(
        model, X_train, y_train, X_test, y_test, device, sample_size=args.mia_samples
    )
    results["MIA"] = {
        "auc": float(mia_auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "samples": args.mia_samples
    }
    print(f"MIA AUC: {mia_auc:.4f}")

    # Gradient Inversion Attack
    print("\n=== Gradient Inversion Attack ===")
    mse_list, cos_list = [], []
    for i in range(min(args.gi_samples, len(X_test))):
        x = torch.from_numpy(X_test[i]).float()
        y = y_test[i]
        recon = gradient_inversion_attack_tabular(model, x, y, device)

        true_flat = X_test[i].flatten()
        recon_flat = recon.flatten()

        if np.isnan(recon_flat).any():
            mse = float('inf')
            cos = 0.0
        else:
            mse = np.mean((true_flat - recon_flat) ** 2)
            cos = np.dot(true_flat, recon_flat) / (
                np.linalg.norm(true_flat) * np.linalg.norm(recon_flat) + 1e-12
            )

        mse_list.append(mse)
        cos_list.append(cos)
        print(f"Sample {i+1}: MSE={mse:.6f}  CosSim={cos:.4f}")

    results["GradientInversion"] = {
        "avg_mse": float(np.mean(mse_list)),
        "avg_cossim": float(np.mean(cos_list)),
        "samples": len(mse_list)
    }

    # Save
    out_path = Path(args.output_dir) / "attack_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAttack results saved to: {out_path}")


if __name__ == "__main__":
    main()