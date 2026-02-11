import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(
    path="data/training_ready_wifi_dataset.csv",
    target_col="label",
    test_size=0.2,
    val_size=0.1,
    random_state=42
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found")

    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # full train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # train / val split
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, stratify=y_train, random_state=random_state
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
