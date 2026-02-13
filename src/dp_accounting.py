import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from opacus.accountants import RDPAccountant
import matplotlib.pyplot as plt
import copy
import random
import warnings
warnings.filterwarnings("ignore")

# USER CONFIGURATION
DATA_CSV = "/content/training_ready_wifi_dataset.csv"
TARGET_COL = "label"

NUM_CLIENTS = 10
GLOBAL_ROUNDS = 15
CLIENTS_PER_ROUND = 0.4
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LR = 0.01
CLIP_NORM = 1.0
NOISE_MULTIPLIERS = [0.5, 0.8, 1.0, 1.2, 1.5]
DELTA = 1e-5
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# LOAD DATA
df = pd.read_csv(DATA_CSV)

if TARGET_COL not in df.columns:
    raise ValueError("Target column not found")

X_df = df.drop(columns=[TARGET_COL])
if X_df.select_dtypes(exclude=[np.number]).shape[1] > 0:
    X_df = pd.get_dummies(X_df, drop_first=True)

X = X_df.values.astype(np.float32)
y = df[TARGET_COL].values.astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

input_dim = X.shape[1]
n_classes = len(np.unique(y))

# MODEL
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# CLIENT PARTITION
indices = np.arange(len(X_train))
np.random.shuffle(indices)
client_parts = np.array_split(indices, NUM_CLIENTS)

# FEDERATED TRAINING WITH DP
def federated_training(noise_multiplier):

    global_model = MLP(input_dim).to(device)
    accountant = RDPAccountant()

    sample_rate = CLIENTS_PER_ROUND

    for round in range(GLOBAL_ROUNDS):

        selected_clients = np.random.choice(
            NUM_CLIENTS,
            max(1, int(NUM_CLIENTS * CLIENTS_PER_ROUND)),
            replace=False
        )

        local_states = []
        local_sizes = []

        for cid in selected_clients:
            idx = client_parts[cid]
            X_local = X_train[idx]
            y_local = y_train[idx]

            dataset = TensorDataset(
                torch.tensor(X_local),
                torch.tensor(y_local)
            )

            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            local_model = copy.deepcopy(global_model).to(device)
            optimizer = optim.SGD(local_model.parameters(), lr=LR, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            local_model.train()

            for epoch in range(LOCAL_EPOCHS):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)

                    optimizer.zero_grad()
                    loss = criterion(local_model(xb), yb)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), CLIP_NORM)

                    # Add Gaussian noise manually
                    for param in local_model.parameters():
                        if param.grad is not None:
                            noise = torch.normal(
                                0,
                                noise_multiplier * CLIP_NORM,
                                size=param.grad.shape
                            ).to(device)
                            param.grad += noise

                    optimizer.step()

            local_states.append(copy.deepcopy(local_model.state_dict()))
            local_sizes.append(len(idx))

        # Aggregate
        new_state = {}
        total_size = sum(local_sizes)

        for key in global_model.state_dict().keys():
            new_state[key] = sum(
                local_states[i][key] * (local_sizes[i] / total_size)
                for i in range(len(local_states))
            )

        global_model.load_state_dict(new_state)

        accountant.step(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate
        )

    epsilon = accountant.get_epsilon(delta=DELTA)

    # Evaluate
    global_model.eval()
    with torch.no_grad():
        logits = global_model(torch.tensor(X_test).to(device)).cpu()
    preds = torch.argmax(logits, dim=1).numpy()
    acc = accuracy_score(y_test, preds)

    return acc, epsilon

# RUN EXPERIMENTS
accuracies = []
epsilons = []

print("\nRunning DP-FL experiments...\n")

for sigma in NOISE_MULTIPLIERS:
    acc, eps = federated_training(noise_multiplier=sigma)
    accuracies.append(acc)
    epsilons.append(eps)

print("\nFinal Results Summary:")
for i in range(len(NOISE_MULTIPLIERS)):
    print(f"σ={NOISE_MULTIPLIERS[i]}  |  ε={epsilons[i]:.3f}  |  Acc={accuracies[i]:.4f}")