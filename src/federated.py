import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .models import TorchMLP
from .utils import get_model_state, set_model_state

def client_local_update(
    global_model, X_local, y_local, device,
    epochs=1, batch_size=32, lr=0.01, momentum=0.9, clip_norm=1.0
):
    model = copy.deepcopy(global_model).to(device)
    model.train()

    dataset = TensorDataset(torch.from_numpy(X_local).float(), torch.from_numpy(y_local).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            opt.step()

    return get_model_state(model)


def average_states(states, sizes, dp_sigma=None):
    total = sum(sizes)
    avg = {}
    keys = states[0].keys()
    for k in keys:
        stacked = torch.stack([s[k].float() * (sizes[i]/total) for i, s in enumerate(states)])
        avg_k = torch.sum(stacked, dim=0)
        if dp_sigma is not None and dp_sigma > 0:
            noise = torch.normal(0, dp_sigma, avg_k.shape)
            avg_k += noise.to(avg_k.device)
        avg[k] = avg_k
    return avg


def fedavg_train(
    X_train, y_train, client_parts, input_dim, device,
    global_rounds=20, clients_per_round=0.4,
    local_epochs=2, local_lr=0.01, momentum=0.9,
    dp_epsilon=float('inf'), clip_norm=1.0
):
    global_model = TorchMLP(input_dim).to(device)
    global_state = get_model_state(global_model)

    dp_sigma = clip_norm / dp_epsilon if np.isfinite(dp_epsilon) else None

    for r in range(global_rounds):
        m = max(1, int(clients_per_round * len(client_parts)))
        selected = np.random.choice(range(len(client_parts)), m, replace=False)

        local_states = []
        local_sizes = []

        for c in selected:
            idx = client_parts[c]
            if len(idx) == 0:
                continue
            state = client_local_update(
                global_model, X_train[idx], y_train[idx], device,
                epochs=local_epochs, lr=local_lr, momentum=momentum, clip_norm=clip_norm
            )
            local_states.append(state)
            local_sizes.append(len(idx))

        if not local_states:
            continue

        avg_state = average_states(local_states, local_sizes, dp_sigma=dp_sigma)
        global_state = {k: avg_state[k].to(global_state[k].dtype) for k in global_state}
        set_model_state(global_model, global_state)

    return global_model