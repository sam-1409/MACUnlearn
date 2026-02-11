def fedsf_retrain_excluding(
    X, y, client_parts, input_dim, device,
    forget_clients=[0], retrain_rounds=10, **fedavg_kwargs
):
    retained_parts = [p for i, p in enumerate(client_parts) if i not in forget_clients]
    if not retained_parts:
        model = TorchMLP(input_dim).to(device)
        return model

    retained_idx = np.concatenate(retained_parts)
    X_keep, y_keep = X[retained_idx], y[retained_idx]

    from torch.utils.data import DataLoader, TensorDataset
    model = TorchMLP(input_dim).to(device)
    ds = TensorDataset(torch.from_numpy(X_keep).float(), torch.from_numpy(y_keep).long())
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = torch.nn.CrossEntropyLoss()

    for _ in range(retrain_rounds):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    return model


def sfu_sharded_unlearning(
    X, y, client_parts, input_dim, device,
    forget_clients=[0], num_shards=4, retrain_epochs=5
):
    n = len(client_parts)
    shard_size = max(1, n // num_shards)
    shards = []

    for s in range(num_shards):
        start = s * shard_size
        end = min(start + shard_size, n)
        cids = list(range(start, end))
        kept = [c for c in cids if c not in forget_clients]

        if not kept:
            m = TorchMLP(input_dim).to(device)
            shards.append(m)
            continue

        idx = np.concatenate([client_parts[c] for c in kept])
        Xs, ys = X[idx], y[idx]

        m = TorchMLP(input_dim).to(device)
        ds = torch.utils.data.TensorDataset(torch.from_numpy(Xs).float(), torch.from_numpy(ys).long())
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
        opt = torch.optim.Adam(m.parameters(), lr=0.001)
        crit = torch.nn.CrossEntropyLoss()

        for _ in range(retrain_epochs):
            m.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(m(xb), yb)
                loss.backward()
                opt.step()

        shards.append(m)

    return shards