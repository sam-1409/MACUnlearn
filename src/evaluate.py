import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

@torch.no_grad()
def evaluate_torch(model, X, y, device, batch_size=128, return_probs=False):
    model.eval()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(y).long()
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    preds, probs, trues = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()   # positive class prob
        pr = torch.argmax(logits, dim=1).cpu().numpy()
        probs.extend(p)
        preds.extend(pr)
        trues.extend(yb.numpy())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, zero_division=0)
    auc = roc_auc_score(trues, probs) if len(np.unique(trues)) > 1 else np.nan

    if return_probs:
        return {"acc": acc, "f1": f1, "auc": auc}, np.array(probs)
    return {"acc": acc, "f1": f1, "auc": auc}