import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

def gradient_inversion_attack_tabular(
    model, data_point, target_label, device,
    lr=0.1, iters=300
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    dp = data_point.unsqueeze(0).to(device)
    tgt = torch.tensor([target_label], device=device)

    real_out = model(dp)
    real_loss = criterion(real_out, tgt)
    real_grad = torch.autograd.grad(real_loss, model.parameters(), create_graph=False)

    dummy = torch.randn_like(dp, requires_grad=True, device=device)
    opt = optim.Adam([dummy], lr=lr)

    for _ in range(iters):
        opt.zero_grad()
        d_out = model(dummy)
        d_loss = criterion(d_out, tgt)
        d_grad = torch.autograd.grad(d_loss, model.parameters(), create_graph=True)

        grad_loss = sum((rg - dg).pow(2).sum() for rg, dg in zip(real_grad, d_grad))
        grad_loss.backward()
        opt.step()

    return dummy.detach().cpu().numpy().flatten()


def membership_inference_maxconf(
    model, X_member, y_member, X_nonmember, y_nonmember, device, sample_size=200
):
    model.eval()
    with torch.no_grad():
        m_idx = np.random.choice(len(X_member), min(sample_size//2, len(X_member)), False)
        nm_idx = np.random.choice(len(X_nonmember), min(sample_size//2, len(X_nonmember)), False)

        Xm = torch.from_numpy(X_member[m_idx]).float().to(device)
        Xnm = torch.from_numpy(X_nonmember[nm_idx]).float().to(device)

        pm = torch.softmax(model(Xm), dim=1)[:, 1].cpu().numpy()
        pnm = torch.softmax(model(Xnm), dim=1)[:, 1].cpu().numpy()

        conf_m = np.max(pm, axis=1) if pm.ndim > 1 else pm
        conf_nm = np.max(pnm, axis=1) if pnm.ndim > 1 else pnm

        labels = np.concatenate([np.ones(len(conf_m)), np.zeros(len(conf_nm))])
        scores = np.concatenate([conf_m, conf_nm])

        auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)

    return auc, fpr, tpr