#!/usr/bin/env python3
"""
adacurve_pipeline_ann.py  — modified for SIFT-1M with score-based sorting

Changes:
- Use real SIFT-1M base vectors for training
- Removed all ANN evaluation / query sampling logic
- After training, compute score for ALL base vectors
- Sort by descending score
- Save to result/f5_adacurve.csv : id, score, v0, v1, ..., v127
"""

import os
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# Dataset reading functions
# -------------------------
DATASET_ROOT = "../dataset"
DATASET_NAME = "sift-1m"

def get_dataset_dir(dataset_root_dir, dataset):
    dataset_dir = dataset_root_dir
    for part in dataset.split('-'):
        dataset_dir = os.path.join(dataset_dir, part)
    return dataset_dir

def read_vecs(fname):
    data = np.fromfile(fname, dtype='int32')
    dim = data[0]
    vectors = data.reshape(-1, dim + 1)[:, 1:].copy()
    if fname.endswith(".fvecs"):
        vectors = vectors.view('float32')
    return vectors

def read_ivecs(fname):
    data = np.fromfile(fname, dtype='int32')
    dim = data[0]
    return data.reshape(-1, dim + 1)[:, 1:].copy()

def save_vectors_to_csv(vectors, ids, scores, filename, out_dir="result"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    
    data = np.hstack([
        ids.reshape(-1, 1),
        scores.reshape(-1, 1),
        vectors
    ])
    columns = ["id", "score"] + [f"v{i}" for i in range(vectors.shape[1])]
    
    pd.DataFrame(data, columns=columns).to_csv(path, index=False)
    print(f"Saved sorted vectors with scores: {path}  ({len(ids)} rows)")

# -------------------------
# Model
# -------------------------
class CompressionRankingModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dims=(512,256), bottleneck_dim=32):
        super().__init__()
        enc = []
        d = input_dim
        for h in hidden_dims:
            enc.append(nn.Linear(d, h))
            enc.append(nn.ReLU(inplace=True))
            d = h
        enc.append(nn.Linear(d, bottleneck_dim))
        self.encoder = nn.Sequential(*enc)

        dec = []
        d = bottleneck_dim
        for h in reversed(hidden_dims):
            dec.append(nn.Linear(d, h))
            dec.append(nn.ReLU(inplace=True))
            d = h
        dec.append(nn.Linear(d, input_dim))
        self.decoder = nn.Sequential(*dec)

        # score head
        hid = max(8, bottleneck_dim // 2)
        self.score = nn.Sequential(
            nn.Linear(bottleneck_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, 1)
        )

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        xrec = self.decoder(z)
        s = self.score(z).squeeze(-1)
        return xrec, s, z

# -------------------------
# Losses (kept the parts used in training)
# -------------------------
def multi_positive_nce_loss(scores: torch.Tensor, positives_mask: torch.Tensor, temperature: float = 0.07):
    device = scores.device
    pos_mask = positives_mask.bool().squeeze(-1)
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    scaled = scores / (temperature + 1e-12)
    max_all = torch.max(scaled)
    denom = torch.log(torch.sum(torch.exp(scaled - max_all))) + max_all

    scaled_pos = scaled[pos_mask]
    max_pos = torch.max(scaled_pos)
    numer = torch.log(torch.sum(torch.exp(scaled_pos - max_pos))) + max_pos

    loss = -(numer - denom)
    return loss

def soft_ranks_from_scores(scores: torch.Tensor, tau: float = 0.2):
    s_i = scores.unsqueeze(1)
    s_j = scores.unsqueeze(0)
    P = torch.sigmoid((s_j - s_i) / (tau + 1e-12))
    r = 1.0 + P.sum(dim=1)
    return r

def L_global_pairwise(scores: torch.Tensor, positives_mask: torch.Tensor, tau: float = 0.2, eps: float = 1e-6, max_pairs: int = 10000):
    device = scores.device
    pos_idx = positives_mask.bool().squeeze(-1)
    k = int(pos_idx.sum().item())
    if k < 2:
        return torch.tensor(0.0, device=device)

    r = soft_ranks_from_scores(scores, tau=tau)
    pos_r = r[pos_idx]

    if k * (k - 1) // 2 > max_pairs:
        sel_k = int(max(2, (2 * max_pairs) ** 0.5))
        perm = torch.randperm(k, device=device)[:sel_k]
        pos_r = pos_r[perm]
        k = sel_k

    ri = pos_r.unsqueeze(1)
    rj = pos_r.unsqueeze(0)
    diffs = ri - rj
    abs_diffs = torch.sqrt(diffs * diffs + eps)
    sum_pairwise = 0.5 * abs_diffs.sum()
    norm = float(k * (k - 1) / 2)
    return sum_pairwise / (norm + 1e-12)

# -------------------------
# Seed & Training pipeline
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_pipeline(
    seed: int = 42,
    device_str: str = None,
    batch_size: int = 1024,
    epochs: int = 40,
    initial_lr: float = 1e-3,
    final_lr: float = 5e-5,
    weight_decay: float = 0.0,
    alpha_lae: float = 0.5,
    alpha_nce: float = 1.0,
    alpha_pair: float = 1.0,
    tau_sim: float = 0.07,
    tau_rank: float = 0.15,
    max_pairs: int = 10000,
    queries_per_batch: int = 8,
    k_choices: list = [1,5,10],
    global_sample_size: int = 3000,
    clip_grad_norm: float = 5.0,
    out_dir: str = "./out_gpu",
    switch_epoch: int = None
):
    set_seed(seed)
    device = torch.device(device_str if device_str is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(out_dir, exist_ok=True)

    if switch_epoch is None:
        switch_epoch = max(1, epochs // 2)

    # ── Read SIFT-1M ─────────────────────────────────────────────
    dataset_dir = get_dataset_dir(DATASET_ROOT, DATASET_NAME)
    sift_dir = os.path.join(dataset_dir, "sift")

    print(f"Reading SIFT-1M from: {sift_dir}")

    vectors_base = read_vecs(os.path.join(sift_dir, "sift_base.fvecs"))
    # vectors_query = read_vecs(os.path.join(sift_dir, "sift_query.fvecs"))   # not used in training
    # vectors_gt   = read_ivecs(os.path.join(sift_dir, "sift_groundtruth.ivecs"))  # not used

    X = vectors_base.astype(np.float32)
    base_ids = np.arange(len(X), dtype=np.int64)

    print(f"Loaded base vectors shape: {X.shape}")

    # split (80%/10%/10%)
    n_rows = X.shape[0]
    idx = np.arange(n_rows); np.random.shuffle(idx)
    ntr = int(0.8 * n_rows); nval = int(0.1 * n_rows)
    train_idx = idx[:ntr]; val_idx = idx[ntr:ntr+nval]; test_idx = idx[ntr+nval:]
    X_train = X[train_idx]; X_val = X[val_idx]; X_test = X[test_idx]
    print("Split sizes (train/val/test):", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train).float())
    val_ds   = TensorDataset(torch.from_numpy(X_val).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=(device.type=='cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=(device.type=='cuda'))

    # Model & Optimizer
    dim = X.shape[1]
    model = CompressionRankingModel(input_dim=dim, hidden_dims=(512,256), bottleneck_dim=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < switch_epoch:
            return 1.0
        return float(final_lr / initial_lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    history = {"epoch": [], "train_loss": [], "LAE": [], "L_ann": [], "lr": []}

    best_loss = float('inf')
    best_checkpoint_path = os.path.join(out_dir, "model_best.pth")

    # ── Training Loop ────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        global_sample_size_eff = min(global_sample_size, X_train.shape[0])
        global_idx = np.random.choice(X_train.shape[0], size=global_sample_size_eff, replace=False)
        X_global_sample = X_train[global_idx]

        model.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(X_global_sample).float().to(device)
            _, _, z_global_ex = model(Xt)
        model.train()

        running_loss = running_LAE = running_Lann = 0.0

        for batch in train_loader:
            bx = batch[0].to(device)
            opt.zero_grad()

            xrec, _, z_batch = model(bx)
            LAE = torch.mean((bx - xrec) ** 2)

            Lann_accum = torch.tensor(0.0, device=device)
            valid_qcnt = 0

            for _ in range(queries_per_batch):
                qv_np, kq, knn_idx = sample_ann_query_from_X(X_train, k_choices=k_choices)
                qv = torch.from_numpy(qv_np).float().to(device).unsqueeze(0)

                with torch.no_grad():
                    zq = model.encode(qv)

                zg = z_global_ex
                zg_norm = zg / (zg.norm(dim=1, keepdim=True) + 1e-9)
                zq_norm = zq / (zq.norm(dim=1, keepdim=True) + 1e-9)
                scores_global = torch.matmul(zg_norm, zq_norm.t()).squeeze(-1)

                mask_global_np = np.zeros(X_global_sample.shape[0], dtype=bool)
                inter_mask = np.isin(global_idx, knn_idx)
                if inter_mask.sum() > 0:
                    pos_positions = np.where(inter_mask)[0]
                    mask_global_np[pos_positions] = True
                else:
                    knn_in_g = get_knn_indices(X_global_sample, qv_np, k=kq)
                    mask_global_np[knn_in_g] = True

                mask_global = torch.from_numpy(mask_global_np).to(device)

                nce_loss = multi_positive_nce_loss(scores_global, mask_global, temperature=tau_sim)
                pair_disp = L_global_pairwise(scores_global, mask_global.unsqueeze(-1), tau=tau_rank, max_pairs=max_pairs)

                per_q_ann = alpha_nce * nce_loss + alpha_pair * pair_disp
                Lann_accum += per_q_ann
                valid_qcnt += 1

            Lann = Lann_accum / float(valid_qcnt) if valid_qcnt > 0 else torch.tensor(0.0, device=device)
            Loverall = alpha_lae * LAE + Lann

            Loverall.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            opt.step()

            running_loss += float(Loverall.item())
            running_LAE  += float(LAE.item())
            running_Lann += float(Lann.item())

        scheduler.step()
        current_lr = opt.param_groups[0]['lr']

        avg_loss = running_loss / len(train_loader)
        avg_LAE  = running_LAE  / len(train_loader)
        avg_Lann = running_Lann / len(train_loader)

        # val LAE
        model.eval()
        with torch.no_grad():
            val_mse_vals = []
            for vb in val_loader:
                bxv = vb[0].to(device)
                xrec_v, _, _ = model(bxv)
                val_mse_vals.append(float(torch.mean((bxv - xrec_v)**2).item()))
            val_LAE = float(np.mean(val_mse_vals)) if val_mse_vals else 0.0

        epoch_time = time.time() - epoch_start

        print(f"[Epoch {epoch}] lr={current_lr:.6g} loss={avg_loss:.6f} LAE(train)={avg_LAE:.8f} Lann={avg_Lann:.8f} LAE(val)={val_LAE:.8f} {epoch_time:.1f}s")

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["LAE"].append(avg_LAE)
        history["L_ann"].append(avg_Lann)
        history["lr"].append(current_lr)

        # save best by validation LAE
        if val_LAE < best_loss:
            best_loss = val_LAE
            save_dict = {
                "state_dict": model.state_dict(),
                "input_dim": dim,
                "hidden_dims": [512,256],
                "bottleneck_dim": 32,
                "epoch": epoch,
                "seed": seed,
            }
            torch.save(save_dict, best_checkpoint_path)
            print(f"→ new best (val_LAE={best_loss:.8f}) saved to {best_checkpoint_path}")

    # ── After training: score ALL base vectors and sort ────────────────
    print("\nComputing scores for all base vectors ...")
    model.eval()
    all_scores = []
    score_batch_size = 65536  # adjust according to GPU memory

    with torch.no_grad():
        for i in range(0, len(X), score_batch_size):
            batch = torch.from_numpy(X[i:i+score_batch_size]).float().to(device)
            _, batch_scores, _ = model(batch)
            all_scores.append(batch_scores.cpu().numpy())

    all_scores = np.concatenate(all_scores)

    # Sort by descending score
    sort_idx = np.argsort(-all_scores)
    sorted_ids    = base_ids[sort_idx]
    sorted_scores = all_scores[sort_idx]
    sorted_vectors = X[sort_idx]

    # Save
    save_vectors_to_csv(
        vectors=sorted_vectors,
        ids=sorted_ids,
        scores=sorted_scores,
        filename="f5_adacurve.csv",
        out_dir="result"
    )

    # final model
    torch.save({"state_dict": model.state_dict(), "input_dim": dim},
               os.path.join(out_dir, "model_final.pth"))

    print("Training finished.")
    return history


def sample_ann_query_from_X(X: np.ndarray, k_choices: list = [1,5,10]):
    qi = np.random.randint(0, X.shape[0])
    qv = X[qi].astype(np.float32)
    k = int(random.choice(k_choices))
    dif = X - qv.reshape(1, -1)
    d2 = np.sum(dif * dif, axis=1)
    idx = np.argsort(d2)[:k]
    return qv, k, idx

def get_knn_indices(X: np.ndarray, query_vec: np.ndarray, k: int = 10):
    dif = X - query_vec.reshape(1, -1)
    d2 = np.sum(dif * dif, axis=1)
    idx = np.argsort(d2)[:k]
    return idx


def main():
    parser = argparse.ArgumentParser(description="Train adacurve model on SIFT-1M base")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out_dir", type=str, default="./out_gpu")
    parser.add_argument("--switch_epoch", type=int, default=None)
    args = parser.parse_args()

    train_pipeline(
        epochs=args.epochs,
        out_dir=args.out_dir,
        device_str=args.device,
        switch_epoch=(args.switch_epoch - 1 if args.switch_epoch else None)
    )


if __name__ == "__main__":
    main()