#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-BODY (planar) + HNN + Manifold Retraction Mixup (ENERGY-only)  [SAFE/MASKED]
=============================================================================

This is a *patched* version of the earlier 2-body script to prevent training
blow-ups caused by near-collision states produced by Mixup.

Key patch:
- Augmented samples are masked out of loss if ||r|| < r_min (near-collision),
  because dp/dt = -k r / ||r||^3 can explode.
- Lower default lr, slightly stronger stability.
- Optional: you can also clamp acceleration if you want (not enabled by default).

Methods:
  1) NoAug
  2) RetractionMixup (oracle, energy-only, anchor rule)
  3) EnergyMatched+Retraction (oracle, energy-only, anchor rule)

Evaluations:
  - IID test
  - OOD-energy
  - OOD-orbit (high eccentricity)

Metrics:
  - testMSE
  - rolloutEndErr (final state error)
  - driftE (energy drift)
  - driftL (angular momentum drift)

Outputs:
  ./out_2body_masked/results.csv
"""

from __future__ import annotations

import os
import csv
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Two-body physics (relative coords)
# x = (r_x, r_y, p_x, p_y)
# H(r,p)=||p||^2/(2mu) - k/||r||
# f_true(x) = (p/mu, -k r/||r||^3)
# ----------------------------
def H_true_np(x: np.ndarray, mu: float, k: float, eps: float = 1e-9) -> np.ndarray:
    r = x[..., 0:2]
    p = x[..., 2:4]
    rnorm = np.linalg.norm(r, axis=-1) + eps
    T = np.sum(p * p, axis=-1) / (2.0 * mu)
    V = -k / rnorm
    return T + V


def H_true_torch(x: torch.Tensor, mu: float, k: float, eps: float = 1e-9) -> torch.Tensor:
    r = x[..., 0:2]
    p = x[..., 2:4]
    rnorm = torch.linalg.norm(r, dim=-1) + eps
    T = torch.sum(p * p, dim=-1) / (2.0 * mu)
    V = -k / rnorm
    return T + V


def L_np(x: np.ndarray) -> np.ndarray:
    r = x[..., 0:2]
    p = x[..., 2:4]
    return r[..., 0] * p[..., 1] - r[..., 1] * p[..., 0]


def L_torch(x: torch.Tensor) -> torch.Tensor:
    r = x[..., 0:2]
    p = x[..., 2:4]
    return r[..., 0] * p[..., 1] - r[..., 1] * p[..., 0]


def f_true_np(x: np.ndarray, mu: float, k: float, eps: float = 1e-9) -> np.ndarray:
    r = x[..., 0:2]
    p = x[..., 2:4]
    rnorm = np.linalg.norm(r, axis=-1, keepdims=True) + eps
    dr = p / mu
    dp = -k * r / (rnorm ** 3)
    return np.concatenate([dr, dp], axis=-1)


def f_true_torch(x: torch.Tensor, mu: float, k: float, eps: float = 1e-9) -> torch.Tensor:
    r = x[..., 0:2]
    p = x[..., 2:4]
    rnorm = torch.linalg.norm(r, dim=-1, keepdims=True) + eps
    dr = p / mu
    dp = -k * r / (rnorm ** 3)
    return torch.cat([dr, dp], dim=-1)


# ----------------------------
# Symplectic integrator: velocity Verlet (kick-drift-kick)
# ----------------------------
def verlet_step(r: np.ndarray, p: np.ndarray, dt: float, mu: float, k: float, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    rnorm = (np.linalg.norm(r) + eps)
    a = -k * r / (rnorm ** 3)

    p_half = p + 0.5 * dt * a
    r_next = r + dt * (p_half / mu)

    rnorm2 = (np.linalg.norm(r_next) + eps)
    a_next = -k * r_next / (rnorm2 ** 3)

    p_next = p_half + 0.5 * dt * a_next
    return r_next, p_next


def integrate_verlet(x0: np.ndarray, dt: float, steps: int, mu: float, k: float) -> np.ndarray:
    traj = np.zeros((steps + 1, 4), dtype=np.float64)
    r = x0[0:2].astype(np.float64).copy()
    p = x0[2:4].astype(np.float64).copy()
    traj[0] = x0.astype(np.float64)
    for t in range(steps):
        r, p = verlet_step(r, p, dt=dt, mu=mu, k=k)
        traj[t + 1, 0:2] = r
        traj[t + 1, 2:4] = p
    return traj


# ----------------------------
# Sampling bound orbits by (E,e) at turning points (p_r=0)
# ----------------------------
def rotate2(v: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float64)


def sample_ic_bound_orbit(rng: np.random.Generator, E: float, e: float, mu: float, k: float) -> np.ndarray:
    assert E < 0.0, "Bound orbit requires E < 0"
    e = float(np.clip(e, 0.0, 0.999999))
    a = -k / (2.0 * E)  # > 0
    Lmag = mu * math.sqrt(max(k * a * (1.0 - e * e), 1e-12))

    sign = 1.0 if rng.random() < 0.5 else -1.0
    L = sign * Lmag

    # turning points: E r^2 + k r - L^2/(2mu) = 0
    D = k * k + (2.0 * E * (L * L) / mu)
    D = max(D, 1e-12)
    sqrtD = math.sqrt(D)

    r1 = (-k + sqrtD) / (2.0 * E)
    r2 = (-k - sqrtD) / (2.0 * E)
    r_p = min(r1, r2)
    r_a = max(r1, r2)

    rmag = r_p if rng.random() < 0.5 else r_a
    rmag = float(max(rmag, 5e-2))  # minimal radius guard (data-level)

    p_t = L / rmag
    r_vec = np.array([rmag, 0.0], dtype=np.float64)
    t_hat = np.array([0.0, 1.0], dtype=np.float64)
    p_vec = p_t * t_hat

    ang = float(rng.uniform(0.0, 2.0 * math.pi))
    r_rot = rotate2(r_vec, ang)
    p_rot = rotate2(p_vec, ang)
    return np.concatenate([r_rot, p_rot], axis=0).astype(np.float64)


def make_dataset(
    *,
    n_traj: int,
    steps_per_traj: int,
    dt: float,
    mu: float,
    k: float,
    E_range: Tuple[float, float],
    e_range: Tuple[float, float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_list: List[np.ndarray] = []

    E_lo, E_hi = E_range
    e_lo, e_hi = e_range

    for _ in range(n_traj):
        E0 = float(rng.uniform(E_lo, E_hi))
        e0 = float(rng.uniform(e_lo, e_hi))
        x0 = sample_ic_bound_orbit(rng, E=E0, e=e0, mu=mu, k=k)
        traj = integrate_verlet(x0, dt=dt, steps=steps_per_traj, mu=mu, k=k)
        X_list.append(traj[:-1])

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    dX = f_true_np(X, mu=mu, k=k).astype(np.float32)
    E = H_true_np(X, mu=mu, k=k).astype(np.float32)
    L = L_np(X).astype(np.float32)
    return X, dX, E, L


def split_dataset(X: np.ndarray, dX: np.ndarray, E: np.ndarray, L: np.ndarray, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    split = int((1.0 - test_ratio) * N)
    tr, te = idx[:split], idx[split:]
    return (X[tr], dX[tr], E[tr], L[tr]), (X[te], dX[te], E[te], L[te])


# ----------------------------
# HNN model in R^4: f = J grad H
# x = (r(2), p(2)); f = (dH/dp, -dH/dr)
# ----------------------------
class MLPHamiltonian(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(depth):
            layers += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def hnn_vector_field(model: nn.Module, x: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
    x = x.requires_grad_(True)
    H = model(x)
    gradH = torch.autograd.grad(H.sum(), x, create_graph=create_graph, retain_graph=create_graph)[0]
    dH_dr = gradH[:, 0:2]
    dH_dp = gradH[:, 2:4]
    dr = dH_dp
    dp = -dH_dr
    return torch.cat([dr, dp], dim=-1)


# ----------------------------
# Rollout using RK4 (learned field)
# ----------------------------
def f_theta_inference(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x_ = x.detach().requires_grad_(True)
        f = hnn_vector_field(model, x_.unsqueeze(0), create_graph=False).squeeze(0)
    return f.detach()


def rollout_rk4(model: nn.Module, x0: torch.Tensor, dt: float, steps: int) -> torch.Tensor:
    xs = [x0.detach().clone()]
    x = x0.detach().clone()
    for _ in range(steps):
        k1 = f_theta_inference(model, x)
        k2 = f_theta_inference(model, x + 0.5 * dt * k1)
        k3 = f_theta_inference(model, x + 0.5 * dt * k2)
        k4 = f_theta_inference(model, x + dt * k3)
        x = (x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).detach()
        xs.append(x.clone())
    return torch.stack(xs, dim=0)


# ----------------------------
# Augmentations (oracle-labeled) + SAFETY MASK
# ----------------------------
def beta_sample(alpha: float, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.distributions.Beta(alpha, alpha).sample(shape).to(device)


def energy_matched_perm(E: torch.Tensor, delta_E: float, max_tries: int = 60) -> torch.Tensor:
    B = E.shape[0]
    perm = torch.empty(B, dtype=torch.long, device=E.device)
    for i in range(B):
        ei = E[i]
        j = None
        for _ in range(max_tries):
            cand = int(torch.randint(0, B, (1,), device=E.device).item())
            if abs((E[cand] - ei).item()) <= delta_E:
                j = cand
                break
        if j is None:
            j = int(torch.randint(0, B, (1,), device=E.device).item())
        perm[i] = j
    return perm


def retraction_mixup_energy_only_oracle_masked(
    x: torch.Tensor,
    *,
    alpha: float,
    mu: float,
    k: float,
    energy_rule: str = "anchor_i",
    perm: Optional[torch.Tensor] = None,
    r_min: float = 0.35,     # <<< 핵심: near-collision mask threshold
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      x_star: (B,4)
      dx_star: (B,4)
      mask: (B,) bool, True if safe (||r|| >= r_min)
    """
    B = x.shape[0]
    if perm is None:
        perm = torch.randperm(B, device=x.device)

    lam = beta_sample(alpha, (B,), x.device)

    r_i, p_i = x[:, 0:2], x[:, 2:4]
    r_j, p_j = x[perm, 0:2], x[perm, 2:4]

    r_t = (1.0 - lam).unsqueeze(-1) * r_i + lam.unsqueeze(-1) * r_j
    p_t = (1.0 - lam).unsqueeze(-1) * p_i + lam.unsqueeze(-1) * p_j

    E_i = H_true_torch(x, mu=mu, k=k, eps=eps)
    E_j = H_true_torch(x[perm], mu=mu, k=k, eps=eps)

    if energy_rule == "anchor_i":
        Ebar = E_i
    elif energy_rule == "interpolated":
        Ebar = (1.0 - lam) * E_i + lam * E_j
    else:
        raise ValueError("energy_rule must be 'anchor_i' or 'interpolated'")

    rnorm = torch.linalg.norm(r_t, dim=-1) + eps
    T_target = torch.clamp(Ebar + (k / rnorm), min=0.0)
    p_target_mag = torch.sqrt(2.0 * mu * T_target + eps)

    p_mag = torch.linalg.norm(p_t, dim=-1) + eps
    scale = (p_target_mag / p_mag).unsqueeze(-1)
    p_star = scale * p_t

    x_star = torch.cat([r_t, p_star], dim=-1)

    # SAFETY MASK: exclude near-collision augmented points from loss_aug
    rnorm_star = torch.linalg.norm(x_star[:, 0:2], dim=-1)
    mask = (rnorm_star >= r_min)

    dx_star = f_true_torch(x_star, mu=mu, k=k, eps=eps)
    return x_star, dx_star, mask


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def eval_vector_field_mse(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, dx, _E in loader:
        x = x.to(device)
        dx = dx.to(device)
        with torch.enable_grad():
            pred = hnn_vector_field(model, x, create_graph=False)
            mse = torch.mean((pred - dx) ** 2)
        total += float(mse.item()) * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


def eval_rollout_end_metrics(
    model: nn.Module,
    init_pool: np.ndarray,
    dt: float,
    steps: int,
    k_eval: int,
    seed: int,
    mu: float,
    k_evalgrav: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(init_pool), size=min(k_eval, len(init_pool)), replace=False)

    errs, drE, drL = [], [], []

    for i in idx:
        x0_np = init_pool[i].astype(np.float32)
        true_traj = integrate_verlet(x0_np, dt=dt, steps=steps, mu=mu, k=k_evalgrav).astype(np.float32)

        x0 = torch.from_numpy(x0_np).to(device)
        pred_traj = rollout_rk4(model, x0, dt=dt, steps=steps)

        xT_true = torch.from_numpy(true_traj[-1]).to(device)
        xT_pred = pred_traj[-1]

        errs.append(float(torch.linalg.norm(xT_pred - xT_true).item()))

        H0 = float(H_true_torch(x0.unsqueeze(0), mu=mu, k=k_evalgrav).item())
        HT = float(H_true_torch(xT_pred.unsqueeze(0), mu=mu, k=k_evalgrav).item())
        drE.append(float(abs(HT - H0)))

        L0 = float(L_torch(x0.unsqueeze(0)).item())
        LT = float(L_torch(xT_pred.unsqueeze(0)).item())
        drL.append(float(abs(LT - L0)))

    return float(np.mean(errs)), float(np.mean(drE)), float(np.mean(drL))


# ----------------------------
# Dataloaders
# ----------------------------
def make_loader(X: np.ndarray, dX: np.ndarray, E: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(dX), torch.from_numpy(E))
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, generator=g)


# ----------------------------
# Training
# ----------------------------
def train_one_model(
    *,
    condition: str,
    aug_weight: float,
    cfg,
    seed: int,
    train_loader,
) -> nn.Module:
    set_seed(seed)
    device = torch.device(cfg.device)

    model = MLPHamiltonian(in_dim=4, hidden=cfg.hidden, depth=cfg.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for _ in tqdm(range(cfg.epochs), desc=f"train[{condition}|w={aug_weight}|seed={seed}]", leave=False):
        model.train()
        for x, dx, E in train_loader:
            x = x.to(device)
            dx = dx.to(device)
            E = E.to(device)

            opt.zero_grad(set_to_none=True)

            pred = hnn_vector_field(model, x, create_graph=True)
            loss_main = torch.mean((pred - dx) ** 2)

            if condition == "NoAug" or aug_weight <= 0.0:
                loss = loss_main
            else:
                if condition == "RetractionMixup":
                    x_aug, dx_aug, mask = retraction_mixup_energy_only_oracle_masked(
                        x,
                        alpha=cfg.mixup_alpha,
                        mu=cfg.mu,
                        k=cfg.kgrav,
                        energy_rule=cfg.energy_rule,
                        r_min=cfg.r_min_aug,
                    )
                elif condition == "EnergyMatched+Retraction":
                    perm = energy_matched_perm(E, delta_E=cfg.delta_E)
                    x_aug, dx_aug, mask = retraction_mixup_energy_only_oracle_masked(
                        x,
                        alpha=cfg.mixup_alpha,
                        mu=cfg.mu,
                        k=cfg.kgrav,
                        energy_rule=cfg.energy_rule,
                        perm=perm,
                        r_min=cfg.r_min_aug,
                    )
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                pred_aug = hnn_vector_field(model, x_aug, create_graph=True)

                if mask.any():
                    loss_aug = torch.mean((pred_aug[mask] - dx_aug[mask]) ** 2)
                else:
                    # keep graph; "0 * loss_main" yields a tensor requiring grad
                    loss_aug = 0.0 * loss_main

                loss = loss_main + aug_weight * loss_aug

            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

    return model


# ----------------------------
# CSV logging
# ----------------------------
def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class Config:
    out_dir: str = "./out_2body_masked"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # physics
    mu: float = 1.0
    kgrav: float = 1.0

    # data
    data_seed: int = 123
    n_traj_train: int = 120
    steps_per_traj: int = 180
    dt: float = 0.01
    test_ratio: float = 0.2

    # train distribution
    E_train: Tuple[float, float] = (-1.6, -0.9)
    e_train: Tuple[float, float] = (0.0, 0.6)

    # OOD-energy
    E_ood_energy: Tuple[float, float] = (-0.9, -0.4)
    e_ood_energy: Tuple[float, float] = (0.0, 0.6)

    # OOD-orbit: high eccentricity
    E_ood_orbit: Tuple[float, float] = (-1.6, -0.9)
    e_ood_orbit: Tuple[float, float] = (0.8, 0.95)

    # model
    hidden: int = 128
    depth: int = 3

    # training (more stable defaults)
    epochs: int = 60
    batch_size: int = 512
    lr: float = 5e-4          # <<< lowered from 2e-3
    grad_clip: float = 0.5    # <<< tighter clip

    # augmentation
    mixup_alpha: float = 0.4
    energy_rule: str = "anchor_i"
    aug_weights: Tuple[float, ...] = (0.3, 0.7)
    delta_E: float = 0.10
    r_min_aug: float = 0.35   # <<< mask threshold for augmented samples

    # evaluation
    rollout_steps: int = 600
    eval_k: int = 10
    eval_seed: int = 999

    # runs
    seeds: Tuple[int, ...] = (0, 1, 2)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, "results.csv")

    print("== Build TRAIN/IID dataset ==")
    X_iid, dX_iid, E_iid, L_iid = make_dataset(
        n_traj=cfg.n_traj_train,
        steps_per_traj=cfg.steps_per_traj,
        dt=cfg.dt,
        mu=cfg.mu,
        k=cfg.kgrav,
        E_range=cfg.E_train,
        e_range=cfg.e_train,
        seed=cfg.data_seed,
    )
    (Xtr, dXtr, Etr, Ltr), (Xte, dXte, Ete, Lte) = split_dataset(
        X_iid, dX_iid, E_iid, L_iid, test_ratio=cfg.test_ratio, seed=cfg.data_seed
    )

    print("== Build OOD-energy eval pool ==")
    X_oodE, dX_oodE, E_oodE, L_oodE = make_dataset(
        n_traj=max(80, cfg.n_traj_train // 2),
        steps_per_traj=cfg.steps_per_traj,
        dt=cfg.dt,
        mu=cfg.mu,
        k=cfg.kgrav,
        E_range=cfg.E_ood_energy,
        e_range=cfg.e_ood_energy,
        seed=cfg.data_seed + 777,
    )

    print("== Build OOD-orbit eval pool (high eccentricity) ==")
    X_oodO, dX_oodO, E_oodO, L_oodO = make_dataset(
        n_traj=max(80, cfg.n_traj_train // 2),
        steps_per_traj=cfg.steps_per_traj,
        dt=cfg.dt,
        mu=cfg.mu,
        k=cfg.kgrav,
        E_range=cfg.E_ood_orbit,
        e_range=cfg.e_ood_orbit,
        seed=cfg.data_seed + 1337,
    )

    results: List[Dict[str, object]] = []

    plan = []
    for seed in cfg.seeds:
        plan.append(("NoAug", 0.0, seed))
        for w in cfg.aug_weights:
            plan.append(("RetractionMixup", float(w), seed))
            plan.append(("EnergyMatched+Retraction", float(w), seed))

    print(f"== Total runs: {len(plan)} ==")
    rid = 0

    for condition, w, seed in plan:
        rid += 1
        print(f"\n[{rid}/{len(plan)}] condition={condition}  aug_weight={w}  seed={seed}")

        train_loader = make_loader(Xtr, dXtr, Etr, batch_size=cfg.batch_size, shuffle=True, seed=seed)
        iid_test_loader = make_loader(Xte, dXte, Ete, batch_size=cfg.batch_size, shuffle=False, seed=seed)
        oodE_loader = make_loader(X_oodE, dX_oodE, E_oodE, batch_size=cfg.batch_size, shuffle=False, seed=seed)
        oodO_loader = make_loader(X_oodO, dX_oodO, E_oodO, batch_size=cfg.batch_size, shuffle=False, seed=seed)

        model = train_one_model(condition=condition, aug_weight=w, cfg=cfg, seed=seed, train_loader=train_loader)

        device = torch.device(cfg.device)

        testMSE_iid = eval_vector_field_mse(model, iid_test_loader, device)
        testMSE_oodE = eval_vector_field_mse(model, oodE_loader, device)
        testMSE_oodO = eval_vector_field_mse(model, oodO_loader, device)

        roll_iid, driftE_iid, driftL_iid = eval_rollout_end_metrics(
            model, init_pool=Xte, dt=cfg.dt, steps=cfg.rollout_steps, k_eval=cfg.eval_k,
            seed=cfg.eval_seed + seed, mu=cfg.mu, k_evalgrav=cfg.kgrav, device=device
        )
        roll_oodE, driftE_oodE, driftL_oodE = eval_rollout_end_metrics(
            model, init_pool=X_oodE, dt=cfg.dt, steps=cfg.rollout_steps, k_eval=cfg.eval_k,
            seed=cfg.eval_seed + 1000 + seed, mu=cfg.mu, k_evalgrav=cfg.kgrav, device=device
        )
        roll_oodO, driftE_oodO, driftL_oodO = eval_rollout_end_metrics(
            model, init_pool=X_oodO, dt=cfg.dt, steps=cfg.rollout_steps, k_eval=cfg.eval_k,
            seed=cfg.eval_seed + 2000 + seed, mu=cfg.mu, k_evalgrav=cfg.kgrav, device=device
        )

        row = {
            "seed": int(seed),
            "condition": condition,
            "aug_weight": float(w),
            "energy_rule": cfg.energy_rule,
            "delta_E": float(cfg.delta_E),
            "r_min_aug": float(cfg.r_min_aug),

            "testMSE_iid": float(testMSE_iid),
            "rolloutEndErr_iid": float(roll_iid),
            "driftE_iid": float(driftE_iid),
            "driftL_iid": float(driftL_iid),

            "testMSE_oodEnergy": float(testMSE_oodE),
            "rolloutEndErr_oodEnergy": float(roll_oodE),
            "driftE_oodEnergy": float(driftE_oodE),
            "driftL_oodEnergy": float(driftL_oodE),

            "testMSE_oodOrbit": float(testMSE_oodO),
            "rolloutEndErr_oodOrbit": float(roll_oodO),
            "driftE_oodOrbit": float(driftE_oodO),
            "driftL_oodOrbit": float(driftL_oodO),

            "dt": cfg.dt,
            "rollout_steps": cfg.rollout_steps,
            "train_steps_per_traj": cfg.steps_per_traj,
            "n_traj_train": cfg.n_traj_train,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
        }
        results.append(row)
        write_csv(out_path, results)

        print(f"  IID:  testMSE={testMSE_iid:.3e}  rollEnd={roll_iid:.3e}  driftE={driftE_iid:.3e}  driftL={driftL_iid:.3e}")
        print(f"  OOD-E testMSE={testMSE_oodE:.3e}  rollEnd={roll_oodE:.3e}  driftE={driftE_oodE:.3e}  driftL={driftL_oodE:.3e}")
        print(f"  OOD-O testMSE={testMSE_oodO:.3e}  rollEnd={roll_oodO:.3e}  driftE={driftE_oodO:.3e}  driftL={driftL_oodO:.3e}")

    print(f"\n[saved] {out_path}")
    print("Done (2-body masked). No aggregation/summary here.")


if __name__ == "__main__":
    main()
