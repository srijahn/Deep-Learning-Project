"""
NTK Spectral Analysis: Regularization Effects on Generalization
Based on: Hu et al. (2021) "Regularization Matters: A Nonparametric Perspective
on Overparametrized Neural Network" (AISTATS 2021)

This script runs ALL experiments from the project proposal:
1. NTK computation under different regularization settings
2. Spectral analysis (eigenvalue distributions)
3. Generalization evaluation (L2 error + test accuracy)
4. Hybrid regularization study
5. Label noise robustness

Usage:
    python ntk_experiment.py --task regression   # synthetic regression
    python ntk_experiment.py --task mnist         # MNIST binary classification
    python ntk_experiment.py --task mnist --mnist-source torchvision --mnist-reps 100
    python ntk_experiment.py --task all           # run everything
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os, argparse, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────── REPRODUCIBILITY ────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/data", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1.  NEURAL NETWORK ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════

class OneHiddenReLU(nn.Module):
    """One-hidden-layer ReLU network (matches paper setup exactly)."""
    def __init__(self, d_in: int, width: int = 500, dropout_p: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, width, bias=False)
        self.fc2 = nn.Linear(width, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.1)
        nn.init.uniform_(self.fc2.weight, -1, 1)
        # Fix second layer sign only (as in paper: a_r ∈ {-1,+1})
        with torch.no_grad():
            self.fc2.weight.data = torch.sign(self.fc2.weight.data)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1) / (self.fc1.weight.shape[0] ** 0.5)


class TwoHiddenReLU(nn.Module):
    """Two-hidden-layer ReLU (used in paper's numerical experiments)."""
    def __init__(self, d_in: int, width: int = 500, dropout_p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width, bias=False),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(width, width, bias=False),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(width, 1, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════
# 2.  NTK COMPUTATION (empirical)
# ══════════════════════════════════════════════════════════════════════════

def compute_empirical_ntk(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """
    Compute empirical NTK matrix K[i,j] = <∇f(x_i), ∇f(x_j)>.
    Uses the Jacobian via torch.autograd.
    Returns numpy array of shape (n, n).
    """
    model.eval()
    n = X.shape[0]
    grads = []
    for i in range(n):
        model.zero_grad()
        out = model(X[i:i+1])
        out.backward()
        g = torch.cat([p.grad.view(-1) for p in model.parameters()
                       if p.grad is not None])
        grads.append(g.detach().cpu())
    G = torch.stack(grads)          # (n, num_params)
    K = (G @ G.T).numpy()           # (n, n)
    return K


def ntk_eigenvalues(K: np.ndarray) -> np.ndarray:
    """Return sorted (descending) eigenvalues of the NTK matrix."""
    eigs = np.linalg.eigvalsh(K)
    return np.sort(eigs)[::-1]


def spectral_summary_metrics(eigs: np.ndarray, small_thresh: float = 0.01) -> dict:
    """Compute spectrum summary metrics used in proposal analysis."""
    eigs_abs = np.abs(eigs)
    lmax = float(eigs_abs[0])
    lmin = float(eigs_abs[-1])
    cond = lmax / (lmin + 1e-12)
    p = eigs_abs + 1e-12
    p = p / p.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-15))))
    small_count = int(np.sum(eigs_abs < small_thresh * lmax))
    return {
        "lmax": lmax,
        "lmin": lmin,
        "condition_number": cond,
        "effective_rank": eff_rank,
        "small_eig_count": small_count,
    }


# ══════════════════════════════════════════════════════════════════════════
# 3.  KERNEL RIDGE REGRESSION WITH NTK (baseline)
# ══════════════════════════════════════════════════════════════════════════

def analytical_ntk_kernel(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """
    Closed-form NTK for one-hidden-layer ReLU on the unit sphere:
        h(s,t) = (s^T t * (π - arccos(s^T t))) / (2π)
    Eq. (3.2) in Hu et al. 2021.
    """
    if Y is None:
        Y = X
    # normalize to sphere
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    dot = np.clip(Xn @ Yn.T, -1.0, 1.0)
    K = dot * (np.pi - np.arccos(dot)) / (2 * np.pi)
    return K


def kernel_ridge_regression(K_train: np.ndarray, y_train: np.ndarray,
                             K_test: np.ndarray, mu: float) -> np.ndarray:
    """
    KRR predictor: f(x) = k(x,X)(K + μI)^{-1} y   (Eq. 3.4 in paper)
    """
    n = K_train.shape[0]
    alpha = np.linalg.solve(K_train + mu * np.eye(n), y_train)
    return K_test @ alpha


# ══════════════════════════════════════════════════════════════════════════
# 4.  TRAINING ROUTINES
# ══════════════════════════════════════════════════════════════════════════

def train_model(model, X_tr, y_tr, X_te, y_te,
                weight_decay: float = 0.0,
                epochs: int = 3000,
                lr: float = 1e-3,
                task: str = "regression",
                noise_sigma: float = 0.0,
                collect_ntk_every: int = 500):
    """
    Train model with RMSProp (matches paper's numerical section).
    Returns dict with training history and NTK snapshots.
    """
    optimizer = optim.RMSprop(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "test_metric": [], "ntk_eigs": {}}

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = criterion(pred, y_tr)
        if epoch > 0:
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                if task == "regression":
                    te_pred = model(X_te)
                    metric = nn.MSELoss()(te_pred, y_te).item()
                else:
                    te_pred = model(X_te)
                    preds_bin = (te_pred > 0).float()
                    metric = (preds_bin == y_te).float().mean().item()
            history["train_loss"].append(loss.item())
            history["test_metric"].append(metric)

        # Collect NTK snapshots at selected epochs
        if epoch in [0, collect_ntk_every, 2 * collect_ntk_every, epochs]:
            # Use a subsample for speed
            n_ntk = min(50, X_tr.shape[0])
            idx = np.random.choice(X_tr.shape[0], n_ntk, replace=False)
            K = compute_empirical_ntk(model, X_tr[idx])
            history["ntk_eigs"][epoch] = ntk_eigenvalues(K)

    return history


# ══════════════════════════════════════════════════════════════════════════
# 5.  SYNTHETIC REGRESSION EXPERIMENT  (Section 6.1 of paper)
# ══════════════════════════════════════════════════════════════════════════

def run_synthetic_regression():
    """
    Reproduce Figure 1 of the paper: L2 estimation error vs noise level σ.
    Four settings: No-reg, WeightDecay, Dropout, Hybrid.
    Two target functions: f*(x)=0 and f*(x)=x^T x.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Synthetic Regression (Paper §6.1 analog)")
    print("="*60)

    d = 5           # input dimension (paper uses d=2; we use 5 for harder problem)
    n_train = 150
    n_test = 500
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    n_reps = 20     # replications (paper uses 100; reduce for speed)

    target_fns = {
        "f1_zero":    lambda x: np.zeros(x.shape[0]),
        "f2_quadratic": lambda x: np.sum(x**2, axis=1),
    }

    settings = {
        "No-reg":      dict(weight_decay=0.0, dropout_p=0.0),
        "WeightDecay": dict(weight_decay=1e-3, dropout_p=0.0),
        "Dropout":     dict(weight_decay=0.0, dropout_p=0.3),
        "Hybrid":      dict(weight_decay=1e-3, dropout_p=0.3),
    }
    # Also include KRR with NTK as in paper
    krr_mu = 0.1

    all_results = {}
    for fn_name, f_star in target_fns.items():
        all_results[fn_name] = {s: [] for s in list(settings.keys()) + ["KRR+NTK"]}
        for sigma in sigmas:
            rep_errors = {s: [] for s in list(settings.keys()) + ["KRR+NTK"]}
            for rep in range(n_reps):
                # Data
                X_tr = np.random.uniform(-1, 1, (n_train, d)).astype(np.float32)
                X_te = np.random.uniform(-1, 1, (n_test, d)).astype(np.float32)
                y_tr_clean = f_star(X_tr).astype(np.float32)
                y_te_clean = f_star(X_te).astype(np.float32)
                noise = np.random.normal(0, sigma, n_train).astype(np.float32)
                y_tr = y_tr_clean + noise

                # Tensors
                Xt = torch.tensor(X_tr)
                yt = torch.tensor(y_tr)
                Xte_t = torch.tensor(X_te)
                yte_t = torch.tensor(y_te_clean)

                # KRR with analytical NTK (paper baseline)
                K_tr = analytical_ntk_kernel(X_tr)
                K_te = analytical_ntk_kernel(X_te, X_tr)
                pred_krr = kernel_ridge_regression(K_tr, y_tr, K_te, mu=krr_mu)
                l2_krr = np.mean((pred_krr - y_te_clean)**2)
                rep_errors["KRR+NTK"].append(l2_krr)

                # Neural network settings
                for s_name, cfg in settings.items():
                    torch.manual_seed(rep)
                    model = TwoHiddenReLU(d, width=200,
                                          dropout_p=cfg["dropout_p"])
                    hist = train_model(model, Xt, yt, Xte_t, yte_t,
                                       weight_decay=cfg["weight_decay"],
                                       epochs=1500, lr=1e-3,
                                       task="regression",
                                       collect_ntk_every=5000)  # skip NTK collection for speed
                    model.eval()
                    with torch.no_grad():
                        pred = model(Xte_t).numpy()
                    l2_err = np.mean((pred - y_te_clean)**2)
                    rep_errors[s_name].append(l2_err)

            for s_name in rep_errors:
                errs = rep_errors[s_name]
                all_results[fn_name][s_name].append(
                    (sigma, np.mean(errs), np.std(errs)))

    # Save results
    np.save("results/data/synthetic_regression.npy", all_results, allow_pickle=True)
    print("  Saved results/data/synthetic_regression.npy")
    plot_synthetic_regression(all_results)
    return all_results


def plot_synthetic_regression(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"No-reg": "#e74c3c", "WeightDecay": "#3498db",
              "Dropout": "#2ecc71", "Hybrid": "#9b59b6", "KRR+NTK": "#f39c12"}
    markers = {"No-reg": "o", "WeightDecay": "s", "Dropout": "^",
               "Hybrid": "D", "KRR+NTK": "*"}

    for ax, (fn_name, results) in zip(axes, all_results.items()):
        for s_name, data in results.items():
            sigmas = [d[0] for d in data]
            means  = [d[1] for d in data]
            stds   = [d[2] for d in data]
            ax.plot(sigmas, means, label=s_name,
                    color=colors[s_name], marker=markers[s_name], linewidth=2)
            ax.fill_between(sigmas,
                            [m-s for m,s in zip(means,stds)],
                            [m+s for m,s in zip(means,stds)],
                            color=colors[s_name], alpha=0.15)
        fn_label = "f*(x) = 0" if "f1" in fn_name else "f*(x) = x^T x"
        ax.set_title(fn_label, fontsize=14)
        ax.set_xlabel("Noise σ", fontsize=12)
        ax.set_ylabel("L2 Estimation Error", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("L2 Estimation Error vs. Noise Level\n(Synthetic Regression)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("results/figures/synthetic_regression.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved results/figures/synthetic_regression.png")


# ══════════════════════════════════════════════════════════════════════════
# 6.  NTK SPECTRAL ANALYSIS EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════

def run_spectral_analysis():
    """
    Core spectral analysis:
    - Train 4 models (no-reg, WD, dropout, hybrid)
    - Compute NTK before/after training
    - Plot eigenvalue distributions
    - Analyze conditioning, eigenvalue suppression
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: NTK Spectral Analysis")
    print("="*60)

    d, n, width = 10, 80, 300
    X = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    X_t = torch.tensor(X)

    # True function for labels
    f_star = lambda x: np.sum(x**2, axis=1).astype(np.float32)
    y_clean = f_star(X)
    noise = np.random.normal(0, 0.2, n).astype(np.float32)
    y = y_clean + noise
    y_t = torch.tensor(y)

    settings = {
        "No-reg":      dict(weight_decay=0.0, dropout_p=0.0),
        "WeightDecay": dict(weight_decay=1e-3, dropout_p=0.0),
        "Dropout":     dict(weight_decay=0.0, dropout_p=0.3),
        "Hybrid":      dict(weight_decay=1e-3, dropout_p=0.3),
    }

    spectral_results = {}

    for s_name, cfg in settings.items():
        print(f"  Computing NTK for: {s_name}")
        torch.manual_seed(SEED)
        model = TwoHiddenReLU(d, width=width, dropout_p=cfg["dropout_p"])

        # Before training NTK
        K_before = compute_empirical_ntk(model, X_t)
        eigs_before = ntk_eigenvalues(K_before)

        # Train
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3,
                                   weight_decay=cfg["weight_decay"])
        model.train()
        for _ in range(2000):
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(X_t), y_t)
            loss.backward()
            optimizer.step()

        # After training NTK
        K_after = compute_empirical_ntk(model, X_t)
        eigs_after = ntk_eigenvalues(K_after)

        metrics_before = spectral_summary_metrics(eigs_before)
        metrics_after = spectral_summary_metrics(eigs_after)
        cond_before = metrics_before["condition_number"]
        cond_after = metrics_after["condition_number"]

        spectral_results[s_name] = {
            "eigs_before": eigs_before,
            "eigs_after":  eigs_after,
            "K_before":    K_before,
            "K_after":     K_after,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "cond_before": cond_before,
            "cond_after":  cond_after,
        }
        print(f"    Condition number: {cond_before:.1f} → {cond_after:.1f}")
        print(
            "    Effective rank: "
            f"{metrics_before['effective_rank']:.2f} → {metrics_after['effective_rank']:.2f} | "
            "Small eig count (<1% λmax): "
            f"{metrics_before['small_eig_count']} → {metrics_after['small_eig_count']}"
        )

    np.save("results/data/spectral_analysis.npy", spectral_results, allow_pickle=True)
    print("  Saved results/data/spectral_analysis.npy")
    plot_spectral_analysis(spectral_results)
    return spectral_results


def plot_spectral_analysis(spectral_results):
    settings = list(spectral_results.keys())
    n_settings = len(settings)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    # Row 1: Eigenvalue spectrum (log scale) — before training
    for i, (s_name, col) in enumerate(zip(settings, colors)):
        ax = fig.add_subplot(gs[0, i])
        eigs = spectral_results[s_name]["eigs_before"]
        ax.semilogy(range(1, len(eigs)+1), np.abs(eigs) + 1e-12,
                    color=col, linewidth=2)
        ax.set_title(f"{s_name}\n(Before Training)", fontsize=10)
        ax.set_xlabel("Index")
        ax.set_ylabel("Eigenvalue")
        ax.grid(True, alpha=0.3)

    # Row 2: Eigenvalue spectrum — after training
    for i, (s_name, col) in enumerate(zip(settings, colors)):
        ax = fig.add_subplot(gs[1, i])
        eigs = spectral_results[s_name]["eigs_after"]
        ax.semilogy(range(1, len(eigs)+1), np.abs(eigs) + 1e-12,
                    color=col, linewidth=2, linestyle="-")
        ax.set_title(f"{s_name}\n(After Training)", fontsize=10)
        ax.set_xlabel("Index")
        ax.set_ylabel("Eigenvalue")
        ax.grid(True, alpha=0.3)

    # Row 3 left: Before vs After overlay (all settings)
    ax = fig.add_subplot(gs[2, :2])
    for s_name, col in zip(settings, colors):
        e_b = spectral_results[s_name]["eigs_before"]
        e_a = spectral_results[s_name]["eigs_after"]
        ax.semilogy(range(1, len(e_b)+1), np.abs(e_b)+1e-12,
                    color=col, linewidth=1.5, linestyle="--", alpha=0.6)
        ax.semilogy(range(1, len(e_a)+1), np.abs(e_a)+1e-12,
                    color=col, linewidth=2, label=s_name)
    ax.set_title("Eigenvalue Spectra Comparison (solid=after, dashed=before)", fontsize=11)
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 3 right: Condition number comparison
    ax = fig.add_subplot(gs[2, 2:])
    conds_before = [spectral_results[s]["cond_before"] for s in settings]
    conds_after  = [spectral_results[s]["cond_after"]  for s in settings]
    x_pos = np.arange(n_settings)
    w = 0.35
    bars1 = ax.bar(x_pos - w/2, conds_before, w, label="Before Training",
                   color=colors, alpha=0.5, edgecolor="black")
    bars2 = ax.bar(x_pos + w/2, conds_after,  w, label="After Training",
                   color=colors, alpha=0.9, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(settings, fontsize=9)
    ax.set_ylabel("Condition Number (λ_max/λ_min)")
    ax.set_title("NTK Matrix Condition Number", fontsize=11)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("NTK Eigenvalue Spectral Analysis Under Different Regularization",
                 fontsize=14, y=1.01)
    plt.savefig("results/figures/spectral_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved results/figures/spectral_analysis.png")


# ══════════════════════════════════════════════════════════════════════════
# 7.  MNIST EXPERIMENT  (Section 6.2 of paper)
# ══════════════════════════════════════════════════════════════════════════

def run_mnist_experiment(mnist_source: str = "digits", n_reps_override: int = None):
    """
    Reproduce Figure 2a of the paper:
    Test misclassification rate vs. additive label noise σ.
    Digits 5 vs 8 (matches paper's MNIST binary classification).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: MNIST Binary Classification (Paper §6.2)")
    print("="*60)

    if mnist_source == "torchvision":
        try:
            from torchvision import datasets, transforms  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for --mnist-source torchvision. "
                "Install with: pip install torchvision"
            ) from exc

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        train_mask = (train_set.targets == 5) | (train_set.targets == 8)
        test_mask = (test_set.targets == 5) | (test_set.targets == 8)

        X_tr = train_set.data[train_mask].float().view(-1, 784).numpy() / 255.0
        y_tr_clean = ((train_set.targets[train_mask] == 8).float() * 2 - 1).numpy().astype(np.float32)
        X_te = test_set.data[test_mask].float().view(-1, 784).numpy() / 255.0
        y_te = ((test_set.targets[test_mask] == 8).float() * 2 - 1).numpy().astype(np.float32)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)
        print(f"  Dataset: torchvision MNIST 5 vs 8 (train={len(X_tr)}, test={len(X_te)})")
    else:
        # Default fast path: sklearn 8x8 digits
        digits = load_digits()
        mask = (digits.target == 5) | (digits.target == 8)
        X_all = digits.data[mask].astype(np.float32)
        y_all = (digits.target[mask] == 8).astype(np.float32) * 2 - 1  # {-1,+1}

        scaler = StandardScaler()
        X_all = scaler.fit_transform(X_all).astype(np.float32)

        X_tr, X_te, y_tr_clean, y_te = train_test_split(
            X_all, y_all, test_size=0.25, random_state=SEED)
        print(f"  Dataset: sklearn digits 5 vs 8 (train={len(X_tr)}, test={len(X_te)})")

    Xtr_t = torch.tensor(X_tr)
    Xte_t = torch.tensor(X_te)
    yte_t = torch.tensor(y_te)

    sigmas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    default_reps = 10 if mnist_source == "digits" else 20
    n_reps = n_reps_override if n_reps_override is not None else default_reps
    print(f"  Repetitions per noise level: {n_reps}")
    settings = {
        "No-reg":      dict(weight_decay=0.0, dropout_p=0.0),
        "WeightDecay": dict(weight_decay=1e-3, dropout_p=0.0),
        "Dropout":     dict(weight_decay=0.0, dropout_p=0.3),
        "Hybrid":      dict(weight_decay=1e-3, dropout_p=0.3),
    }

    results = {s: [] for s in settings}

    for sigma in sigmas:
        print(f"  σ = {sigma:.2f}")
        rep_errors = {s: [] for s in settings}
        for rep in range(n_reps):
            noise = np.random.normal(0, sigma, y_tr_clean.shape).astype(np.float32)
            y_tr_noisy = torch.tensor(y_tr_clean + noise)

            for s_name, cfg in settings.items():
                torch.manual_seed(rep * 100 + SEED)
                model = TwoHiddenReLU(X_tr.shape[1], width=200,
                                      dropout_p=cfg["dropout_p"])
                optimizer = optim.RMSprop(model.parameters(), lr=1e-3,
                                           weight_decay=cfg["weight_decay"])
                # Train
                for _ in range(800):
                    model.train()
                    optimizer.zero_grad()
                    loss = nn.MSELoss()(model(Xtr_t), y_tr_noisy)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    pred = model(Xte_t)
                    acc = ((pred > 0).float() == (yte_t > 0).float()).float().mean().item()
                rep_errors[s_name].append(1.0 - acc)   # misclassification rate

        for s_name in settings:
            errs = rep_errors[s_name]
            results[s_name].append((sigma, np.mean(errs)*100, np.std(errs)*100))

    np.save("results/data/mnist_results.npy", results, allow_pickle=True)
    print("  Saved results/data/mnist_results.npy")
    plot_mnist_results(results)
    return results


def plot_mnist_results(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"No-reg":"#e74c3c","WeightDecay":"#3498db",
              "Dropout":"#2ecc71","Hybrid":"#9b59b6"}
    markers = {"No-reg":"o","WeightDecay":"s","Dropout":"^","Hybrid":"D"}

    for s_name, data in results.items():
        sigmas = [d[0] for d in data]
        means  = [d[1] for d in data]
        stds   = [d[2] for d in data]
        ax.plot(sigmas, means, label=s_name, color=colors[s_name],
                marker=markers[s_name], linewidth=2, markersize=7)
        ax.fill_between(sigmas,
                        [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)],
                        color=colors[s_name], alpha=0.15)

    ax.set_xlabel("Noise σ (additive label noise)", fontsize=12)
    ax.set_ylabel("Test Misclassification Rate (%)", fontsize=12)
    ax.set_title("MNIST Binary Classification: Misclassification vs. Noise\n(Digits 5 vs. 8)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/mnist_classification.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved results/figures/mnist_classification.png")


# ══════════════════════════════════════════════════════════════════════════
# 8.  HYBRID REGULARIZATION STUDY: NTK eigenvalue vs performance
# ══════════════════════════════════════════════════════════════════════════

def run_hybrid_study():
    """
    Systematic grid search over (weight_decay, dropout_p) combinations.
    For each combination:
      - Compute final NTK eigenvalue metrics (condition number, effective rank)
      - Measure test L2 error on noisy regression
    Shows that hybrid setting can achieve best spectral + performance balance.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Hybrid Regularization Grid Study")
    print("="*60)

    d, n, width = 8, 100, 200
    n_test = 300
    sigma = 0.3  # fixed noise

    X_tr = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    X_te = np.random.uniform(-1, 1, (n_test, d)).astype(np.float32)
    f_star = lambda x: np.sum(x**2, axis=1).astype(np.float32)
    y_tr_clean = f_star(X_tr)
    y_te = f_star(X_te)
    noise = np.random.normal(0, sigma, n).astype(np.float32)
    y_tr = y_tr_clean + noise

    Xtr_t = torch.tensor(X_tr)
    Xte_t = torch.tensor(X_te)
    ytr_t = torch.tensor(y_tr)

    wd_values = [0.0, 1e-4, 1e-3, 5e-3, 1e-2]
    dp_values = [0.0, 0.1, 0.2, 0.3, 0.5]

    grid_l2    = np.zeros((len(wd_values), len(dp_values)))
    grid_cond  = np.zeros((len(wd_values), len(dp_values)))
    grid_effrank = np.zeros((len(wd_values), len(dp_values)))

    for i, wd in enumerate(wd_values):
        for j, dp in enumerate(dp_values):
            torch.manual_seed(SEED)
            model = TwoHiddenReLU(d, width=width, dropout_p=dp)
            optimizer = optim.RMSprop(model.parameters(), lr=1e-3,
                                       weight_decay=wd)
            for _ in range(1500):
                model.train()
                optimizer.zero_grad()
                loss = nn.MSELoss()(model(Xtr_t), ytr_t)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_te = model(Xte_t).numpy()
            l2_err = np.mean((pred_te - y_te)**2)

            # NTK metrics
            n_ntk = 50
            idx = np.random.choice(n, n_ntk, replace=False)
            K = compute_empirical_ntk(model, Xtr_t[idx])
            eigs = ntk_eigenvalues(K)
            cond = eigs[0] / (np.abs(eigs[-1]) + 1e-12)
            # Effective rank = exp(entropy of normalized eigenvalues)
            eigs_pos = np.abs(eigs) + 1e-12
            p = eigs_pos / eigs_pos.sum()
            eff_rank = np.exp(-np.sum(p * np.log(p + 1e-15)))

            grid_l2[i, j]     = l2_err
            grid_cond[i, j]   = cond
            grid_effrank[i, j] = eff_rank
            print(f"  WD={wd:.4f}, Dropout={dp:.1f} → L2={l2_err:.4f}, cond={cond:.1f}, eff_rank={eff_rank:.1f}")

    np.save("results/data/hybrid_grid.npy",
            {"l2": grid_l2, "cond": grid_cond, "eff_rank": grid_effrank,
             "wd_values": wd_values, "dp_values": dp_values},
            allow_pickle=True)
    print("  Saved results/data/hybrid_grid.npy")
    plot_hybrid_study(grid_l2, grid_cond, grid_effrank, wd_values, dp_values)
    return grid_l2, grid_cond, grid_effrank


def plot_hybrid_study(grid_l2, grid_cond, grid_effrank, wd_values, dp_values):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    wd_labels = [str(w) for w in wd_values]
    dp_labels = [str(d) for d in dp_values]

    def heatmap(ax, data, title, cmap, fmt=".3f"):
        sns.heatmap(data, ax=ax, annot=True, fmt=fmt,
                    xticklabels=dp_labels, yticklabels=wd_labels,
                    cmap=cmap, linewidths=0.5)
        ax.set_xlabel("Dropout Rate", fontsize=11)
        ax.set_ylabel("Weight Decay", fontsize=11)
        ax.set_title(title, fontsize=12)

    heatmap(axes[0], grid_l2,      "Test L2 Error (↓ better)",     "YlOrRd")
    heatmap(axes[1], np.log10(grid_cond+1), "log10(Condition Number) (↓ better)", "YlOrRd", fmt=".2f")
    heatmap(axes[2], grid_effrank, "Effective Rank (↑ broader spec.)", "YlGn", fmt=".1f")

    plt.suptitle("Hybrid Regularization Grid: (Weight Decay × Dropout)\nσ_noise = 0.3",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("results/figures/hybrid_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved results/figures/hybrid_grid.png")


# ══════════════════════════════════════════════════════════════════════════
# 9.  NTK EIGENVALUE vs L2 ERROR CORRELATION
# ══════════════════════════════════════════════════════════════════════════

def run_eigenvalue_error_correlation():
    """
    Show that NTK eigenvalue metrics (λ_min, condition number)
    are predictive of L2 generalization error.
    Vary regularization strength and plot the relationship.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: NTK Eigenvalue ↔ L2 Error Correlation")
    print("="*60)

    d, n, width = 8, 80, 200
    n_test = 300
    sigma = 0.25

    X_tr = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    X_te = np.random.uniform(-1, 1, (n_test, d)).astype(np.float32)
    f_star = lambda x: np.sum(x**2, axis=1).astype(np.float32)
    y_te = f_star(X_te)
    y_tr = f_star(X_tr) + np.random.normal(0, sigma, n).astype(np.float32)
    Xtr_t = torch.tensor(X_tr)
    Xte_t = torch.tensor(X_te)
    ytr_t = torch.tensor(y_tr)

    wd_range = np.logspace(-4, -1, 15)
    lmin_vals, lmax_vals, cond_vals, l2_vals = [], [], [], []

    for wd in wd_range:
        torch.manual_seed(SEED)
        model = TwoHiddenReLU(d, width=width, dropout_p=0.0)
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=float(wd))
        for _ in range(1500):
            model.train()
            optimizer.zero_grad()
            nn.MSELoss()(model(Xtr_t), ytr_t).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            l2_err = np.mean((model(Xte_t).numpy() - y_te)**2)

        K = compute_empirical_ntk(model, Xtr_t[:50])
        eigs = ntk_eigenvalues(K)
        lmin_vals.append(np.abs(eigs[-1]))
        lmax_vals.append(eigs[0])
        cond_vals.append(eigs[0] / (np.abs(eigs[-1]) + 1e-12))
        l2_vals.append(l2_err)

    r_lmin, p_lmin = pearsonr(lmin_vals, l2_vals)
    rho_cond, p_cond = spearmanr(cond_vals, l2_vals)

    corr_results = {
        "wd_range": wd_range,
        "lmin_vals": np.array(lmin_vals),
        "lmax_vals": np.array(lmax_vals),
        "cond_vals": np.array(cond_vals),
        "l2_vals": np.array(l2_vals),
        "pearson_lmin_l2": float(r_lmin),
        "pearson_lmin_l2_pvalue": float(p_lmin),
        "spearman_cond_l2": float(rho_cond),
        "spearman_cond_l2_pvalue": float(p_cond),
    }
    np.save("results/data/eigenvalue_error_correlation.npy", corr_results, allow_pickle=True)
    print(
        "  λ_min ↔ L2 (Pearson): "
        f"r={r_lmin:.3f}, p={p_lmin:.4f}"
    )
    print(
        "  κ ↔ L2 (Spearman): "
        f"rho={rho_cond:.3f}, p={p_cond:.4f}"
    )
    print("  Saved results/data/eigenvalue_error_correlation.npy")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc = axes[0].scatter(lmin_vals, l2_vals, c=np.log10(wd_range), cmap="viridis", s=80)
    plt.colorbar(sc, ax=axes[0], label="log10(Weight Decay)")
    axes[0].set_xlabel("λ_min of NTK", fontsize=12)
    axes[0].set_ylabel("Test L2 Error", fontsize=12)
    axes[0].set_title(f"λ_min vs. L2 Error (r={r_lmin:.2f})", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    sc2 = axes[1].scatter(cond_vals, l2_vals, c=np.log10(wd_range), cmap="plasma", s=80)
    plt.colorbar(sc2, ax=axes[1], label="log10(Weight Decay)")
    axes[1].set_xlabel("Condition Number (λ_max/λ_min)", fontsize=12)
    axes[1].set_ylabel("Test L2 Error", fontsize=12)
    axes[1].set_title(f"Condition Number vs. L2 Error (rho={rho_cond:.2f})", fontsize=12)
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("NTK Spectral Properties vs. Generalization Error\n(Varying Weight Decay)", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/figures/eigenvalue_error_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved results/figures/eigenvalue_error_correlation.png")
    return corr_results


# ══════════════════════════════════════════════════════════════════════════
# 10.  TRAINING DYNAMICS: L2 Error & NTK spectrum over time
# ══════════════════════════════════════════════════════════════════════════

def run_training_dynamics():
    """
    Replicate Figure 2b of the paper:
    Training RMSE and test error over iterations for ONN vs ONN+L2.
    Also tracks NTK eigenvalue evolution during training.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 6: Training Dynamics (Paper Fig 2b analog)")
    print("="*60)

    d, n, width = 8, 100, 300
    n_test = 300
    sigma = 0.5

    X_tr = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    X_te = np.random.uniform(-1, 1, (n_test, d)).astype(np.float32)
    f_star = lambda x: np.sum(x**2, axis=1).astype(np.float32)
    y_te = f_star(X_te)
    y_tr_clean = f_star(X_tr)
    y_tr = y_tr_clean + np.random.normal(0, sigma, n).astype(np.float32)

    Xtr_t = torch.tensor(X_tr)
    Xte_t = torch.tensor(X_te)
    ytr_t = torch.tensor(y_tr)
    yte_clean_t = torch.tensor(y_te)

    epochs = 3000
    record_every = 50
    ntk_snapshot_epochs = [0, 500, 1000, 2000, 3000]

    settings = {
        "No-reg (ONN)":   dict(weight_decay=0.0,  dropout_p=0.0),
        "WeightDecay (ONN+L2)": dict(weight_decay=1e-3, dropout_p=0.0),
        "Dropout":        dict(weight_decay=0.0,  dropout_p=0.3),
        "Hybrid":         dict(weight_decay=1e-3, dropout_p=0.3),
    }

    dyn_results = {}

    for s_name, cfg in settings.items():
        print(f"  Training: {s_name}")
        torch.manual_seed(SEED)
        model = TwoHiddenReLU(d, width=width, dropout_p=cfg["dropout_p"])
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3,
                                   weight_decay=cfg["weight_decay"])

        train_rmse_hist, test_l2_hist, epoch_hist = [], [], []
        ntk_snapshots = {}

        for ep in range(epochs + 1):
            model.train()
            if ep > 0:
                optimizer.zero_grad()
                loss = nn.MSELoss()(model(Xtr_t), ytr_t)
                loss.backward()
                optimizer.step()

            if ep % record_every == 0:
                model.eval()
                with torch.no_grad():
                    tr_pred = model(Xtr_t)
                    te_pred = model(Xte_t)
                tr_rmse = torch.sqrt(nn.MSELoss()(tr_pred, ytr_t)).item()
                te_l2 = nn.MSELoss()(te_pred, yte_clean_t).item()
                train_rmse_hist.append(tr_rmse)
                test_l2_hist.append(te_l2)
                epoch_hist.append(ep)

            if ep in ntk_snapshot_epochs:
                K = compute_empirical_ntk(model, Xtr_t[:50])
                ntk_snapshots[ep] = ntk_eigenvalues(K)

        dyn_results[s_name] = {
            "epochs": epoch_hist,
            "train_rmse": train_rmse_hist,
            "test_l2": test_l2_hist,
            "ntk_snapshots": ntk_snapshots,
        }

    np.save("results/data/training_dynamics.npy", dyn_results, allow_pickle=True)
    print("  Saved results/data/training_dynamics.npy")
    plot_training_dynamics(dyn_results, ntk_snapshot_epochs)
    return dyn_results


def plot_training_dynamics(dyn_results, snapshot_epochs):
    settings = list(dyn_results.keys())
    colors = {"No-reg (ONN)":"#e74c3c","WeightDecay (ONN+L2)":"#3498db",
              "Dropout":"#2ecc71","Hybrid":"#9b59b6"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Training RMSE over time
    ax = axes[0, 0]
    for s_name in settings:
        d = dyn_results[s_name]
        ax.plot(d["epochs"], d["train_rmse"], label=s_name,
                color=colors[s_name], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training RMSE")
    ax.set_title("Training RMSE over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top right: Test L2 error over time
    ax = axes[0, 1]
    for s_name in settings:
        d = dyn_results[s_name]
        ax.plot(d["epochs"], d["test_l2"], label=s_name,
                color=colors[s_name], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test L2 Error")
    ax.set_title("Test L2 Error over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom: NTK eigenvalue evolution at snapshots for two key settings
    for ax_idx, s_name in enumerate(["No-reg (ONN)", "WeightDecay (ONN+L2)"]):
        ax = axes[1, ax_idx]
        snaps = dyn_results[s_name]["ntk_snapshots"]
        cmap = plt.cm.Blues if ax_idx == 0 else plt.cm.Oranges
        ep_list = sorted(snaps.keys())
        for k, ep in enumerate(ep_list):
            eigs = snaps[ep]
            alpha = 0.3 + 0.7 * (k / max(len(ep_list)-1, 1))
            ax.semilogy(range(1, len(eigs)+1),
                        np.abs(eigs) + 1e-12,
                        color=cmap(0.3 + 0.6*k/max(len(ep_list)-1,1)),
                        label=f"ep={ep}", linewidth=1.8, alpha=alpha)
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue (log)")
        ax.set_title(f"NTK Eigenvalue Evolution: {s_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Dynamics and NTK Spectral Evolution", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/figures/training_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved results/figures/training_dynamics.png")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["regression","spectral","mnist",
                                            "hybrid","dynamics","correlation","all"],
                        default="all")
    parser.add_argument(
        "--mnist-source",
        choices=["digits", "torchvision"],
        default="digits",
        help="Dataset backend for MNIST experiment: fast sklearn digits or full torchvision MNIST.",
    )
    parser.add_argument(
        "--mnist-reps",
        type=int,
        default=None,
        help="Override number of repetitions per noise level for MNIST experiment.",
    )
    args = parser.parse_args()

    print("\n" + "🔬 " * 20)
    print("NTK Spectral Analysis: Regularization Effects")
    print("Based on Hu et al. (2021), AISTATS")
    print("🔬 " * 20)

    if args.task in ("regression", "all"):
        run_synthetic_regression()

    if args.task in ("spectral", "all"):
        run_spectral_analysis()

    if args.task in ("mnist", "all"):
        run_mnist_experiment(mnist_source=args.mnist_source, n_reps_override=args.mnist_reps)

    if args.task in ("hybrid", "all"):
        run_hybrid_study()

    if args.task in ("correlation", "all"):
        run_eigenvalue_error_correlation()

    if args.task in ("dynamics", "all"):
        run_training_dynamics()

    print("\n✅ All experiments complete! Results in: results/")
