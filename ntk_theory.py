"""
NTK Theory Reference — Key Equations from Hu et al. (2021)
============================================================
This module consolidates the theoretical backbone of the project.
Import and use these in your report / analysis notebook.
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# 1. CLOSED-FORM NTK for one-hidden-layer ReLU on the unit sphere
#    Eq. (3.2) in Hu et al. (2021)
# ──────────────────────────────────────────────────────────────────────────

def relu_ntk(s: np.ndarray, t: np.ndarray) -> float:
    """
    h(s,t) = (s^T t * (π - arccos(s^T t))) / (2π)

    Valid when s, t ∈ S^{d-1} (unit sphere).
    """
    dot = np.clip(np.dot(s, t), -1.0, 1.0)
    return dot * (np.pi - np.arccos(dot)) / (2 * np.pi)


def relu_ntk_matrix(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """
    Full NTK matrix H_∞ ∈ R^{n×n} or cross-matrix R^{n×m}.
    """
    if Y is None:
        Y = X
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    dot = np.clip(Xn @ Yn.T, -1.0, 1.0)
    return dot * (np.pi - np.arccos(dot)) / (2 * np.pi)


# ──────────────────────────────────────────────────────────────────────────
# 2. KERNEL RIDGE REGRESSION with NTK
#    Eq. (3.3) and representer theorem (3.4) in Hu et al. (2021)
#
#    min_{f ∈ N}  ½ Σ (y_i - f(x_i))²  +  (μ/2) ‖f‖²_N
#
#    Solution: f̂(x) = h(x, X) (H_∞ + μI)^{-1} y
# ──────────────────────────────────────────────────────────────────────────

def krr_predict(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, mu: float) -> np.ndarray:
    """
    Kernel Ridge Regression with NTK kernel.
    mu: regularization parameter (λ in some notations)
    """
    K_train = relu_ntk_matrix(X_train)
    K_test  = relu_ntk_matrix(X_test, X_train)
    n = K_train.shape[0]
    alpha = np.linalg.solve(K_train + mu * np.eye(n), y_train)
    return K_test @ alpha


# ──────────────────────────────────────────────────────────────────────────
# 3. EIGENVALUE DECAY RATE (Lemma 3.1)
#
#    λ_j ≍ j^{-d/(d-1)}   for the one-hidden-layer ReLU NTK on S^{d-1}
#
#    This determines the minimax optimal rate via:
#    If λ_j ≍ j^{-2ν}, the minimax rate is n^{-2ν/(2ν+1)}
#    For ReLU NTK: 2ν = d/(d-1) so ν = d/(2(d-1))
#    → Rate = n^{-d/(2d-1)}   (Theorem 3.2)
# ──────────────────────────────────────────────────────────────────────────

def theoretical_eigenvalue_decay(j: np.ndarray, d: int) -> np.ndarray:
    """Theoretical eigenvalue decay λ_j ≍ j^{-d/(d-1)} (Lemma 3.1)."""
    return j ** (-d / (d - 1))


def minimax_optimal_rate(n: int, d: int) -> float:
    """
    Minimax optimal L2 convergence rate from Theorem 3.2:
    E[‖f̂ - f*‖²₂] = O(n^{-d/(2d-1)})
    """
    return n ** (-d / (2 * d - 1))


# ──────────────────────────────────────────────────────────────────────────
# 4. OPTIMAL REGULARIZATION PARAMETER (Theorem 3.2)
#
#    Choose μ ≍ n^{(d-1)/(2d-1)} for minimax optimality
# ──────────────────────────────────────────────────────────────────────────

def optimal_mu(n: int, d: int) -> float:
    """Theoretical optimal regularization parameter (Theorem 3.2)."""
    return n ** ((d - 1) / (2 * d - 1))


# ──────────────────────────────────────────────────────────────────────────
# 5. EFFECTIVE RANK of NTK matrix
#    (Measures how "spread" the eigenvalue distribution is)
# ──────────────────────────────────────────────────────────────────────────

def effective_rank(K: np.ndarray) -> float:
    """
    Effective rank = exp(entropy of normalized eigenvalues).
    High effective rank ↔ broader, more uniform spectrum.
    """
    eigs = np.linalg.eigvalsh(K)
    eigs_pos = np.abs(eigs) + 1e-12
    p = eigs_pos / eigs_pos.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-15))))


def condition_number(K: np.ndarray) -> float:
    """κ(K) = λ_max / λ_min. Well-conditioned ↔ easier inversion."""
    eigs = np.linalg.eigvalsh(K)
    return float(np.abs(eigs).max() / (np.abs(eigs).min() + 1e-12))


def spectral_gap(K: np.ndarray) -> float:
    """λ_1 - λ_2: gap between largest and second eigenvalue."""
    eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
    return float(eigs[0] - eigs[1]) if len(eigs) > 1 else 0.0


# ──────────────────────────────────────────────────────────────────────────
# 6. LOCAL RADEMACHER COMPLEXITY (from Theorem 4.2)
#
#    R̂_{H_∞}(ε) = [ (1/n) Σ min(λ̂_i/n, ε²) ]^{1/2}
#
#    Used to define optimal early stopping time k*
# ──────────────────────────────────────────────────────────────────────────

def local_rademacher_complexity(eigs: np.ndarray, epsilon: float, n: int) -> float:
    """
    Local empirical Rademacher complexity (Theorem 4.2, Eq. 4.2).
    eigs: eigenvalues of H_∞ (sorted descending)
    """
    return float(np.sqrt(np.mean(np.minimum(eigs / n, epsilon**2))))


def optimal_stopping_time(eigs: np.ndarray, sigma: float, eta: float, n: int) -> int:
    """
    Find k* = argmin_k { R̂(1/√(ηk)) > 1/(2eσηk) - 1 }
    This is the early stopping criterion from Theorem 4.2.
    """
    for k in range(1, 100000):
        eps = 1.0 / np.sqrt(eta * k)
        lhs = local_rademacher_complexity(eigs, eps, n)
        rhs = 1.0 / (2 * np.e * sigma * eta * k) - 1
        if lhs > rhs:
            return k
    return 100000  # fallback


# ──────────────────────────────────────────────────────────────────────────
# 7. L2-REGULARIZED GD CONVERGENCE (Theorem 5.1)
#
#    ‖u_D(k) - H_∞(CμI + H_∞)^{-1}y‖₂ = O_P(√n · (1 - η₂μ)^k)
#
#    Required iteration: k ≍ (η₂μ)^{-1}  (up to log factors)
#    Regularization: μ ≍ n^{(d-1)/(2d-1)}
# ──────────────────────────────────────────────────────────────────────────

def convergence_bound_l2reg(k: int, n: int, eta2: float, mu: float) -> float:
    """
    Upper bound on ‖u_D(k) - KRR_solution‖₂ from Theorem 5.1.
    Returns the decay factor √n · (1 - η₂μ)^k.
    """
    return np.sqrt(n) * ((1 - eta2 * mu) ** k)


if __name__ == "__main__":
    print("NTK Theory Module — Key equations from Hu et al. (2021)")
    print()

    # Example: compute minimax rates for different dimensions
    print("Minimax optimal L2 rate n^{-d/(2d-1)} for n=1000:")
    for d in [2, 5, 10, 20]:
        rate = minimax_optimal_rate(1000, d)
        mu_opt = optimal_mu(1000, d)
        print(f"  d={d:2d}: rate={rate:.4f}, optimal μ={mu_opt:.2f}")

    print()
    print("Theoretical eigenvalue decay λ_j ≍ j^{-d/(d-1)} (d=5):")
    d = 5
    for j in [1, 5, 10, 50, 100]:
        print(f"  j={j:3d}: λ_j ≍ {theoretical_eigenvalue_decay(np.array([j]), d)[0]:.4f}")
