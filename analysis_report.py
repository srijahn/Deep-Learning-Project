"""
NTK Project — Final Analysis & Report Figures
==============================================
Run AFTER ntk_experiment.py to produce polished report-quality figures
and summary statistics tables.

Usage: python analysis_report.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs("results/report", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def load_results():
    """Load all saved experimental results."""
    results = {}
    files = {
        "synthetic": "results/data/synthetic_regression.npy",
        "spectral":  "results/data/spectral_analysis.npy",
        "mnist":     "results/data/mnist_results.npy",
        "hybrid":    "results/data/hybrid_grid.npy",
        "dynamics":  "results/data/training_dynamics.npy",
        "correlation": "results/data/eigenvalue_error_correlation.npy",
    }
    for name, path in files.items():
        if os.path.exists(path):
            results[name] = np.load(path, allow_pickle=True).item()
            print(f"✓ Loaded {name} from {path}")
        else:
            print(f"✗ Missing {path} — run ntk_experiment.py first")
    return results


def table_correlation_summary(corr_data: dict):
    """Print Exp 5 correlation statistics for report inclusion."""
    print("\n" + "="*60)
    print("TABLE: NTK Eigenvalue-Error Correlation Summary")
    print("="*60)
    print(
        "λ_min vs L2 (Pearson): "
        f"r={corr_data['pearson_lmin_l2']:.4f}, "
        f"p={corr_data['pearson_lmin_l2_pvalue']:.6f}"
    )
    print(
        "κ vs L2 (Spearman): "
        f"rho={corr_data['spearman_cond_l2']:.4f}, "
        f"p={corr_data['spearman_cond_l2_pvalue']:.6f}"
    )


def table_spectral_summary(spectral_data: dict):
    """Print LaTeX-ready table of spectral metrics."""
    print("\n" + "="*60)
    print("TABLE: NTK Spectral Summary (for report)")
    print("="*60)
    print(f"{'Setting':<20} {'λ_max(before)':>14} {'λ_min(before)':>14} "
          f"{'κ(before)':>10} {'λ_max(after)':>13} {'λ_min(after)':>13} {'κ(after)':>10}")
    print("-"*95)
    for s_name, d in spectral_data.items():
        eb = d["eigs_before"]
        ea = d["eigs_after"]
        kb = eb[0] / (abs(eb[-1]) + 1e-12)
        ka = ea[0] / (abs(ea[-1]) + 1e-12)
        print(f"{s_name:<20} {eb[0]:>14.4f} {eb[-1]:>14.4f} {kb:>10.1f} "
              f"{ea[0]:>13.4f} {ea[-1]:>13.4f} {ka:>10.1f}")

    # LaTeX version
    print("\nLaTeX table:")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\hline")
    print(r"Setting & $\lambda_{max}^\text{pre}$ & $\lambda_{min}^\text{pre}$ & "
          r"$\kappa^\text{pre}$ & $\lambda_{max}^\text{post}$ & "
          r"$\lambda_{min}^\text{post}$ & $\kappa^\text{post}$ \\")
    print(r"\hline")
    for s_name, d in spectral_data.items():
        eb = d["eigs_before"]
        ea = d["eigs_after"]
        kb = eb[0] / (abs(eb[-1]) + 1e-12)
        ka = ea[0] / (abs(ea[-1]) + 1e-12)
        print(f"{s_name} & {eb[0]:.3f} & {eb[-1]:.3f} & {kb:.1f} & "
              f"{ea[0]:.3f} & {ea[-1]:.3f} & {ka:.1f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


def figure_summary_dashboard(results: dict):
    """
    Create a single comprehensive dashboard figure for the report.
    """
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"No-reg":"#e74c3c","WeightDecay":"#3498db",
              "Dropout":"#2ecc71","Hybrid":"#9b59b6","KRR+NTK":"#f39c12"}

    # ── Panel A: Synthetic regression L2 error vs noise ──
    if "synthetic" in results:
        ax = fig.add_subplot(gs[0, :2])
        data = results["synthetic"]
        fn_data = data.get("f2_quadratic", data.get(list(data.keys())[0]))
        for s_name, series in fn_data.items():
            sigmas = [x[0] for x in series]
            means  = [x[1] for x in series]
            stds   = [x[2] for x in series]
            ax.plot(sigmas, means, label=s_name,
                    color=colors.get(s_name, "gray"), linewidth=2, marker="o")
            ax.fill_between(sigmas,
                            [m-s for m,s in zip(means,stds)],
                            [m+s for m,s in zip(means,stds)],
                            color=colors.get(s_name, "gray"), alpha=0.1)
        ax.set_title("(A) L2 Error vs. Label Noise — f*(x) = x^T x", fontweight="bold")
        ax.set_xlabel("Noise σ")
        ax.set_ylabel("L2 Estimation Error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Panel B: MNIST misclassification ──
    if "mnist" in results:
        ax = fig.add_subplot(gs[0, 2])
        for s_name, series in results["mnist"].items():
            sigmas = [x[0] for x in series]
            means  = [x[1] for x in series]
            ax.plot(sigmas, means, label=s_name,
                    color=colors.get(s_name, "gray"), linewidth=2, marker="s")
        ax.set_title("(B) MNIST Misclassification\nvs. Label Noise", fontweight="bold")
        ax.set_xlabel("Noise σ")
        ax.set_ylabel("Misclass. Rate (%)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Panel C & D: NTK eigenvalue spectra ──
    if "spectral" in results:
        sp = results["spectral"]
        settings_order = ["No-reg", "WeightDecay", "Dropout", "Hybrid"]
        palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
        sub_gs = gs[1, :].subgridspec(2, 2, hspace=0.35, wspace=0.30)

        for idx, (s_name, col) in enumerate(zip(settings_order, palette)):
            if s_name not in sp:
                continue
            ax = fig.add_subplot(sub_gs[idx // 2, idx % 2])
            eigs_b = sp[s_name]["eigs_before"]
            eigs_a = sp[s_name]["eigs_after"]
            ax.semilogy(range(1, len(eigs_b)+1), np.abs(eigs_b)+1e-12,
                        color=col, linewidth=1.5, linestyle="--", alpha=0.5, label="Before")
            ax.semilogy(range(1, len(eigs_a)+1), np.abs(eigs_a)+1e-12,
                        color=col, linewidth=2, label="After")
            ax.set_title(f"({chr(67 + idx)}) {s_name}", fontweight="bold")
            ax.set_xlabel("Eigenvalue Index")
            ax.set_ylabel("Eigenvalue")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    # ── Panel E: Hybrid grid heatmap ──
    if "hybrid" in results:
        import seaborn as sns
        ax = fig.add_subplot(gs[2, :2])
        hd = results["hybrid"]
        l2_grid = hd["l2"]
        wd_vals = hd["wd_values"]
        dp_vals = hd["dp_values"]
        sns.heatmap(l2_grid, ax=ax, annot=True, fmt=".3f",
                    xticklabels=[str(d) for d in dp_vals],
                    yticklabels=[str(w) for w in wd_vals],
                    cmap="YlOrRd", linewidths=0.5)
        ax.set_xlabel("Dropout Rate")
        ax.set_ylabel("Weight Decay")
        ax.set_title("(G) Hybrid Grid: Test L2 Error\n(Weight Decay × Dropout)", fontweight="bold")

    # ── Panel F: Training dynamics ──
    if "dynamics" in results:
        ax = fig.add_subplot(gs[2, 2])
        for s_name, dyn in results["dynamics"].items():
            ax.plot(dyn["epochs"], dyn["test_l2"],
                    label=s_name.replace(" (ONN)", "").replace("No-reg", "No-reg"),
                    color=colors.get(s_name.split(" ")[0], "gray"), linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test L2 Error")
        ax.set_title("(H) Test L2 Error over Training", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("NTK Spectral Analysis: Regularization Effects on Generalization\n"
                 "Based on Hu et al. (2021), AISTATS",
                 fontsize=15, fontweight="bold", y=1.01)

    plt.savefig("results/report/full_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved results/report/full_dashboard.png")


def eigenvalue_suppression_analysis(spectral_data: dict):
    """
    Quantify small-eigenvalue suppression per regularization method.
    Key insight: weight decay suppresses small eigenvalues;
    dropout introduces stochastic perturbations.
    """
    print("\n" + "="*60)
    print("EIGENVALUE SUPPRESSION ANALYSIS")
    print("="*60)
    threshold_frac = 0.01   # eigenvalues < 1% of λ_max are "small"

    for s_name, d in spectral_data.items():
        ea = d["eigs_after"]
        lmax = ea[0]
        small_before = np.sum(np.abs(d["eigs_before"]) < threshold_frac * np.abs(d["eigs_before"][0]))
        small_after  = np.sum(np.abs(ea) < threshold_frac * lmax)
        ratio = ea[0] / ea[len(ea)//4]   # ratio of top eigenvalue to 25th percentile

        print(f"\n{s_name}:")
        print(f"  Small eigenvalues (< 1% λ_max): {small_before} → {small_after}")
        print(f"  λ_1 / λ_{len(ea)//4}: {ratio:.2f}")
        print(f"  Condition number: {d['cond_before']:.1f} → {d['cond_after']:.1f}")
        print(f"  Interpretation:", end=" ")
        if "Hybrid" in s_name:
            print("Best conditioning: WD suppresses small λ, Dropout adds regularization noise")
        elif "WeightDecay" in s_name:
            print("L2 shifts spectrum — equivalent to KRR with μI shift (Theorem 5.1)")
        elif "Dropout" in s_name:
            print("Stochastic perturbation redistributes eigenvalue mass")
        else:
            print("No modification — small eigenvalues amplify noise → poor generalization")


if __name__ == "__main__":
    print("NTK Project — Report Analysis")
    print("="*60)

    results = load_results()

    if "spectral" in results:
        table_spectral_summary(results["spectral"])
        eigenvalue_suppression_analysis(results["spectral"])

    if "correlation" in results:
        table_correlation_summary(results["correlation"])

    figure_summary_dashboard(results)

    print("\n✅ Report analysis complete! Check results/report/")
