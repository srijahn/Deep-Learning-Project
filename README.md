# NTK Spectral Analysis: Regularization Effects on Generalization

This project investigates how different regularization techniques influence **Neural Tangent Kernel (NTK)** eigenspectra and generalization behavior in overparameterized ReLU networks. The work is inspired by *Hu et al., AISTATS 2021*.

---

## 📌 Overview

We explore the relationship between:

* Regularization methods (L2, dropout, hybrid)
* NTK eigenspectrum evolution
* Generalization performance

The project combines **empirical experiments** with **theoretical baselines** to provide deeper insights into how regularization shapes learning dynamics.

---

## ✨ Features

* Empirical NTK computation via Jacobians
* Synthetic regression with analytical NTK Kernel Ridge Regression (KRR) baseline
* Spectral analysis before and after regularization
* Binary classification under additive label noise
* Hybrid regularization grid search
* Eigenvalue–error correlation study with statistical validation
* Training dynamics tracking using NTK snapshots

---

## 📁 Project Structure

```
.
├── ntk_experiment.py        # Main experiment runner
├── ntk_theory.py           # Theoretical utilities and equations
├── analysis_report.py      # Report generation and dashboard
├── results/
│   ├── data/               # Saved NumPy outputs
│   ├── figures/            # Per-experiment plots
│   └── report/             # Final consolidated outputs
└── requirements.txt        # Dependencies
```

---

## ⚙️ Setup

### 1. Create Virtual Environment (Windows PowerShell)

```powershell
python -m venv ntk_env
.\ntk_env\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Experiments

### Run All Tasks

```bash
python ntk_experiment.py --task all
```

### Run Individual Tasks

```bash
python ntk_experiment.py --task regression
python ntk_experiment.py --task spectral
python ntk_experiment.py --task mnist
python ntk_experiment.py --task hybrid
python ntk_experiment.py --task correlation
python ntk_experiment.py --task dynamics
```

### Optional: Extended MNIST Run

```bash
python ntk_experiment.py --task mnist --mnist-source torchvision --mnist-reps 100
```

---

## 📊 Generate Final Report

After completing experiments:

```bash
python analysis_report.py
```

### Outputs Generated

* Console summary tables (spectral + correlation metrics)
* Dashboard visualization:

  ```
  results/report/full_dashboard.png
  ```

---

## ⏱️ Typical Runtime

| Task Type     | Estimated Time |
| ------------- | -------------- |
| Single task   | 20–60 minutes  |
| Full pipeline | Few hours      |

*(Depends on CPU and dataset size)*

---

## 📚 Key Concepts

* Neural Tangent Kernel (NTK)
* Kernel Ridge Regression (KRR)
* Spectral bias and eigenspectrum shaping
* Regularization-driven generalization

---

## 📖 Reference

Hu et al., *AISTATS 2021* – Analysis of regularization effects in NTK regime.

---

## 🧠 Notes

* Designed for **CPU execution**, but can benefit from GPU acceleration for larger runs
* Results are reproducible via saved NumPy outputs
* Modular design allows easy extension for new regularization methods

---


