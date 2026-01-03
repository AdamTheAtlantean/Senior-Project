# Recurrent Neural Networks as Nonlinear Dynamical Systems  
### A Study in ECG-Derived Heart Rate Variability

This repository accompanies a senior undergraduate project exploring **recurrent neural networks (RNNs) through the lens of discrete-time nonlinear dynamical systems**. Rather than treating neural networks as black-box predictors, the project emphasizes **interpretability, stability, and state-space behavior**, using tools from classical dynamical systems theory.

A minimal *vanilla* recurrent neural network is constructed explicitly from first principles and applied to **heart rate variability (HRV)** signals derived from electrocardiographic (ECG) recordings in the **MIT-BIH Arrhythmia Database**. The primary goal is not clinical prediction or architectural optimization, but conceptual understanding of how recurrence, nonlinearity, and learning interact to produce stable temporal dynamics.

---

## Project Overview

The project develops a unified interpretation of neural networks by:

- Viewing feedforward and recurrent networks as **iterated maps on state space**
- Interpreting learning as a **dynamical process in parameter space**
- Analyzing **local stability** via Jacobians and eigenvalues
- Connecting **spectral radius** to memory, stability, and gradient behavior
- Applying these ideas to a real biological time series (HRV)

The emphasis throughout is **foundational** rather than performance-driven.

---

## Repository Structure

```text
SeniorProjectHRV/
├── README.md
├── requirements.txt
├── src/
│   └── RNNv2.py
├── figures/
│   ├── LossPerEpoch.png
│   ├── RRnRRn-1.png
│   ├── SpectralRadius.png
│   ├── PQRST.png
│   └── ...
├── report/
│   └── SeniorProjectHRV.tex
└── data/
    └── README.md
