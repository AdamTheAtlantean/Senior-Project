"""
Vanilla RNN demo on HRV (RR intervals) from MIT-BIH + Jacobian eigenvalue analysis.

What this script does:
1) Extract RR intervals from WFDB annotations (MIT-BIH)
2) Normalize (z-score) the RR series
3) Train a *vanilla* RNN from scratch (NumPy) with explicit BPTT
4) Compute local Jacobians J_t = d h_{t+1} / d h_t along a trajectory
5) Compute eigenvalues of J_t and summarize spectral radius rho(J_t)
6) Plot RR–RR_{n-1} (Poincaré) plots: true HRV and true vs predicted

Requirements:
  pip install wfdb numpy matplotlib
"""

from __future__ import annotations
import numpy as np
import wfdb
import matplotlib.pyplot as plt


# 1) Data: MIT-BIH -> RR intervals ------------------------------------------------------

def load_rr_intervals(
    record_name: str = "100",
    db_dir: str = "data/mit-bih-arrhythmia-database-1.0.0",
    ann_extension: str = "atr",
    fs: float | None = None,
) -> np.ndarray:
    """
    Loads R-peak sample indices from MIT-BIH annotations and converts to RR intervals in seconds.
    """
    ann = wfdb.rdann(f"{db_dir}/{record_name}", extension=ann_extension)

    if fs is None:
        rec = wfdb.rdrecord(f"{db_dir}/{record_name}", sampto=1)
        fs = float(rec.fs)

    r_peaks = ann.sample.astype(np.int64)
    rr_samples = np.diff(r_peaks)
    rr_seconds = rr_samples / fs

    # Very light sanity filtering for demo purposes
    rr_seconds = rr_seconds[(rr_seconds > 0.3) & (rr_seconds < 2.0)]
    return rr_seconds


def zscore(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-12)
    return (x - mu) / sd, mu, sd


def make_sequences(series: np.ndarray, seq_len: int = 50):
    """
    Creates (X, Y) pairs:
      X[i]: series[i : i+seq_len]
      Y[i]: series[i+1 : i+seq_len+1]  (next-step targets)
    """
    X, Y = [], []
    for i in range(len(series) - seq_len - 1):
        X.append(series[i : i + seq_len])
        Y.append(series[i + 1 : i + seq_len + 1])
    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)


# NEW: RR–RR_{n-1} (Poincare) plots ------------------------------------------------------

def plot_rr_poincare(rr_seconds: np.ndarray, title: str = "RR–RR$_{n-1}$ (Poincaré) plot"):
    """
    Plots the classic HRV Poincaré plot: (RR_{n-1}, RR_n).
    """
    if len(rr_seconds) < 2:
        print("Not enough RR samples for Poincaré plot.")
        return

    x = rr_seconds[:-1]
    y = rr_seconds[1:]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=6, alpha=0.5)
    plt.xlabel(r"$RR_{n-1}$ (s)")
    plt.ylabel(r"$RR_n$ (s)")
    plt.title(title)
    plt.axis("equal")
    plt.show()


def plot_rr_poincare_comparison(
    rr_true: np.ndarray,
    rr_pred: np.ndarray,
    title: str = "RR–RR$_{n-1}$ plot: true vs predicted"
):
    """
    Overlays Poincaré plots for true RR and predicted RR.
    Both arrays should be in seconds, same length.
    """
    L = min(len(rr_true), len(rr_pred))
    if L < 2:
        print("Not enough samples for comparison Poincaré plot.")
        return

    rr_true = rr_true[:L]
    rr_pred = rr_pred[:L]

    plt.figure(figsize=(6, 6))
    plt.scatter(rr_true[:-1], rr_true[1:], s=10, alpha=0.45, label="True HRV")
    plt.scatter(rr_pred[:-1], rr_pred[1:], s=10, alpha=0.45, label="RNN prediction")
    plt.xlabel(r"$RR_{n-1}$ (s)")
    plt.ylabel(r"$RR_n$ (s)")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.show()


# 2) Vanilla RNN (NumPy) with explicit  ------------------------------------------------------

class VanillaRNN:
    """
    Vanilla RNN:
      a_t = W_hh h_{t-1} + W_xh x_t + b_h
      h_t = tanh(a_t)
      y_t = W_hy h_t + b_y

    Loss: MSE over a length-T window (one-step-ahead prediction).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)

        self.W_xh = 0.1 * rng.standard_normal((hidden_dim, input_dim))
        self.W_hh = 0.1 * rng.standard_normal((hidden_dim, hidden_dim))
        self.b_h  = np.zeros((hidden_dim, 1))

        self.W_hy = 0.1 * rng.standard_normal((output_dim, hidden_dim))
        self.b_y  = np.zeros((output_dim, 1))

        # grads
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h  = np.zeros_like(self.b_h)
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_y  = np.zeros_like(self.b_y)

    @staticmethod
    def tanh(x): return np.tanh(x)

    @staticmethod
    def dtanh_from_h(h):
        return 1.0 - h**2

    def forward(self, x_seq: np.ndarray, h0: np.ndarray | None = None):
        T, input_dim = x_seq.shape
        H = self.W_hh.shape[0]
        O = self.W_hy.shape[0]

        h_prev = np.zeros((H, 1)) if h0 is None else h0.copy()

        hs = np.zeros((T, H, 1))
        ys = np.zeros((T, O, 1))
        a_s = np.zeros((T, H, 1))

        for t in range(T):
            x_t = x_seq[t].reshape(input_dim, 1)

            a_t = self.W_hh @ h_prev + self.W_xh @ x_t + self.b_h
            h_t = self.tanh(a_t)
            y_t = self.W_hy @ h_t + self.b_y

            a_s[t] = a_t
            hs[t] = h_t
            ys[t] = y_t
            h_prev = h_t

        cache = {"x_seq": x_seq, "hs": hs, "ys": ys, "a_s": a_s}
        return ys, cache

    @staticmethod
    def loss_mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_true = y_true.reshape(-1, 1, 1)
        return float(np.mean((y_pred - y_true) ** 2))

    def zero_grads(self):
        self.dW_xh.fill(0.0)
        self.dW_hh.fill(0.0)
        self.db_h.fill(0.0)
        self.dW_hy.fill(0.0)
        self.db_y.fill(0.0)

    def bptt(self, cache, y_true: np.ndarray, clip: float = 1.0):
        x_seq = cache["x_seq"]
        hs = cache["hs"]
        ys = cache["ys"]

        T = x_seq.shape[0]
        input_dim = x_seq.shape[1]
        H = self.W_hh.shape[0]

        y_true = y_true.reshape(T, 1, 1)

        self.zero_grads()
        dh_next = np.zeros((H, 1))

        for t in reversed(range(T)):
            x_t = x_seq[t].reshape(input_dim, 1)
            h_t = hs[t]
            h_prev = hs[t - 1] if t > 0 else np.zeros_like(h_t)

            dy = (2.0 / T) * (ys[t] - y_true[t])

            self.dW_hy += dy @ h_t.T
            self.db_y  += dy

            dh = (self.W_hy.T @ dy) + dh_next
            da = dh * self.dtanh_from_h(h_t)

            self.dW_hh += da @ h_prev.T
            self.dW_xh += da @ x_t.T
            self.db_h  += da

            dh_next = self.W_hh.T @ da

        for g in [self.dW_xh, self.dW_hh, self.db_h, self.dW_hy, self.db_y]:
            np.clip(g, -clip, clip, out=g)

    def step(self, lr: float = 1e-3):
        self.W_xh -= lr * self.dW_xh
        self.W_hh -= lr * self.dW_hh
        self.b_h  -= lr * self.db_h
        self.W_hy -= lr * self.dW_hy
        self.b_y  -= lr * self.db_y

    # 3) Jacobian + eigenvalues utilities  ------------------------------------------------------

    def jacobian_hidden(self, h_t: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        a_t = self.W_hh @ h_t + self.W_xh @ x_t + self.b_h
        dtanh = (1.0 - np.tanh(a_t) ** 2).reshape(-1)
        D = np.diag(dtanh)
        J = D @ self.W_hh
        return J

    def spectral_radius_and_eigs(self, J: np.ndarray):
        eigvals = np.linalg.eigvals(J)
        rho = float(np.max(np.abs(eigvals)))
        return rho, eigvals


# 4) Training + Jacobian eigenvalue analysis + plots ---------------------------------------

def main():
    # --- Load HRV (RR) from MIT-BIH ---
    rr = load_rr_intervals(record_name="100", db_dir="data/mit-bih-arrhythmia-database-1.0.0")

    # Plot Poincare plot for the raw HRV data (baseline geometry)
    plot_rr_poincare(rr, title="RR–RR$_{n-1}$ (Poincaré) plot: MIT-BIH HRV (raw RR intervals)")

    rr_z, mu, sd = zscore(rr)

    # --- Make sequences ---
    seq_len = 50
    X, Y = make_sequences(rr_z, seq_len=seq_len)

    N = min(2000, len(X))
    X, Y = X[:N], Y[:N]

    input_dim = 1
    hidden_dim = 16
    output_dim = 1

    rnn = VanillaRNN(input_dim, hidden_dim, output_dim, seed=0)

    lr = 1e-3
    epochs = 10

    losses = []
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for i in range(len(X)):
            x_seq = X[i].reshape(seq_len, 1)
            y_true = Y[i].reshape(seq_len)

            y_pred, cache = rnn.forward(x_seq)
            loss = rnn.loss_mse(y_pred, y_true)

            rnn.bptt(cache, y_true, clip=1.0)
            rnn.step(lr=lr)

            epoch_loss += loss

        epoch_loss /= len(X)
        losses.append(epoch_loss)
        print(f"Epoch {ep:02d} | MSE loss = {epoch_loss:.6f}")

    # --- Plot training loss ---
    plt.figure()
    plt.plot(losses)
    plt.title("Training loss (MSE) per epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    # --- Qualitative one step prediction on one window ---
    test_x = X[-1].reshape(seq_len, 1)
    test_y = Y[-1].reshape(seq_len)

    pred_z, cache = rnn.forward(test_x)
    pred_z = pred_z.reshape(seq_len)

    pred_rr = pred_z * sd + mu
    true_rr = test_y * sd + mu

    plt.figure()
    plt.plot(true_rr, label="true RR (sec)")
    plt.plot(pred_rr, label="pred RR (sec)")
    plt.title("One-step-ahead prediction (qualitative)")
    plt.xlabel("t within window")
    plt.ylabel("RR interval (sec)")
    plt.legend()
    plt.show()

    # Poincare plot comparison of true vs predicted (learned geometry)
    plot_rr_poincare_comparison(true_rr, pred_rr, title="RR–RR$_{n-1}$ plot: true vs predicted (one window)")

    # --- Jacobian / eigenvalue analysis along the trajectory ---
    h = np.zeros((hidden_dim, 1))

    rhos = []
    eigvals_list = []

    for t in range(seq_len):
        x_t = test_x[t].reshape(input_dim, 1)

        J = rnn.jacobian_hidden(h, x_t)
        rho, eigvals = rnn.spectral_radius_and_eigs(J)

        rhos.append(rho)
        eigvals_list.append(eigvals)

        a = rnn.W_hh @ h + rnn.W_xh @ x_t + rnn.b_h
        h = np.tanh(a)

    rhos = np.array(rhos, dtype=np.float64)

    print("\n--- Jacobian / eigenvalue summary along one HRV window ---")
    print(f"hidden_dim n = {hidden_dim}")
    print(f"mean spectral radius  rho(J_t) = {np.mean(rhos):.4f}")
    print(f"max  spectral radius  rho(J_t) = {np.max(rhos):.4f}")
    print(f"min  spectral radius  rho(J_t) = {np.min(rhos):.4f}")

    mid = seq_len // 2
    mid_eigs = eigvals_list[mid]
    mid_eigs_sorted = mid_eigs[np.argsort(-np.abs(mid_eigs))]

    print(f"\nExample eigenvalues at t = {mid} (sorted by |lambda|):")
    for i in range(min(10, len(mid_eigs_sorted))):
        lam = mid_eigs_sorted[i]
        print(f"  lambda_{i+1} = {lam.real:+.4f}{lam.imag:+.4f}j   |lambda|={abs(lam):.4f}")

    plt.figure()
    plt.plot(rhos)
    plt.axhline(1.0, linestyle="--")
    plt.title("Spectral radius of local Jacobian along trajectory")
    plt.xlabel("t within window")
    plt.ylabel(r"$\rho(J_t)$")
    plt.show()


if __name__ == "__main__":
    main()
