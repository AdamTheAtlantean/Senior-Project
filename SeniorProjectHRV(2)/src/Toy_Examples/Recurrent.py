import numpy as np

# ----------------------------
# Activations
# ----------------------------
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime_from_yhat(yhat: float) -> float:
    # derivative of sigmoid(z) using yhat = sigmoid(z)
    return yhat * (1.0 - yhat)

def tanh_prime_from_h(h: np.ndarray) -> np.ndarray:
    # derivative of tanh(z) using h = tanh(z)
    return 1.0 - h**2

# ----------------------------
# Forward + BPTT for 2-step toy RNN
# ----------------------------
def forward_backward_2step(
    x1: np.ndarray, x2: np.ndarray, y: float,
    W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray,
    W_hy: np.ndarray, b_y: float,
    h0: np.ndarray
):
    """
    Shapes:
      x1, x2, h0, b_h: (H,1) or (D,1) as appropriate
      W_xh: (H,D)
      W_hh: (H,H)
      W_hy: (H,1)
      b_y: scalar
    Returns:
      cache dict with forward values
      grads dict with dW_xh, dW_hh, db_h, dW_hy, db_y
    """

    # ---- Forward ----
    z1 = W_xh @ x1 + W_hh @ h0 + b_h
    h1 = np.tanh(z1)

    z2 = W_xh @ x2 + W_hh @ h1 + b_h
    h2 = np.tanh(z2)

    z_out = float(W_hy.T @ h2 + b_y)
    yhat = sigmoid(z_out)

    L = 0.5 * (yhat - y)**2

    # ---- Backward (BPTT) ----
    # Output delta
    dL_dyhat = (yhat - y)
    dyhat_dzout = sigmoid_prime_from_yhat(yhat)
    delta_out = dL_dyhat * dyhat_dzout  # scalar = dL/dz_out

    # Gradients for output layer
    dW_hy = delta_out * h2              # (H,1)
    db_y  = delta_out                   # scalar

    # Backprop into h2
    dL_dh2 = W_hy * delta_out           # (H,1)

    # delta2 = dL/dz2
    delta2 = dL_dh2 * tanh_prime_from_h(h2)  # (H,1)

    # Backprop to h1 through recurrent linear map
    dL_dh1 = W_hh.T @ delta2            # (H,1)

    # delta1 = dL/dz1
    delta1 = dL_dh1 * tanh_prime_from_h(h1)  # (H,1)

    # Parameter gradients (shared across time)
    dW_xh = delta1 @ x1.T + delta2 @ x2.T     # (H,D)
    dW_hh = delta1 @ h0.T + delta2 @ h1.T     # (H,H)  (h0 term vanishes if h0=0)
    db_h  = delta1 + delta2                   # (H,1)

    cache = {
        "z1": z1, "h1": h1,
        "z2": z2, "h2": h2,
        "z_out": z_out, "yhat": yhat, "L": L,
        "delta_out": delta_out, "delta2": delta2, "delta1": delta1
    }
    grads = {
        "dW_xh": dW_xh, "dW_hh": dW_hh, "db_h": db_h,
        "dW_hy": dW_hy, "db_y": db_y
    }
    return cache, grads

def gd_step(params: dict, grads: dict, lr: float):
    # in-place update
    params["W_xh"] -= lr * grads["dW_xh"]
    params["W_hh"] -= lr * grads["dW_hh"]
    params["b_h"]  -= lr * grads["db_h"]
    params["W_hy"] -= lr * grads["dW_hy"]
    params["b_y"]  -= lr * grads["db_y"]

# ----------------------------
# Demo: our toy setup (editable)
# ----------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # Dimensions
    D, H = 2, 2

    # Inputs (change these)
    x1 = np.array([[1.0], [2.0]])
    x2 = np.array([[2.0], [1.0]])
    y  = 1.0

    # Initial hidden state
    h0 = np.zeros((H, 1))

    # Parameters (start with your earlier numbers)
    params = {
        "W_xh": np.array([[0.10, 0.20],
                          [0.30, 0.40]]),
        "W_hh": np.array([[0.10, 0.20],
                          [0.30, 0.40]]),
        "b_h":  np.zeros((H, 1)),
        "W_hy": np.array([[0.50],
                          [0.60]]),
        "b_y":  0.0
    }

    lr = 0.5
    steps = 1  # set >1 if you want to iterate

    for s in range(steps):
        cache, grads = forward_backward_2step(
            x1, x2, y,
            params["W_xh"], params["W_hh"], params["b_h"],
            params["W_hy"], params["b_y"],
            h0
        )

        print(f"\nStep {s}")
        print("yhat:", cache["yhat"], "Loss:", cache["L"])
        print("delta_out:", cache["delta_out"])
        print("delta2:\n", cache["delta2"])
        print("delta1:\n", cache["delta1"])

        print("\nGradients:")
        for k, v in grads.items():
            print(k, "=\n", v)

        gd_step(params, grads, lr)

        print("\nUpdated params:")
        for k, v in params.items():
            print(k, "=\n", v)
