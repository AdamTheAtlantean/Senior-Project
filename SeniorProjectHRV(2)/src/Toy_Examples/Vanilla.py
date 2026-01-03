import numpy as np

# Functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime_from_a(a):
    # if a = sigmoid(z), then sigmoid'(z) = a(1-a)
    return a * (1.0 - a)

def to_col(x):
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1)

# -----------------------------
# Two-layer (1 hidden layer) NN
# Uses squared error: L = 0.5 * ||y_hat - y||^2
# Output activation: sigmoid (so y_hat in (0,1))
# -----------------------------
def forward(x, W1, b1, W2, b2):
    """
    Shapes:
      x  : (d,1)
      W1 : (k,d)
      b1 : (k,1)
      W2 : (m,k)
      b2 : (m,1)
    Returns cache for backprop.
    """
    z1 = W1 @ x + b1          # (k,1)
    a1 = sigmoid(z1)          # (k,1)
    z2 = W2 @ a1 + b2         # (m,1)
    a2 = sigmoid(z2)          # (m,1)  == y_hat
    return {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}

def loss(y_hat, y):
    # 0.5 * sum((y_hat - y)^2)
    diff = (y_hat - y)
    return 0.5 * float((diff.T @ diff))

def backward(cache, y, W2):
    """
    Computes deltas and gradients.

    delta2 = dL/dz2 = (y_hat - y) ⊙ sigmoid'(z2)
    dW2 = delta2 @ a1^T
    db2 = delta2

    delta1 = (W2^T @ delta2) ⊙ sigmoid'(z1)
    dW1 = delta1 @ x^T
    db1 = delta1
    """
    x  = cache["x"]
    a1 = cache["a1"]
    a2 = cache["a2"]

    # Output delta
    delta2 = (a2 - y) * sigmoid_prime_from_a(a2)   # (m,1)

    # Gradients for output layer
    dW2 = delta2 @ a1.T                            # (m,k)
    db2 = delta2                                   # (m,1)

    # Hidden delta
    delta1 = (W2.T @ delta2) * sigmoid_prime_from_a(a1)  # (k,1)

    # Gradients for hidden layer
    dW1 = delta1 @ x.T                             # (k,d)
    db1 = delta1                                   # (k,1)

    grads = {"delta1": delta1, "delta2": delta2, "dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

def step_params(params, grads, lr):
    params["W1"] = params["W1"] - lr * grads["dW1"]
    params["b1"] = params["b1"] - lr * grads["db1"]
    params["W2"] = params["W2"] - lr * grads["dW2"]
    params["b2"] = params["b2"] - lr * grads["db2"]
    return params

def train_toy(x, y, params, lr=0.5, steps=1, verbose=True):
    """
    x, y: array-like; will be treated as column vectors.
    params: dict with W1,b1,W2,b2 (numpy arrays)
    """
    x = to_col(x)
    y = to_col(y)

    for t in range(1, steps + 1):
        cache = forward(x, params["W1"], params["b1"], params["W2"], params["b2"])
        L = loss(cache["a2"], y)
        grads = backward(cache, y, params["W2"])

        if verbose:
            print(f"\n--- Step {t} ---")
            print("x:\n", x)
            print("y:\n", y)
            print("z1:\n", cache["z1"])
            print("a1:\n", cache["a1"])
            print("z2:\n", cache["z2"])
            print("y_hat (a2):\n", cache["a2"])
            print("Loss:", L)
            print("delta2:\n", grads["delta2"])
            print("delta1:\n", grads["delta1"])
            print("dW2:\n", grads["dW2"])
            print("db2:\n", grads["db2"])
            print("dW1:\n", grads["dW1"])
            print("db1:\n", grads["db1"])

        params = step_params(params, grads, lr)

        if verbose:
            print("Updated W2:\n", params["W2"])
            print("Updated b2:\n", params["b2"])
            print("Updated W1:\n", params["W1"])
            print("Updated b1:\n", params["b1"])

    return params

# Initial Parameters
params = {
    "W1": np.array([[0.10, 0.20],
                    [0.30, 0.40]], dtype=float),
    "b1": np.array([[0.00],
                    [0.00]], dtype=float),
    # output has 1 neuron, hidden has 2 neurons => W2 shape (1,2)
    "W2": np.array([[0.50, 0.60]], dtype=float),
    "b2": np.array([[0.00]], dtype=float),
}

x = [1, 0]   # input
y = [1]      # target

# Run a single gradient descent update
params = train_toy(x, y, params, lr=0.5, steps=1, verbose=True)

# -----------------------------
# Notes for editing:
# - Change x and y for different samples/targets.
# - Change params["W1"], params["b1"], params["W2"], params["b2"] freely.
# - You can also set steps > 1 to watch learning iterate.
# - For multi-output, set y = [..] and make W2, b2 match output dimension m.
# -----------------------------
