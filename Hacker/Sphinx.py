import numpy as np

# --------------------------
# Utility functions
# --------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def per_example_grads_logreg(X, y, W, b):
    """
    Per-example gradients for binary logistic regression.
    X: (m, d), y: (m,), W: (d,), b: scalar
    Returns:
      grads_W: (m, d)
      grads_b: (m,)
    """
    m, d = X.shape
    logits = X @ W + b
    probs = sigmoid(logits)
    err = (probs - y)  # (m,)
    grads_W = err[:, None] * X    # (m, d)
    grads_b = err                 # (m,)
    return grads_W, grads_b

def clip_per_example(grads, C):
    """
    L2 clip per-example grads to norm C.
    grads: (m, d)
    """
    norms = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-12
    factors = np.minimum(1.0, C / norms)
    return grads * factors

def dp_aggregate(grads, noise_mult, C, batch_size, rng):
    """
    Average clipped grads + Gaussian noise.
    grads: (m, d)
    noise std per coord = noise_mult * C / batch_size
    """
    m = grads.shape[0]
    assert m == batch_size
    g_bar = grads.mean(axis=0)
    sigma = noise_mult * C / batch_size
    noise = rng.normal(loc=0.0, scale=sigma, size=g_bar.shape)
    return g_bar + noise

# --------------------------
# Roles
# --------------------------
class Server:
    """
    Server keeps:
      - W in plaintext (DP-protected via client-side DP-SGD updates)
      - b_s = b + r (biased share; never sees b or r individually)
    """
    def __init__(self, d, lr=0.5, seed=0):
        self.W = np.zeros(d)  # start at zeros
        self.b_share = 0.0    # b_s = b + r
        self.lr = lr
        self.rng = np.random.default_rng(seed)

    def get_W(self):
        # In Sphinx, W is plaintext; sending W to client is fine.
        return self.W.copy()

    def update_from_client(self, noisy_grad_W, new_b_share):
        # Gradient descent on W using noisy grads
        self.W -= self.lr * noisy_grad_W
        # Refresh server bias share
        self.b_share = new_b_share

    # Private inference primitive (server side)
    def infer_on_masked(self, x_masked):
        # Compute masked logit = (x + m)·W + (b + r)
        return float(x_masked @ self.W + self.b_share)

class Client:
    """
    Client keeps:
      - Local data (X, y)
      - Bias mask r (client share)
      - Current (private) bias b
      - DP parameters for W updates (clip C, noise multiplier)
    """
    def __init__(self, X, y, C=1.0, noise_mult=1.0, lr_b=0.5, seed=1234):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.C = C
        self.noise_mult = noise_mult
        self.lr_b = lr_b
        self.rng = np.random.default_rng(seed)

        # initialize bias and its mask r; server share b_s = b + r stored on server
        self.b = 0.0
        self.r = float(self.rng.normal())

    def train_round(self, server: Server, batch_size=32):
        """
        One "online" round:
          1) Pull current W (plaintext) from server.
          2) Sample a minibatch.
          3) Compute per-example grads.
          4) DP-SGD aggregate for W (clip + noise); send to server.
          5) Update b locally (no DP here, but we keep b masked).
          6) Reshare bias: pick new r', send b_s' = b + r' to server.
        """
        # 1) pull W
        W = server.get_W()

        # 2) minibatch
        idx = self.rng.choice(self.n, size=batch_size, replace=False)
        Xb = self.X[idx]
        yb = self.y[idx]

        # 3) per-example grads
        gW, gb = per_example_grads_logreg(Xb, yb, W, self.b)

        # 4) DP aggregate for W
        gW_clipped = clip_per_example(gW, self.C)
        noisy_gW = dp_aggregate(gW_clipped, self.noise_mult, self.C, batch_size, self.rng)

        # 5) update b (plain GD on mean grad_b)
        grad_b = gb.mean()
        self.b -= self.lr_b * grad_b

        # 6) refresh mask for bias share
        new_r = float(self.rng.normal())
        new_b_share = self.b + new_r
        self.r = new_r  # keep only the new client share

        # send updates to server
        server.update_from_client(noisy_gW, new_b_share)

    # Client-side reconstructs the final logit privately
    def private_predict(self, server: Server, x):
        """
        Private inference:
          - Client masks x' = x + m and sends x' to server.
          - Server returns z' = (x+m)·W + (b+r).
          - Client removes masks: z = z' - (m·W) - r.
          - Prediction = sigmoid(z).
        """
        m = self.rng.normal(size=x.shape)      # input mask (vector)
        x_masked = x + m
        z_masked = server.infer_on_masked(x_masked)  # scalar
        # Remove masks:
        W = server.get_W()
        z = z_masked - float(m @ W) - self.r
        return sigmoid(z)

# --------------------------
# Toy data & demo harness
# --------------------------
def make_toy_data(n=400, d=5, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_W = rng.normal(size=d)
    true_b = rng.normal()
    logits = X @ true_W + true_b
    p = sigmoid(logits)
    y = (rng.uniform(size=n) < p).astype(float)
    return X, y

def accuracy(client: Client, server: Server, Xtest, ytest, limit=None):
    if limit is None:
        limit = len(Xtest)
    preds = []
    for i in range(limit):
        preds.append(client.private_predict(server, Xtest[i]))
    preds = np.array(preds)
    yhat = (preds >= 0.5).astype(float)
    return (yhat == ytest[:limit]).mean()

def demo():
    # data
    X, y = make_toy_data(n=500, d=8, seed=0)
    Xtr, ytr = X[:400], y[:400]
    Xte, yte = X[400:], y[400:]

    # roles
    server = Server(d=X.shape[1], lr=0.5, seed=0)
    client = Client(Xtr, ytr, C=1.0, noise_mult=0.8, lr_b=0.5, seed=42)

    # Initialize server bias share to match client's current b+r
    server.b_share = client.b + client.r

    # Training experiments (several online rounds)
    epochs = 8
    rounds_per_epoch = 20
    batch = 64

    print("Starting private online training...")
    for ep in range(1, epochs + 1):
        for _ in range(rounds_per_epoch):
            client.train_round(server, batch_size=batch)
        acc = accuracy(client, server, Xte, yte)
        print(f"Epoch {ep:02d} | Test acc (private inference): {acc:.3f}")

    # Quick privacy checks (what does the server see?)
    #    - Server never sees raw X or y.
    #    - Server sees only noisy grads for W and masked bias share b_s.
    #    - Private inference shows server only x_masked and returns one scalar.
    print("\nServer snapshot (what it holds):")
    print("W (plaintext) shape:", server.W.shape)
    print("b_share (masked bias):", server.b_share)
    print("\nClient snapshot:")
    print("Client bias mask r:", client.r)
    print("Client local bias b:", client.b)

if __name__ == "__main__":
    demo()
