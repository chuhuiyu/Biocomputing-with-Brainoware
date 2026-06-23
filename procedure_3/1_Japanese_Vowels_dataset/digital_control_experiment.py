import io, time, csv, random
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn

torch.set_num_threads(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse(path):
    text = Path(path).read_text(errors="ignore").strip()
    seq = []
    for b in [x for x in text.split("\n\n") if x.strip()]:
        arr = np.loadtxt(io.StringIO(b), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        seq.append(arr)
    return seq


X = parse("./ae.train")[:240]
y = np.repeat(np.arange(8), 30)


def std(Xtr, Xte):
    F = np.vstack(Xtr)
    m = F.mean(0, keepdims=True)
    s = F.std(0, keepdims=True)
    s[s < 1e-6] = 1
    return [(x - m) / s for x in Xtr], [(x - m) / s for x in Xte]


def pad(X, max_len=29):
    n = len(X)
    d = X[0].shape[1]
    out = np.zeros((n, max_len, d), dtype=np.float32)
    L = np.zeros(n, dtype=np.int64)
    for i, seq in enumerate(X):
        l = min(len(seq), max_len)
        out[i, :l] = seq[:l]
        L[i] = l
    return out, L


def flat(Xp, L):
    return np.c_[Xp.reshape(len(Xp), -1), L[:, None] / Xp.shape[1]].astype(
        np.float32
    )


def ridge(Xtr, ytr, Xte, alpha=1.0, C=8):
    Xb = np.c_[np.ones((Xtr.shape[0], 1)), Xtr]
    Xtb = np.c_[np.ones((Xte.shape[0], 1)), Xte]
    Y = np.eye(C)[ytr]
    R = alpha * np.eye(Xb.shape[1])
    R[0, 0] = 0
    W = np.linalg.solve(Xb.T @ Xb + R, Xb.T @ Y)
    return (Xtb @ W).argmax(1)


class MLP(nn.Module):
    def __init__(self, d, c=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(24, c),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, Xte, seed, epochs=120):
    set_seed(seed)
    model = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr)
    yt = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xte)
    model.train()
    for e in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(Xv).argmax(1).numpy()


def make_res(n=200, d=12, seed=0, sr=0.9, density=0.1, input_scale=0.5):
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-input_scale, input_scale, (n, d + 1)).astype(np.float32)
    W = rng.normal(0, 1, (n, n)).astype(np.float32) * (
        rng.random((n, n)) < density
    ).astype(np.float32)
    vals = np.linalg.eigvals(W)
    rad = float(np.max(np.abs(vals)))
    if rad > 1e-6:
        W *= sr / rad
    return Win, W


def esn_feats(Xp, L, Win, W, leak=0.3):
    n = Win.shape[0]
    feats = []
    for seq, l in zip(Xp, L):
        st = np.zeros(n, dtype=np.float32)
        states = []
        for t in range(int(l)):
            u = np.r_[1.0, seq[t]].astype(np.float32)
            ns = np.tanh(Win @ u + W @ st).astype(np.float32)
            st = (1 - leak) * st + leak * ns
            states.append(st.copy())
        S = np.vstack(states)
        feats.append(np.r_[st, S.mean(0), S.std(0)].astype(np.float32))
    return np.vstack(feats)


rows = []
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
for run, (tri, tei) in enumerate(sss.split(np.arange(len(y)), y)):
    t = time.time()
    seed = 42 + run
    Xtr = [X[i] for i in tri]
    Xte = [X[i] for i in tei]
    ytr = y[tri]
    yte = y[tei]
    Xtr, Xte = std(Xtr, Xte)
    Xtrp, Ltr = pad(Xtr)
    Xtep, Lte = pad(Xte)
    Xtrf = flat(Xtrp, Ltr)
    Xtef = flat(Xtep, Lte)
    acc_ann = accuracy_score(yte, train_mlp(Xtrf, ytr, Xtef, seed, epochs=50))
    Win, W = make_res(seed=seed, n=24)
    Ftr = esn_feats(Xtrp, Ltr, Win, W)
    Fte = esn_feats(Xtep, Lte, Win, W)
    acc_esn = accuracy_score(yte, ridge(Ftr, ytr, Fte, alpha=50.0))
    rows.append({"ANN": acc_ann, "ESN": acc_esn})
    print(
        run,
        "ANN",
        round(acc_ann, 4),
        "ESN",
        round(acc_esn, 4),
        "time",
        round(time.time() - t, 2),
        flush=True,
    )

with open(
    "./japanese_vowels_control_accuracy.csv",
    "w",
    newline="",
) as f:
    w = csv.DictWriter(f, fieldnames=["ANN", "ESN"])
    w.writeheader()
    w.writerows(rows)
