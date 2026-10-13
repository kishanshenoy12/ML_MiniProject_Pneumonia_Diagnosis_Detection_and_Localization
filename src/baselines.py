"""LR / SVM / RF baselines on flattened (downsampled) images."""
import argparse, cv2, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from dataset import get_splits

ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="cache.npz"); a = ap.parse_args()
c = dict(np.load(a.cache, allow_pickle=True))
X = np.stack([cv2.resize(i, (32, 32), interpolation=cv2.INTER_AREA).ravel() for i in c["X"]])
y = c["y"]; tr, va, te = get_splits(y)
sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
for name, m in [("LogReg", LogisticRegression(max_iter=1000)), ("SVM", SVC()),
                ("RandomForest", RandomForestClassifier(300, n_jobs=-1, random_state=0))]:
    m.fit(Xs[tr], y[tr])
    print(f"{name:13s} train {m.score(Xs[tr], y[tr]):.4f}  test {m.score(Xs[te], y[te]):.4f}")
