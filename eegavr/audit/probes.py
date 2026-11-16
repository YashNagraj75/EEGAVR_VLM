import warnings
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BAND = (0.45, 0.55)


def make_probes():
    return {"tree_d2": DecisionTreeClassifier(max_depth=2, random_state=0),
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
            "mlp32": make_pipeline(StandardScaler(), MLPClassifier((32,), alpha=1e-3, max_iter=400,
                                                                   random_state=0))}


def cluster_ci(correct, groups, level=0.90, n_boot=2000, seed=0):
    # percentile CI of accuracy, resampling whole groups (twin pairs / base scenes)
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=correct)
    c = np.bincount(inv)
    idx = rng.integers(0, len(ug), (n_boot, len(ug)))
    acc = s[idx].sum(1) / c[idx].sum(1)
    a = (1 - level) / 2
    return [float(np.quantile(acc, a)), float(np.quantile(acc, 1 - a))]


def run_probes(X, y, groups, n_splits=5):
    # out-of-fold accuracy of each probe with grouped CV; returns per-probe acc, CI and the best probe
    X = np.nan_to_num(np.asarray(X, float))
    out = {}
    for name, mk in make_probes().items():
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            m = make_probes()[name]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        corr = (pred == y).astype(float)
        out[name] = {"acc": float(corr.mean()), "ci90": cluster_ci(corr, groups)}
    best = max(out, key=lambda k: out[k]["acc"])
    ci = out[best]["ci90"]
    return {"probes": out, "best": best, "best_acc": out[best]["acc"], "best_ci90": ci,
            "pass": bool(BAND[0] <= ci[0] and ci[1] <= BAND[1])}


def run_univariate(X, y, groups, names):
    # one probe per single cue (depth-2 tree and logistic regression); pass = every cue's CI inside the band
    out = {}
    for j, n in enumerate(names):
        pr = {}
        for name in ("tree_d2", "logreg"):
            pred = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
                m = make_probes()[name]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(X[tr][:, [j]], y[tr])
                pred[te] = m.predict(X[te][:, [j]])
            corr = (pred == y).astype(float)
            pr[name] = {"acc": float(corr.mean()), "ci90": cluster_ci(corr, groups)}
        b = max(pr, key=lambda k: pr[k]["acc"])
        ci = pr[b]["ci90"]
        out[n] = {"best": b, "acc": pr[b]["acc"], "ci90": ci, "pass": bool(BAND[0] <= ci[0] and ci[1] <= BAND[1])}
    worst = max(out, key=lambda k: out[k]["acc"])
    return {"cues": out, "worst_cue": worst, "worst_acc": out[worst]["acc"], "worst_ci90": out[worst]["ci90"],
            "pass": all(v["pass"] for v in out.values())}
