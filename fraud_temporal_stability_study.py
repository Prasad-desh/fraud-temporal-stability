"""
Fraud Temporal Stability Study
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy.stats import ks_2samp, wilcoxon

SEED = 42
K = 15

def make_lr(): return LogisticRegression(solver="liblinear", max_iter=1000, random_state=SEED)
def make_rf(n=200): return RandomForestClassifier(n_estimators=n, max_depth=12, n_jobs=-1, random_state=SEED)
def make_xgb(): return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=SEED)

# ---------------- selectors: return selected feature names ----------------
def sel_mi(X, y, k=K):
    s = SelectKBest(lambda A, b: mutual_info_classif(A, b, random_state=SEED), k=k).fit(X, y)
    return list(X.columns[s.get_support()])

def sel_rfe(X, y, k=K, sample=40000, step=2, trees=100):
    if len(X) > sample:
        X, _, y, _ = train_test_split(X, y, train_size=sample, stratify=y, random_state=SEED)
    est = RandomForestClassifier(n_estimators=trees, max_depth=10, n_jobs=-1, random_state=SEED)
    s = RFE(est, n_features_to_select=k, step=step).fit(X, y)
    return list(X.columns[s.get_support()])

def sel_xgb(X, y, k=K):
    m = make_xgb().fit(X, y)
    imp = pd.Series(m.feature_importances_, index=X.columns)
    return list(imp.nlargest(k).index)

SELECTORS = {"MI (filter)": sel_mi, "RFE (wrapper)": sel_rfe, "XGB (embedded)": sel_xgb}
MODELS = {"LogReg": make_lr, "RandomForest": make_rf, "XGBoost": make_xgb}
PAIRS = {"LogReg": "MI (filter)", "RandomForest": "RFE (wrapper)", "XGBoost": "XGB (embedded)"}

# ---------------- stability ----------------
def kuncheva(sets, n_total):
    """Kuncheva index for equal-size subsets: 0 = chance, 1 = identical."""
    k = len(sets[0]); vals = []
    exp = k * k / n_total
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            r = len(set(sets[i]) & set(sets[j]))
            vals.append((r - exp) / (k - exp))
    return float(np.mean(vals)) if vals else np.nan

def jaccard(sets):
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = set(sets[i]), set(sets[j])
            vals.append(len(a & b) / len(a | b))
    return float(np.mean(vals)) if vals else np.nan

# ---------------- cost model ----------------
def cost_curve(y_true, prob, amounts, review_cost):
    """Total monetary cost over a grid of thresholds.
    FN costs the transaction amount (fraud goes uncharged-back); every alert costs review_cost."""
    order = np.argsort(prob)[::-1]
    p, a, yt = prob[order], amounts[order], np.asarray(y_true)[order]
    total_fraud = a[yt == 1].sum()
    # flagging the top-i transactions
    caught = np.cumsum(np.where(yt == 1, a, 0.0))
    alerts = np.arange(1, len(p) + 1)
    costs = (total_fraud - caught) + review_cost * alerts
    costs = np.insert(costs, 0, total_fraud)         # flag nothing
    thr = np.insert(p, 0, 1.01)
    return thr, costs, total_fraud

def best_threshold(y_true, prob, amounts, review_cost):
    thr, costs, base = cost_curve(y_true, prob, amounts, review_cost)
    i = int(np.argmin(costs))
    return float(thr[i]), float(costs[i]), float(base)

def eval_at(y_true, prob, thr, amounts, review_cost):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    amt = np.asarray(amounts); yt = np.asarray(y_true)
    missed = amt[(yt == 1) & (pred == 0)].sum()
    cost = missed + review_cost * (tp + fp)
    base = amt[yt == 1].sum()
    return dict(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                f1=2 * tp / max(2 * tp + fp + fn, 1),
                pr_auc=average_precision_score(y_true, prob),
                roc_auc=roc_auc_score(y_true, prob),
                cost=cost, cost_baseline=base, savings_pct=100 * (base - cost) / base)

# ---------------- one train/eval ----------------
def fit_eval(model_name, X_tr, y_tr, X_te, y_te, feats, amounts_te, review_cost, thr=None):
    Xb, yb = SMOTE(random_state=SEED).fit_resample(X_tr[feats], y_tr)
    t0 = time.time(); m = MODELS[model_name]().fit(Xb, yb); fit_s = time.time() - t0
    prob = m.predict_proba(X_te[feats])[:, 1]
    use_thr = thr if thr is not None else 0.5
    out = eval_at(y_te, prob, use_thr, amounts_te, review_cost)
    out.update(model=model_name, n_features=len(feats), fit_s=round(fit_s, 2), threshold=use_thr)
    return out, prob


# ------------------------------------------------------------------ experiment driver
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, wilcoxon

def prepare(df):
    day = (df["Time"] >= 86400).astype(int)           # 0 = day 1, 1 = day 2
    y = df["Class"].astype(int)
    amounts = df["Amount"].to_numpy(float)            # raw amounts for the cost model
    X = df.drop(columns=["Time", "Class"])
    return X, y, day, amounts

def scale(X_tr, X_te):
    sc = StandardScaler().fit(X_tr[["Amount"]])
    X_tr, X_te = X_tr.copy(), X_te.copy()
    X_tr["Amount"] = sc.transform(X_tr[["Amount"]]).ravel()
    X_te["Amount"] = sc.transform(X_te[["Amount"]]).ravel()
    return X_tr, X_te

def print_progress(label, completed, total, started):
    width = 30
    ratio = completed / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - started
    print(f"\r  {label}: [{bar}] {completed}/{total} ({ratio:.0%}) "
          f"elapsed {elapsed / 60:.1f} min", end="", flush=True)
    if completed == total:
        print()

def drift_report(X, y, day):
    """KS statistic per feature, day 1 vs day 2 legitimate traffic."""
    rows = []
    for f in X.columns:
        a = X.loc[(day == 0) & (y == 0), f]; b = X.loc[(day == 1) & (y == 0), f]
        st, p = ks_2samp(a, b)
        rows.append(dict(feature=f, ks=st, p=p))
    return pd.DataFrame(rows).sort_values("ks", ascending=False)

def cv_experiment(X, y, amounts, n_splits, n_repeats, review_cost, verbose=True):
    """Random stratified protocol: per-fold metrics + per-fold selected sets."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    rows, chosen = [], {s: [] for s in SELECTORS}
    total_folds = n_splits * n_repeats
    started = time.time()
    for k, (tr, te) in enumerate(cv.split(X, y)):
        X_tr, X_te = scale(X.iloc[tr], X.iloc[te])
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        amt_te = amounts[te]
        feats = {}
        for sname, fn in SELECTORS.items():
            t0 = time.time(); f = fn(X_tr, y_tr); sel_s = time.time() - t0
            feats[sname] = f; chosen[sname].append(f)
            rows.append(dict(fold=k, kind="selection", selector=sname, sel_s=round(sel_s, 2)))
        for mname in MODELS:
            for cond, fs in [("all", list(X.columns)), ("selected", feats[PAIRS[mname]])]:
                out, _ = fit_eval(mname, X_tr, y_tr, X_te, y_te, fs, amt_te, review_cost)
                out.update(fold=k, kind="eval", condition=cond, selector=PAIRS[mname] if cond == "selected" else "-")
                rows.append(out)
        if verbose:
            print_progress("CV folds", k + 1, total_folds, started)
    return pd.DataFrame(rows), chosen

def temporal_experiment(X, y, day, amounts, review_cost):
    """Train on day 1, test on day 2. Threshold tuned on a held-out slice of day 1."""
    tr_idx = np.where(day == 0)[0]; te_idx = np.where(day == 1)[0]
    X_tr_all, X_te = scale(X.iloc[tr_idx], X.iloc[te_idx])
    y_tr_all, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    amt_te = amounts[te_idx]
    # inner split of day 1 for threshold tuning (no peeking at day 2)
    itr, ival = train_test_split(np.arange(len(X_tr_all)), test_size=0.25,
                                 stratify=y_tr_all, random_state=SEED)
    rows, probs, feats = [], {}, {}
    for sname, fn in SELECTORS.items():
        feats[sname] = fn(X_tr_all.iloc[itr], y_tr_all.iloc[itr])
    total_runs = len(MODELS) * 2
    started = time.time()
    completed = 0
    for mname in MODELS:
        for cond, fs in [("all", list(X.columns)), ("selected", feats[PAIRS[mname]])]:
            # tune threshold on validation slice
            _, pv = fit_eval(mname, X_tr_all.iloc[itr], y_tr_all.iloc[itr],
                             X_tr_all.iloc[ival], y_tr_all.iloc[ival], fs,
                             amounts[tr_idx][ival], review_cost)
            thr, _, _ = best_threshold(y_tr_all.iloc[ival], pv, amounts[tr_idx][ival], review_cost)
            # refit on all of day 1, evaluate on day 2 at both thresholds
            o_def, prob = fit_eval(mname, X_tr_all, y_tr_all, X_te, y_te, fs, amt_te, review_cost, thr=0.5)
            o_cost = eval_at(y_te, prob, thr, amt_te, review_cost)
            o_def.update(condition=cond, protocol="temporal", tuning="default 0.5")
            o_cost.update(model=mname, condition=cond, protocol="temporal",
                          tuning="cost-tuned", threshold=thr, n_features=len(fs))
            rows += [o_def, o_cost]; probs[(mname, cond)] = prob
            completed += 1
            print_progress("Temporal models", completed, total_runs, started)
    return pd.DataFrame(rows), probs, feats

def wilcoxon_tests(cv_df, metric="pr_auc"):
    ev = cv_df[cv_df.kind == "eval"]
    out = []
    for m in ev.model.dropna().unique():
        a = ev[(ev.model == m) & (ev.condition == "all")].sort_values("fold")[metric].to_numpy()
        b = ev[(ev.model == m) & (ev.condition == "selected")].sort_values("fold")[metric].to_numpy()
        n = min(len(a), len(b)); a, b = a[:n], b[:n]
        try: st, p = wilcoxon(a, b)
        except ValueError: st, p = np.nan, np.nan
        out.append(dict(model=m, metric=metric, mean_all=a.mean(), mean_sel=b.mean(),
                        delta=b.mean() - a.mean(), sd_all=a.std(ddof=1), sd_sel=b.std(ddof=1),
                        wilcoxon_p=p, n_folds=n))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ analysis
def check_data(path):
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Download creditcard.csv from "
                 "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    with open(path, "rb") as f:
        lines = sum(b.count(b"\n") for b in iter(lambda: f.read(1 << 20), b""))
    if lines < 284808:
        sys.exit(f"{path} looks truncated: {lines:,} lines, expected 284,808. "
                 "Delete it and download again.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default="creditcard.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="reduced grid, ~10 min")
    ap.add_argument("--review-cost", type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n_splits, n_repeats = (3, 1) if args.quick else (5, 3)
    cost_grid = [1.0, 3.0, 10.0, 30.0]
    print(f"{'QUICK' if args.quick else 'FULL'} run: "
          f"{n_splits} splits x {n_repeats} repeats = {n_splits*n_repeats} folds")

    check_data(args.data)
    df = pd.read_csv(args.data)
    assert df.shape == (284807, 31), f"Unexpected shape {df.shape}"
    assert int(df.Class.sum()) == 492, f"Expected 492 frauds, got {int(df.Class.sum())}"

    X, y, day, amounts = prepare(df)
    print(f"\n{len(df):,} transactions | {int(y.sum())} frauds "
          f"({y.mean()*100:.3f}%) | {X.shape[1]} features")
    for d in (0, 1):
        m = day == d
        print(f"  Day {d+1}: {int(m.sum()):>7,} transactions | {int(y[m].sum()):>3} frauds "
              f"({y[m].mean()*100:.3f}%) | fraud value {amounts[m & (y == 1)].sum():>10,.0f}")

    # ---- temporal drift characterisation
    print("\n[1/5] Drift between day 1 and day 2 ...")
    drift = drift_report(X, y, day)
    drift.to_csv(f"{args.out}/drift_ks.csv", index=False)
    print(f"  features with KS p < 0.01: {(drift.p < 0.01).sum()} of {len(drift)}")
    print(drift.head(5).round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    d = drift.set_index("feature")["ks"].sort_values()
    colors = ["#C44E52" if p < 0.01 else "#9aa0a6"
              for p in drift.set_index("feature").loc[d.index, "p"]]
    ax[0].barh(d.index, d.values, color=colors)
    ax[0].set(title="Day 1 -> Day 2 shift (KS, legitimate traffic)", xlabel="KS statistic")
    ax[0].tick_params(labelsize=7)
    rates = [y[day == 0].mean() * 100, y[day == 1].mean() * 100]
    ax[1].bar(["Day 1", "Day 2"], rates, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(rates):
        ax[1].text(i, v, f"{v:.3f}%", ha="center", va="bottom")
    ax[1].set(title="Fraud prevalence by day", ylabel="% fraudulent")
    for d_, c, lab in [(0, "#4C72B0", "Day 1"), (1, "#C44E52", "Day 2")]:
        ax[2].hist(np.log1p(amounts[(day == d_) & (y == 1)]), bins=30, alpha=.6,
                   density=True, color=c, label=lab)
    ax[2].set(title="Fraud amount by day", xlabel="log(1 + Amount)", ylabel="Density")
    ax[2].legend()
    plt.tight_layout(); plt.savefig(f"{args.out}/f1_drift.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---- RQ2 + RQ3: repeated CV under the random protocol
    print(f"\n[2/5] Repeated cross-validation ({n_splits*n_repeats} folds) ...")
    t0 = time.time()
    cv_df, chosen = cv_experiment(X, y, amounts, n_splits, n_repeats, args.review_cost)
    cv_df.to_csv(f"{args.out}/cv_raw.csv", index=False)
    print(f"  done in {(time.time()-t0)/60:.1f} min")

    ev = cv_df[cv_df.kind == "eval"]
    summary = (ev.groupby(["model", "condition"])
                 [["pr_auc", "roc_auc", "recall", "precision", "f1", "fit_s"]]
                 .agg(["mean", "std"]).round(4))
    summary.to_csv(f"{args.out}/t1_cv_summary.csv")
    print("\n" + summary.to_string())

    tests = pd.concat([wilcoxon_tests(cv_df, m) for m in ["pr_auc", "recall", "f1"]],
                      ignore_index=True)
    tests["significant_at_0.05"] = tests.wilcoxon_p < 0.05
    tests.to_csv(f"{args.out}/t2_wilcoxon.csv", index=False)
    print("\n" + tests.round(4).to_string(index=False))

    # ---- stability
    print("\n[3/5] Selection stability ...")
    stab = pd.DataFrame([
        dict(selector=s, kuncheva=kuncheva(sets, X.shape[1]), jaccard=jaccard(sets),
             n_distinct_features=len(set().union(*map(set, sets))))
        for s, sets in chosen.items()])
    stab.to_csv(f"{args.out}/t3_stability.csv", index=False)
    freq = pd.DataFrame({s: pd.Series({f: sum(f in st for st in sets) / len(sets)
                                       for f in X.columns}) for s, sets in chosen.items()})
    freq["mean"] = freq.mean(axis=1)
    freq = freq.sort_values("mean", ascending=False)
    freq.to_csv(f"{args.out}/t4_selection_frequency.csv")
    print(stab.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2]})
    xp = np.arange(len(stab)); w = 0.38
    ax[0].bar(xp - w/2, stab.kuncheva, w, label="Kuncheva", color="#4C72B0")
    ax[0].bar(xp + w/2, stab.jaccard, w, label="Jaccard", color="#C44E52")
    ax[0].set_xticks(xp); ax[0].set_xticklabels(stab.selector, rotation=15, fontsize=8)
    ax[0].set(ylim=(0, 1.05), title="Selection stability across folds"); ax[0].legend(fontsize=8)
    sub = freq.drop(columns="mean").head(25)
    im = ax[1].imshow(sub.T.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax[1].set_xticks(range(len(sub))); ax[1].set_xticklabels(sub.index, rotation=90, fontsize=7)
    ax[1].set_yticks(range(sub.shape[1])); ax[1].set_yticklabels(sub.columns, fontsize=8)
    ax[1].set_title("Selection frequency across folds")
    plt.colorbar(im, ax=ax[1], shrink=.8)
    plt.tight_layout(); plt.savefig(f"{args.out}/f2_stability.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---- RQ1 + RQ4: temporal protocol
    print("\n[4/5] Temporal protocol (train day 1 -> test day 2) ...")
    t0 = time.time()
    temporal_df, temporal_probs, _ = temporal_experiment(X, y, day, amounts, args.review_cost)
    temporal_df.to_csv(f"{args.out}/t5_temporal.csv", index=False)
    print(f"  done in {(time.time()-t0)/60:.1f} min\n")
    print(temporal_df[["model", "condition", "tuning", "n_features", "threshold",
                       "precision", "recall", "f1", "pr_auc", "FP", "FN",
                       "cost", "savings_pct"]].round(4).to_string(index=False))

    rand_mean = (ev.groupby(["model", "condition"])["pr_auc"].mean().rename("pr_auc_random"))
    temp_mean = (temporal_df[temporal_df.tuning == "default 0.5"]
                   .set_index(["model", "condition"])["pr_auc"].rename("pr_auc_temporal"))
    gap = pd.concat([rand_mean, temp_mean], axis=1)
    gap["optimism"] = gap.pr_auc_random - gap.pr_auc_temporal
    gap["optimism_pct"] = 100 * gap.optimism / gap.pr_auc_temporal
    gap.to_csv(f"{args.out}/t6_protocol_gap.csv")
    print("\nRQ1 - optimism of the random protocol:")
    print(gap.round(4).to_string())

    X_d1, X_d2 = scale(X[day == 0], X[day == 1])
    y_d1, y_d2 = y[day == 0], y[day == 1]
    rows = []
    for sname, fn in SELECTORS.items():
        f1_, f2_ = fn(X_d1, y_d1), fn(X_d2, y_d2)
        rows.append(dict(selector=sname, overlap=len(set(f1_) & set(f2_)), k=len(f1_),
                         kuncheva=kuncheva([f1_, f2_], X.shape[1]), jaccard=jaccard([f1_, f2_]),
                         day1_only=sorted(set(f1_) - set(f2_)),
                         day2_only=sorted(set(f2_) - set(f1_))))
    cross_day = pd.DataFrame(rows)
    cross_day.to_csv(f"{args.out}/t7_cross_day_stability.csv", index=False)
    print("\nRQ3 - stability across the day boundary:")
    print(cross_day[["selector", "overlap", "k", "kuncheva", "jaccard"]].round(3).to_string(index=False))

    # ---- cost analysis
    print("\n[5/5] Cost analysis ...")
    y_d2_arr = y[day == 1].to_numpy(); amt_d2 = amounts[day == 1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    base = amt_d2[y_d2_arr == 1].sum()
    for (m, cond), prob in temporal_probs.items():
        if cond != "selected":
            continue
        thr, costs, base = cost_curve(y_d2_arr, prob, amt_d2, args.review_cost)
        i = int(np.argmin(costs))
        ax[0].plot(thr[1:], costs[1:], lw=1.5, label=f"{m} (min {costs[i]:,.0f} @ {thr[i]:.3f})")
        ax[0].scatter([thr[i]], [costs[i]], zorder=5, s=25)
    ax[0].axhline(base, color="k", ls="--", lw=.8, label=f"No model ({base:,.0f})")
    ax[0].set(xlabel="Decision threshold", ylabel="Total cost on day 2",
              title=f"Cost vs threshold (review fee = {args.review_cost:g})")
    ax[0].legend(fontsize=7)

    sens = []
    for c in cost_grid:
        for (m, cond), prob in temporal_probs.items():
            thr, cost, base_ = best_threshold(y_d2_arr, prob, amt_d2, c)
            sens.append(dict(review_cost=c, model=m, condition=cond, threshold=thr,
                             cost=cost, savings_pct=100 * (base_ - cost) / base_))
    sens = pd.DataFrame(sens)
    sens.to_csv(f"{args.out}/t8_cost_sensitivity.csv", index=False)
    for (m, cond), grp in sens.groupby(["model", "condition"]):
        ax[1].plot(grp.review_cost, grp.savings_pct, marker="o", ms=4,
                   ls="-" if cond == "selected" else "--", label=f"{m} ({cond})")
    ax[1].set(xscale="log", xlabel="Review cost per alert", ylabel="% of fraud value saved",
              title="Sensitivity to the review fee")
    ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{args.out}/f4_cost.png", dpi=200, bbox_inches="tight")
    plt.close()

    g = gap.reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    lab = [f"{m}\n({c})" for m, c in zip(g.model, g.condition)]
    xp = np.arange(len(g)); w = 0.38
    ax[0].bar(xp - w/2, g.pr_auc_random, w, label="Random CV", color="#4C72B0")
    ax[0].bar(xp + w/2, g.pr_auc_temporal, w, label="Temporal", color="#C44E52")
    ax[0].set_xticks(xp); ax[0].set_xticklabels(lab, fontsize=7)
    ax[0].set(ylabel="PR-AUC", title="RQ1: optimism of the random protocol"); ax[0].legend(fontsize=8)
    t = temporal_df[temporal_df.tuning == "cost-tuned"]
    ax[1].bar(range(len(t)), t.savings_pct, color=["#4C72B0", "#C44E52"] * 3)
    ax[1].set_xticks(range(len(t)))
    ax[1].set_xticklabels([f"{m}\n({c})" for m, c in zip(t.model, t.condition)], fontsize=7)
    ax[1].set(ylabel="% of fraud value saved", title="RQ4: money saved on day 2")
    plt.tight_layout()
    plt.savefig(f"{args.out}/f3_protocol_and_savings.png", dpi=200, bbox_inches="tight")
    plt.close()

    headline = []
    for m in MODELS:
        for cond in ["all", "selected"]:
            r = ev[(ev.model == m) & (ev.condition == cond)]
            td = temporal_df[(temporal_df.model == m) & (temporal_df.condition == cond)]
            tc = td[td.tuning == "cost-tuned"].iloc[0]
            headline.append({
                "Model": m, "Features": cond,
                "PR-AUC random (mean+-sd)": f"{r.pr_auc.mean():.4f} +- {r.pr_auc.std():.4f}",
                "PR-AUC temporal": f"{td[td.tuning=='default 0.5'].pr_auc.iloc[0]:.4f}",
                "Recall (cost-tuned)": f"{tc.recall:.4f}",
                "Precision (cost-tuned)": f"{tc.precision:.4f}",
                "FN": int(tc.FN), "Savings %": f"{tc.savings_pct:.2f}"})
    headline = pd.DataFrame(headline)
    headline.to_csv(f"{args.out}/t9_headline.csv", index=False)
    print("\nHeadline table:")
    print(headline.to_string(index=False))
    print(f"\nDone. Nine tables and four figures written to ./{args.out}/")


if __name__ == "__main__":
    main()
