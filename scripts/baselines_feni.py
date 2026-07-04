"""STEP 4c-1 - Baselines on v2 splits (identical patches as the U-Net).
A: VV-change threshold delta=VV_flood-VV_pre (raw dB), flood = delta < t, sweep [-10,+2] step 0.5 on VAL.
B: VV_flood absolute, flood = VV_flood < t, sweep [-25,-5] step 0.5 on VAL.
C: per-pixel logistic regression on 4 raw-dB channels, 500k train-pixel sample, seed 42.
Non-finite pixels mean-filled with TRAIN channel means (same handling as training).
Metrics = GLOBAL pixel-level IoU/F1 (water class), matching train_feni.py's metric definition.
Freeze -> results/<ts>/baseline_comparison.csv.
"""
import os, json, csv, datetime
import numpy as np
from sklearn.linear_model import LogisticRegression

SEED = 42
np.random.seed(SEED)
PROC = r"d:\Hamim\DisasterShield\data\processed"
# channel order: 0 VV_flood, 1 VH_flood, 2 VV_pre, 3 VH_pre
CH = {"VV_flood": 0, "VH_flood": 1, "VV_pre": 2, "VH_pre": 3}

means = np.array(json.load(open(os.path.join(PROC, "norm_stats_v2.json")))["mean"], dtype=np.float32)


def load(sp):
    X = np.load(os.path.join(PROC, f"feni_X_{sp}_v2.npy")).astype(np.float32)
    y = np.load(os.path.join(PROC, f"feni_y_{sp}_v2.npy")).astype(np.float32)
    X = np.where(np.isfinite(X), X, means)  # mean-fill non-finite (same as training)
    return X, y


def iou_f1(y_true, pred):
    yt = (y_true.reshape(-1) == 1).astype(np.float64)
    yp = pred.reshape(-1).astype(np.float64)
    tp = float((yt * yp).sum())
    fp = float(((1 - yt) * yp).sum())
    fn = float((yt * (1 - yp)).sum())
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    return iou, f1


def sweep(signal_val, y_val, thresholds):
    """flood = signal < t (both A and B: flood = low value). Pick best VAL IoU."""
    best_t, best_iou = None, -1.0
    for t in thresholds:
        iou, _ = iou_f1(y_val, (signal_val < t).astype(np.int8))
        if iou > best_iou:
            best_iou, best_t = iou, float(t)
    return best_t, best_iou


def main():
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(r"d:\Hamim\DisasterShield\results", ts)
    os.makedirs(out_dir, exist_ok=True)
    print("baselines out_dir:", out_dir)

    Xtr, ytr = load("train")
    Xva, yva = load("val")
    Xte, yte = load("test")
    print("loaded", Xtr.shape, Xva.shape, Xte.shape)

    rows = []

    # ── Baseline A: VV-change ──────────────────────────────────────────────
    dVA = Xva[..., CH["VV_flood"]] - Xva[..., CH["VV_pre"]]
    dTE = Xte[..., CH["VV_flood"]] - Xte[..., CH["VV_pre"]]
    thrA = np.arange(-10, 2 + 1e-9, 0.5)
    tA, valA = sweep(dVA, yva, thrA)
    iouA, f1A = iou_f1(yte, (dTE < tA).astype(np.int8))
    rows.append(["A_VVchange_delta<t", tA, valA, iouA, f1A])
    print(f"A  VVchange  t={tA}  val_IoU={valA:.4f}  test_IoU={iouA:.4f}  test_F1={f1A:.4f}")

    # ── Baseline B: VV_flood absolute ──────────────────────────────────────
    vVA = Xva[..., CH["VV_flood"]]
    vTE = Xte[..., CH["VV_flood"]]
    thrB = np.arange(-25, -5 + 1e-9, 0.5)
    tB, valB = sweep(vVA, yva, thrB)
    iouB, f1B = iou_f1(yte, (vTE < tB).astype(np.int8))
    rows.append(["B_VVflood_abs<t", tB, valB, iouB, f1B])
    print(f"B  VVflood   t={tB}  val_IoU={valB:.4f}  test_IoU={iouB:.4f}  test_F1={f1B:.4f}")

    # ── Baseline C: logistic regression on 4 raw-dB channels ───────────────
    Ptr = Xtr.reshape(-1, 4)
    Ltr = (ytr.reshape(-1) == 1).astype(np.int8)
    n = Ptr.shape[0]
    idx = np.random.RandomState(SEED).choice(n, 500000, replace=False)
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(Ptr[idx], Ltr[idx])
    predVA = clf.predict(Xva.reshape(-1, 4))
    predTE = clf.predict(Xte.reshape(-1, 4))
    valC, _ = iou_f1(yva, predVA)
    iouC, f1C = iou_f1(yte, predTE)
    rows.append(["C_logreg_4ch", "NA(decision=0.5)", valC, iouC, f1C])
    print(f"C  logreg    val_IoU={valC:.4f}  test_IoU={iouC:.4f}  test_F1={f1C:.4f}")
    print("   logreg coef:", [round(float(c), 4) for c in clf.coef_[0]], "intercept", round(float(clf.intercept_[0]), 4))

    with open(os.path.join(out_dir, "baseline_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "chosen_threshold", "val_IoU", "test_IoU", "test_F1"])
        for r in rows:
            w.writerow(r)

    summary = {
        "timestamp_utc": ts, "tag": "v2", "seed": SEED,
        "nonfinite_handling": "mean-filled with train channel means (same as training)",
        "metric": "global pixel-level water-class IoU/F1",
        "logreg": {"coef": [float(c) for c in clf.coef_[0]],
                   "intercept": float(clf.intercept_[0]),
                   "channels": list(CH.keys()), "n_sample": 500000},
        "rows": [{"method": r[0], "chosen_threshold": r[1], "val_IoU": r[2],
                  "test_IoU": r[3], "test_F1": r[4]} for r in rows],
    }
    json.dump(summary, open(os.path.join(out_dir, "baselines_summary.json"), "w"), indent=2)
    print("RESULT_DIR:", out_dir)


if __name__ == "__main__":
    main()
