"""STEP 5d - THRESHOLD-FAIR RE-EVALUATION (v2, no training needed).

For the frozen v2 U-Net (results/20260703T054304Z/feni_unet_best.keras) and the logistic
regression baseline C, sweep the decision threshold on PROBABILITY outputs over
[0.05, 0.95] step 0.05 on VAL ONLY; pick the best-val-IoU threshold per method; then report
TEST IoU/F1 at that threshold. Also report both methods at the fixed 0.5 cut, plus the
threshold baselines A (VV_flood-VV_pre < t) and B (VV_flood < t).

Metric = GLOBAL pixel-level water-class IoU/F1 (same definition as train_feni.py / baselines_feni.py).
Data + standardization identical to training: v2 patches, mean-fill non-finite with TRAIN means,
standardize by norm_stats_v2 mean/std. logreg refit on 500k train-pixel sample, seed 42
(reproduces baseline C). Freeze -> results/<ts>/threshold_fair_v2.csv + .json.
"""
import os, json, csv, datetime
import numpy as np
from sklearn.linear_model import LogisticRegression

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROC = r"d:\Hamim\DisasterShield\data\processed"
UNET_PATH = r"d:\Hamim\DisasterShield\results\20260703T054304Z\feni_unet_best.keras"
CH = {"VV_flood": 0, "VH_flood": 1, "VV_pre": 2, "VH_pre": 3}

ns = json.load(open(os.path.join(PROC, "norm_stats_v2.json")))
MEANS = np.array(ns["mean"], dtype=np.float32)
STDS = np.array(ns["std"], dtype=np.float32)

THRESHOLDS = np.round(np.arange(0.05, 0.95 + 1e-9, 0.05), 2)  # [0.05 .. 0.95]


def load_raw(sp):
    """Raw v2 patches with non-finite mean-filled (for baselines A/B/C features)."""
    X = np.load(os.path.join(PROC, f"feni_X_{sp}_v2.npy")).astype(np.float32)
    y = np.load(os.path.join(PROC, f"feni_y_{sp}_v2.npy")).astype(np.float32)
    X = np.where(np.isfinite(X), X, MEANS)
    return X, y


def standardize(X):
    return ((X - MEANS) / STDS).astype(np.float32)


def iou_f1(y_true, pred_bin):
    yt = (y_true.reshape(-1) == 1).astype(np.float64)
    yp = pred_bin.reshape(-1).astype(np.float64)
    tp = float((yt * yp).sum())
    fp = float(((1 - yt) * yp).sum())
    fn = float((yt * (1 - yp)).sum())
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    return iou, f1


def sweep_prob(prob_val, y_val):
    """Pick threshold maximizing VAL IoU over THRESHOLDS. prob>t (strict, matches training >0.5)."""
    best_t, best_iou = None, -1.0
    for t in THRESHOLDS:
        iou, _ = iou_f1(y_val, (prob_val > t).astype(np.int8))
        if iou > best_iou:
            best_iou, best_t = iou, float(t)
    return best_t


def sweep_signal(sig_val, y_val, thresholds):
    """flood = signal < t. Pick best VAL IoU (baselines A/B)."""
    best_t, best_iou = None, -1.0
    for t in thresholds:
        iou, _ = iou_f1(y_val, (sig_val < t).astype(np.int8))
        if iou > best_iou:
            best_iou, best_t = iou, float(t)
    return best_t


def main():
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(r"d:\Hamim\DisasterShield\results", ts)
    os.makedirs(out_dir, exist_ok=True)
    print("threshold-fair out_dir:", out_dir)

    Xtr, ytr = load_raw("train")
    Xva, yva = load_raw("val")
    Xte, yte = load_raw("test")
    print("loaded", Xtr.shape, Xva.shape, Xte.shape)

    rows = []  # method, val_threshold, val_IoU, val_F1, test_IoU, test_F1

    # ── U-Net probabilities (standardized input, prob = softmax water channel) ──
    print("loading U-Net:", UNET_PATH)
    model = tf.keras.models.load_model(UNET_PATH, compile=False)
    prob_va = model.predict(standardize(Xva), batch_size=16, verbose=0)[..., 1]
    prob_te = model.predict(standardize(Xte), batch_size=16, verbose=0)[..., 1]

    # U-Net @ 0.5
    iou_v, f1_v = iou_f1(yva, (prob_va > 0.5).astype(np.int8))
    iou_t, f1_t = iou_f1(yte, (prob_te > 0.5).astype(np.int8))
    rows.append(["U-Net @0.5", 0.5, iou_v, f1_v, iou_t, f1_t])
    print(f"U-Net @0.5           val_IoU={iou_v:.4f} test_IoU={iou_t:.4f}")

    # U-Net @ val-tuned
    t_un = sweep_prob(prob_va, yva)
    iou_v, f1_v = iou_f1(yva, (prob_va > t_un).astype(np.int8))
    iou_t, f1_t = iou_f1(yte, (prob_te > t_un).astype(np.int8))
    rows.append(["U-Net @val-tuned", t_un, iou_v, f1_v, iou_t, f1_t])
    print(f"U-Net @val-tuned t={t_un}  val_IoU={iou_v:.4f} test_IoU={iou_t:.4f}")

    # ── Logistic regression (refit = baseline C, seed 42, 500k sample) ──────────
    Ptr = Xtr.reshape(-1, 4)
    Ltr = (ytr.reshape(-1) == 1).astype(np.int8)
    idx = np.random.RandomState(SEED).choice(Ptr.shape[0], 500000, replace=False)
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(Ptr[idx], Ltr[idx])
    lp_va = clf.predict_proba(Xva.reshape(-1, 4))[:, 1].reshape(yva.shape)
    lp_te = clf.predict_proba(Xte.reshape(-1, 4))[:, 1].reshape(yte.shape)

    # logreg @ 0.5
    iou_v, f1_v = iou_f1(yva, (lp_va > 0.5).astype(np.int8))
    iou_t, f1_t = iou_f1(yte, (lp_te > 0.5).astype(np.int8))
    rows.append(["logreg @0.5", 0.5, iou_v, f1_v, iou_t, f1_t])
    print(f"logreg @0.5          val_IoU={iou_v:.4f} test_IoU={iou_t:.4f}")

    # logreg @ val-tuned
    t_lr = sweep_prob(lp_va, yva)
    iou_v, f1_v = iou_f1(yva, (lp_va > t_lr).astype(np.int8))
    iou_t, f1_t = iou_f1(yte, (lp_te > t_lr).astype(np.int8))
    rows.append(["logreg @val-tuned", t_lr, iou_v, f1_v, iou_t, f1_t])
    print(f"logreg @val-tuned t={t_lr}  val_IoU={iou_v:.4f} test_IoU={iou_t:.4f}")

    # ── Baselines A / B (signal-space thresholds, val-tuned; reproduces 4c-1) ──
    dVA = Xva[..., CH["VV_flood"]] - Xva[..., CH["VV_pre"]]
    dTE = Xte[..., CH["VV_flood"]] - Xte[..., CH["VV_pre"]]
    tA = sweep_signal(dVA, yva, np.arange(-10, 2 + 1e-9, 0.5))
    iouAv, f1Av = iou_f1(yva, (dVA < tA).astype(np.int8))
    iouAt, f1At = iou_f1(yte, (dTE < tA).astype(np.int8))
    rows.append([f"A VVchange<{tA:g}dB", tA, iouAv, f1Av, iouAt, f1At])
    print(f"A  t={tA}  val_IoU={iouAv:.4f} test_IoU={iouAt:.4f}")

    vVA = Xva[..., CH["VV_flood"]]
    vTE = Xte[..., CH["VV_flood"]]
    tB = sweep_signal(vVA, yva, np.arange(-25, -5 + 1e-9, 0.5))
    iouBv, f1Bv = iou_f1(yva, (vVA < tB).astype(np.int8))
    iouBt, f1Bt = iou_f1(yte, (vTE < tB).astype(np.int8))
    rows.append([f"B VVflood<{tB:g}dB", tB, iouBv, f1Bv, iouBt, f1Bt])
    print(f"B  t={tB}  val_IoU={iouBv:.4f} test_IoU={iouBt:.4f}")

    # ── Freeze ─────────────────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "threshold_fair_v2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "val_threshold", "val_IoU", "val_F1", "test_IoU", "test_F1"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.6f}", f"{r[5]:.6f}"])

    summary = {
        "timestamp_utc": ts, "tag": "v2", "seed": SEED,
        "unet_model": UNET_PATH,
        "metric": "global pixel-level water-class IoU/F1",
        "prob_threshold_sweep": {"start": 0.05, "stop": 0.95, "step": 0.05,
                                 "selection": "max VAL IoU, decision = prob > t"},
        "norm_stats": "data/processed/norm_stats_v2.json",
        "rows": [{"method": r[0], "val_threshold": r[1], "val_IoU": r[2], "val_F1": r[3],
                  "test_IoU": r[4], "test_F1": r[5]} for r in rows],
    }
    json.dump(summary, open(os.path.join(out_dir, "threshold_fair_v2.json"), "w"), indent=2)
    print("RESULT_DIR:", out_dir)


if __name__ == "__main__":
    main()
