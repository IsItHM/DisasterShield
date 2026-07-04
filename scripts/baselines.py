"""
Baselines on the SAME frozen splits as train_flood_model.py.

Baseline A: NDWI threshold. Sweep threshold in [-0.2, 0.5] step 0.05 on the 2022
            VALIDATION patches only; pick best-val threshold; report test IoU/F1.
Baseline B: per-pixel logistic regression on the 5 channels, trained on a
            500k-pixel random sample (seed 42) from 2019-2021 training patches.

Patch extraction is deterministic, so regenerating the patches here reproduces the
exact tensors used in training. Writes results/<UTC-timestamp>/baseline_comparison.csv.
Optionally embeds the U-Net row from a training run's metrics.json (--unet-metrics).
"""
import os
import csv
import json
import argparse
import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression

import train_flood_model as T  # reuse the exact data pipeline

SEED = 42
np.random.seed(SEED)


def iou_f1(y_true_flat, y_pred_flat):
    """Micro IoU + F1 for the water class over all pixels (matches training metric)."""
    yt = (y_true_flat == 1).astype(np.float64)
    yp = (y_pred_flat == 1).astype(np.float64)
    inter = float((yt * yp).sum())
    union = float(yt.sum() + yp.sum() - inter)
    iou = (inter + 1e-6) / (union + 1e-6)
    tp = inter
    fp = float(((1 - yt) * yp).sum())
    fn = float((yt * (1 - yp)).sum())
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return iou, f1


def build_splits():
    yf = T.discover_files(T.DATA_DIR)
    train_Xs, train_ys = [], []
    for yr in T.TRAIN_YEARS:
        Xyr, yyr = T.load_year(yr, yf)
        train_Xs.append(Xyr); train_ys.append(yyr)
    X_train = np.concatenate(train_Xs, axis=0)
    y_train = np.concatenate(train_ys, axis=0)
    X_val, y_val = T.load_year(T.VAL_YEAR, yf)
    X_test, y_test = T.load_year(T.TEST_YEAR, yf)
    return X_train, y_train, X_val, y_val, X_test, y_test


def ndwi_from_channel(X):
    """Channel 4 stores (ndwi+1)/2; invert to raw NDWI in [-1, 1]."""
    return 2.0 * X[..., 4] - 1.0


def baseline_ndwi(X_val, y_val, X_test, y_test):
    thresholds = np.arange(-0.2, 0.5 + 1e-9, 0.05)
    ndwi_val = ndwi_from_channel(X_val)
    best_thr, best_iou = None, -1.0
    sweep = []
    for thr in thresholds:
        pred = (ndwi_val > thr).astype(np.int32)
        iou, f1 = iou_f1(y_val.reshape(-1), pred.reshape(-1))
        sweep.append((round(float(thr), 2), iou, f1))
        if iou > best_iou:
            best_iou, best_thr = iou, float(thr)
    # Evaluate chosen threshold on TEST (never tuned on test)
    ndwi_test = ndwi_from_channel(X_test)
    pred_test = (ndwi_test > best_thr).astype(np.int32)
    test_iou, test_f1 = iou_f1(y_test.reshape(-1), pred_test.reshape(-1))
    return {
        "method": "NDWI_threshold",
        "val_IoU": round(best_iou, 6),
        "test_IoU": round(test_iou, 6),
        "test_F1": round(test_f1, 6),
        "threshold": round(best_thr, 2),
        "sweep": sweep,
    }


def baseline_logreg(X_train, y_train, X_val, y_val, X_test, y_test):
    Xtr = X_train.reshape(-1, 5)
    ytr = (y_train.reshape(-1) == 1).astype(np.int32)
    rng = np.random.RandomState(SEED)
    n = min(500_000, Xtr.shape[0])
    idx = rng.choice(Xtr.shape[0], n, replace=False)
    Xs, ys = Xtr[idx], ytr[idx]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xs, ys)

    val_pred = clf.predict(X_val.reshape(-1, 5))
    val_iou, _ = iou_f1((y_val.reshape(-1) == 1).astype(np.int32), val_pred)
    test_pred = clf.predict(X_test.reshape(-1, 5))
    test_iou, test_f1 = iou_f1((y_test.reshape(-1) == 1).astype(np.int32), test_pred)
    return {
        "method": "logistic_regression",
        "val_IoU": round(val_iou, 6),
        "test_IoU": round(test_iou, 6),
        "test_F1": round(test_f1, 6),
        "threshold": "",
        "n_pixels_trained": int(n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet-metrics", default="", help="Path to U-Net metrics.json to embed a U-Net row.")
    ap.add_argument("--out-dir", default="", help="Existing results dir to write into; else new timestamp dir.")
    args = ap.parse_args()

    print("Rebuilding frozen splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = build_splits()
    print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

    a = baseline_ndwi(X_val, y_val, X_test, y_test)
    print("Baseline A (NDWI) sweep on val:")
    for thr, iou, f1 in a["sweep"]:
        print(f"  thr={thr:+.2f}  val_IoU={iou:.4f}  val_F1={f1:.4f}")
    print(f"  -> chosen thr={a['threshold']} | test_IoU={a['test_IoU']} | test_F1={a['test_F1']}")

    b = baseline_logreg(X_train, y_train, X_val, y_val, X_test, y_test)
    print(f"Baseline B (LogReg on {b['n_pixels_trained']} px): "
          f"val_IoU={b['val_IoU']} | test_IoU={b['test_IoU']} | test_F1={b['test_F1']}")

    rows = []
    if args.unet_metrics and os.path.exists(args.unet_metrics):
        with open(args.unet_metrics) as f:
            m = json.load(f)
        rows.append({
            "method": "UNet_5ch",
            "val_IoU": round(m["best_val_iou"], 9),
            "test_IoU": round(m["test_iou"], 9),
            "test_F1": round(m["test_f1"], 9),
            "threshold": "",
        })
    rows.append({k: a[k] for k in ("method", "val_IoU", "test_IoU", "test_F1", "threshold")})
    rows.append({k: b[k] for k in ("method", "val_IoU", "test_IoU", "test_F1", "threshold")})

    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = os.path.join(r"d:\Hamim\DisasterShield\results", ts)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "baseline_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "val_IoU", "test_IoU", "test_F1", "threshold"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # Also dump the NDWI sweep for the record
    with open(os.path.join(out_dir, "ndwi_threshold_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "val_IoU", "val_F1"])
        for thr, iou, f1 in a["sweep"]:
            w.writerow([thr, round(iou, 6), round(f1, 6)])

    print("\nWROTE:", csv_path)
    print("OUT_DIR:", out_dir)


if __name__ == "__main__":
    main()
