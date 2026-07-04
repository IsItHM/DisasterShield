"""PERMANENT-WATER DIAGNOSTIC (Phase 2 closeout, Part 1).

For the FINAL v2 U-Net (results/20260703T054304Z/feni_unet_best.keras) at operating
threshold 0.65, take the three permanent-water false-positive test patches (129, 153, 178;
true flood frac 0, high predicted flood) and measure what fraction of each patch's
FALSE-POSITIVE pixels (pred flood AND true non-flood) fall inside the pre-flood
permanent-water proxy VV_pre < -16 dB (raw dB, the Step-2c definition).

Verdict per patch: CONFIRMED river-or-pond confusion if most FP pixels sit on permanent
water; NOT CONFIRMED otherwise.

Freezes: results/<ts>/permwater_diagnostic/{patch_XXX.png, permwater_diagnostic.json}.
No training. Numbers only from frozen inputs. Seed 42.
"""
import os, json, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROC = r"d:\Hamim\DisasterShield\data\processed"
UNET = r"d:\Hamim\DisasterShield\results\20260703T054304Z\feni_unet_best.keras"
CH = {"VV_flood": 0, "VH_flood": 1, "VV_pre": 2, "VH_pre": 3}
PATCHES = [129, 153, 178]
THRESH = 0.65
PERMWATER_DB = -16.0
CONFIRM_FRAC = 0.5  # >= this fraction of FP pixels on permanent water => CONFIRMED

ns = json.load(open(os.path.join(PROC, "norm_stats_v2.json")))
MEANS = np.array(ns["mean"], dtype=np.float32)
STDS = np.array(ns["std"], dtype=np.float32)

X = np.load(os.path.join(PROC, "feni_X_test_v2.npy")).astype(np.float32)
y = np.load(os.path.join(PROC, "feni_y_test_v2.npy")).astype(np.float32)

# Raw VV_pre (dB) for the permanent-water proxy, kept BEFORE mean-fill so NaN pixels
# never count as permanent water.
VVpre_raw = X[..., CH["VV_pre"]].copy()

Xf = np.where(np.isfinite(X), X, MEANS)          # mean-fill non-finite (as in training)
Xs = ((Xf - MEANS) / STDS).astype(np.float32)    # standardize with train stats

print("loading U-Net:", UNET)
model = tf.keras.models.load_model(UNET, compile=False)

ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out_dir = os.path.join(r"d:\Hamim\DisasterShield\results", ts, "permwater_diagnostic")
os.makedirs(out_dir, exist_ok=True)

records = []
for p in PATCHES:
    prob = model.predict(Xs[p:p + 1], verbose=0)[0, ..., 1]
    pred = (prob > THRESH).astype(np.uint8)
    true = (y[p] == 1).astype(np.uint8)
    fp = (pred == 1) & (true == 0)
    permw = np.isfinite(VVpre_raw[p]) & (VVpre_raw[p] < PERMWATER_DB)
    fp_n = int(fp.sum())
    fp_in = int((fp & permw).sum())
    frac = (fp_in / fp_n) if fp_n > 0 else float("nan")
    verdict = "CONFIRMED" if (fp_n > 0 and frac >= CONFIRM_FRAC) else "NOT CONFIRMED"
    rec = {
        "patch_index": p,
        "true_flood_frac": float(true.mean()),
        "pred_flood_frac_at_0.65": float(pred.mean()),
        "fp_pixels": fp_n,
        "fp_in_permwater": fp_in,
        "frac_fp_in_permwater": frac,
        "permwater_frac_of_patch": float(permw.mean()),
        "verdict": verdict,
    }
    records.append(rec)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4.6))
    im0 = ax[0].imshow(VVpre_raw[p], cmap="gray")
    ax[0].set_title(f"VV_pre (dB) — patch {p}")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    ax[1].imshow(pred, cmap="Blues", vmin=0, vmax=1)
    ax[1].set_title(f"Prediction > {THRESH} (flood)")
    ax[2].imshow(permw, cmap="Oranges", vmin=0, vmax=1)
    ax[2].set_title(f"Permanent water (VV_pre < {PERMWATER_DB:g} dB)")
    for a in ax:
        a.axis("off")
    fig.suptitle(
        f"patch {p}:  FP inside permanent water / FP = {fp_in}/{fp_n} = "
        f"{frac:.3f}  ->  {verdict}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"patch_{p:03d}.png"), dpi=130)
    plt.close()
    print(f"patch {p}: true={true.mean():.3f} pred@0.65={pred.mean():.3f} "
          f"FP={fp_n} in_permw={fp_in} frac={frac:.4f} -> {verdict}")

summary = {
    "timestamp_utc": ts, "seed": SEED, "model": UNET, "threshold": THRESH,
    "permwater_proxy": "VV_pre (raw dB) < -16",
    "confirm_fraction": CONFIRM_FRAC,
    "metric": ("fraction of false-positive pixels (pred flood AND true non-flood) "
               "that fall inside the pre-flood permanent-water proxy"),
    "patches": PATCHES,
    "records": records,
}
json.dump(summary, open(os.path.join(out_dir, "permwater_diagnostic.json"), "w"), indent=2)
print("OUT:", out_dir)
