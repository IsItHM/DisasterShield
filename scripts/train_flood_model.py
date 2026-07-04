"""
DisasterShield-X: Multi-Year Flood Segmentation (2019-2023)
Script conversion of models/flood_model_2019_2023.ipynb, logic preserved exactly.

Year split: train 2019-2021 / val 2022 / test 2023
5-channel input (4 spectral bands + NDWI), PATCH_SIZE=64, STRIDE=48
Loss: Dice + weighted cross-entropy. Metrics: water-class IoU and F1.
Callbacks: ModelCheckpoint + EarlyStopping(patience=12) + ReduceLROnPlateau.

All outputs -> results/<UTC-timestamp>/.
"""
import os
import sys
import json
import time
import glob
import random
import argparse
import datetime
import csv

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ─── Reproducibility: seed everything to 42 ─────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = r"d:\Hamim\DisasterShield\data\Raw_data"
YEARS = [2019, 2020, 2021, 2022, 2023]
TRAIN_YEARS = [2019, 2020, 2021]
VAL_YEAR = 2022
TEST_YEAR = 2023

PATCH_SIZE = 64
STRIDE = 48
BATCH_SIZE = 16
L2_REG = 1e-4


# ─── File discovery (filenames WITHOUT FloodSeason token) ───────────────────
def discover_files(data_dir):
    year_files = {}
    for year in YEARS:
        rgb_path = os.path.join(data_dir, f"Bangladesh_RGB_{year}.tif")
        mask_path = os.path.join(data_dir, f"Bangladesh_WaterMask_{year}.tif")
        rgb_exists = os.path.exists(rgb_path)
        mask_exists = os.path.exists(mask_path)
        print(f"{year}: RGB={'OK' if rgb_exists else 'MISSING'}  "
              f"Mask={'OK' if mask_exists else 'MISSING'}")
        if rgb_exists and mask_exists:
            year_files[year] = (rgb_path, mask_path)
    print(f"\nFound complete pairs for: {list(year_files.keys())}")
    return year_files


# ─── NDWI + patch extraction (preserved from notebook) ──────────────────────
def compute_ndwi(rgb_data):
    """Bands: B2(0) B3/Green(1) B4(2) B8/NIR(3). NDWI = (Green - NIR)/(Green + NIR)."""
    green = rgb_data[1].astype(np.float32)
    nir = rgb_data[3].astype(np.float32)
    ndwi = (green - nir) / (green + nir + 1e-8)
    ndwi = np.clip(ndwi, -1, 1)
    return ndwi


def extract_patches(rgb_data, mask_data, min_water_fraction=0.0):
    ndwi = compute_ndwi(rgb_data)
    bands = np.clip(rgb_data / 10000.0, 0, 1)          # (4, H, W)
    ndwi_norm = (ndwi + 1) / 2.0                        # -1..1 -> 0..1
    data_5ch = np.concatenate([bands, ndwi_norm[np.newaxis]], axis=0)  # (5,H,W)
    data_5ch = data_5ch.transpose(1, 2, 0)             # (H,W,5)

    _, H, W = rgb_data.shape
    patches_X, patches_y = [], []
    for i in range(0, H - PATCH_SIZE, STRIDE):
        for j in range(0, W - PATCH_SIZE, STRIDE):
            patch_x = data_5ch[i:i + PATCH_SIZE, j:j + PATCH_SIZE, :]
            patch_y = mask_data[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
            if not np.isfinite(patch_x).all():
                continue
            if patch_x[..., :4].max() == 0:
                continue
            water_frac = (patch_y == 1).mean()
            if water_frac < min_water_fraction:
                continue
            patches_X.append(patch_x)
            patches_y.append(patch_y)

    if len(patches_X) == 0:
        return (np.empty((0, PATCH_SIZE, PATCH_SIZE, 5)),
                np.empty((0, PATCH_SIZE, PATCH_SIZE)))
    return (np.array(patches_X, dtype=np.float32),
            np.array(patches_y, dtype=np.float32))


def load_year(year, year_files):
    if year not in year_files:
        print(f"  WARNING: {year} not found, skipping.")
        return None, None
    rgb_path, mask_path = year_files[year]
    # Cast to float32 immediately after reading (stored float64 -> halves RAM)
    with rasterio.open(rgb_path) as src:
        rgb_data = src.read().astype(np.float32)
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1).astype(np.float32)
    X, y = extract_patches(rgb_data, mask_data)
    wf = (y == 1).mean() if X.shape[0] > 0 else 0.0
    print(f"  {year}: {X.shape[0]} patches | water: {wf * 100:.1f}%")
    return X, y


def ground_pixel_size_m(rgb_path):
    """Approximate ground pixel size in meters from the GeoTIFF transform + latitude."""
    with rasterio.open(rgb_path) as src:
        t = src.transform
        px_deg_x = abs(t.a)
        px_deg_y = abs(t.e)
        b = src.bounds
        center_lat = (b.top + b.bottom) / 2.0
        crs = str(src.crs)
    lat_rad = np.deg2rad(center_lat)
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    return {
        "crs": crs,
        "center_lat_deg": float(center_lat),
        "pixel_deg_x": float(px_deg_x),
        "pixel_deg_y": float(px_deg_y),
        "pixel_size_m_x": float(px_deg_x * m_per_deg_lon),
        "pixel_size_m_y": float(px_deg_y * m_per_deg_lat),
    }


# ─── Model ──────────────────────────────────────────────────────────────────
def conv_block(x, filters, dropout_rate=0.0):
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    return x


def build_unet(input_shape=(64, 64, 5), num_classes=2):
    inputs = layers.Input(shape=input_shape)
    e1 = conv_block(inputs, 32, 0.0)
    p1 = layers.MaxPooling2D(2)(e1)
    e2 = conv_block(p1, 64, 0.2)
    p2 = layers.MaxPooling2D(2)(e2)
    e3 = conv_block(p2, 128, 0.3)
    p3 = layers.MaxPooling2D(2)(e3)
    b = conv_block(p3, 256, 0.4)
    u3 = layers.UpSampling2D(2)(b)
    u3 = layers.Concatenate()([u3, e3])
    d3 = conv_block(u3, 128, 0.3)
    u2 = layers.UpSampling2D(2)(d3)
    u2 = layers.Concatenate()([u2, e2])
    d2 = conv_block(u2, 64, 0.2)
    u1 = layers.UpSampling2D(2)(d2)
    u1 = layers.Concatenate()([u1, e1])
    d1 = conv_block(u1, 32, 0.0)
    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(d1)
    return Model(inputs, outputs, name="DisasterShield_UNet")


# ─── Loss + metrics ─────────────────────────────────────────────────────────
def make_combined_loss(water_weight):
    def combined_loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_water = y_pred[..., 1]
        numerator = 2.0 * tf.reduce_sum(y_true_f * y_pred_water)
        denominator = tf.reduce_sum(y_true_f + y_pred_water)
        dice_loss = 1.0 - (numerator + 1e-6) / (denominator + 1e-6)
        weights = y_true_f * (water_weight - 1.0) + 1.0
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        wce = tf.reduce_mean(ce * weights)
        return dice_loss + wce
    return combined_loss


def iou_metric(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_w = tf.reshape(y_pred[..., 1], [-1])
    y_pred_b = tf.cast(y_pred_w > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_b)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_b) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def f1_metric(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_w = tf.reshape(y_pred[..., 1], [-1])
    y_pred_b = tf.cast(y_pred_w > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred_b)
    fp = tf.reduce_sum((1 - y_true) * y_pred_b)
    fn = tf.reduce_sum(y_true * (1 - y_pred_b))
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6)


def per_patch_iou(y_true, pred_bin):
    """IoU per patch (water class). y_true (N,H,W), pred_bin (N,H,W)."""
    out = []
    for i in range(y_true.shape[0]):
        yt = (y_true[i] == 1).astype(np.float32)
        yp = pred_bin[i].astype(np.float32)
        inter = float((yt * yp).sum())
        union = float(yt.sum() + yp.sum() - inter)
        out.append((inter + 1e-6) / (union + 1e-6))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: cap train to 200 patches, force 2 epochs.")
    args = ap.parse_args()

    epochs = 2 if args.smoke else args.epochs

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(r"d:\Hamim\DisasterShield\results", ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Mode: {'SMOKE' if args.smoke else 'FULL'} | epochs={epochs}")
    print("TensorFlow:", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

    year_files = discover_files(DATA_DIR)
    missing = [y for y in YEARS if y not in year_files]
    if missing:
        raise FileNotFoundError(f"Missing complete year pairs for: {missing}")

    pixel_info = ground_pixel_size_m(year_files[2019][0])
    print("Ground pixel size (m):",
          round(pixel_info["pixel_size_m_x"], 2), "x",
          round(pixel_info["pixel_size_m_y"], 2))

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nLoading TRAINING years:")
    train_Xs, train_ys, per_year_counts = [], [], {}
    for yr in TRAIN_YEARS:
        Xyr, yyr = load_year(yr, year_files)
        per_year_counts[yr] = int(Xyr.shape[0])
        if Xyr is not None and Xyr.shape[0] > 0:
            train_Xs.append(Xyr)
            train_ys.append(yyr)
    X_train = np.concatenate(train_Xs, axis=0)
    y_train = np.concatenate(train_ys, axis=0)

    print(f"\nLoading VALIDATION year ({VAL_YEAR}):")
    X_val, y_val = load_year(VAL_YEAR, year_files)
    per_year_counts[VAL_YEAR] = int(X_val.shape[0])

    print(f"\nLoading TEST year ({TEST_YEAR}):")
    X_test, y_test = load_year(TEST_YEAR, year_files)
    per_year_counts[TEST_YEAR] = int(X_test.shape[0])

    if args.smoke and X_train.shape[0] > 200:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(X_train.shape[0], 200, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"  [smoke] train capped to {X_train.shape[0]} patches")

    print(f"\nTrain: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # ── Class balance + water_weight ───────────────────────────────────────
    wf_train = float((y_train == 1).mean())
    wf_val = float((y_val == 1).mean())
    wf_test = float((y_test == 1).mean())
    if wf_train > 0:
        water_weight = float(np.clip((1 - wf_train) / wf_train, 2, 15))
    else:
        water_weight = 8.0
    print(f"Train water {wf_train*100:.2f}% | Val water {wf_val*100:.2f}% "
          f"| water_weight={water_weight:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, yd) in zip(axes, [("Train", y_train), ("Val", y_val)]):
        ax.bar(["Land", "Water"], [(yd == 0).sum(), (yd == 1).sum()],
               color=["#8B7355", "#1E90FF"])
        ax.set_title(f"{name} class distribution")
        ax.set_ylabel("Pixel count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_balance.png"), dpi=120)
    plt.close()

    # ── Datasets ───────────────────────────────────────────────────────────
    @tf.function
    def augment(x, y):
        if tf.random.uniform(()) > 0.5:
            x = tf.image.flip_left_right(x)
            y = tf.image.flip_left_right(y[..., tf.newaxis])[..., 0]
        if tf.random.uniform(()) > 0.5:
            x = tf.image.flip_up_down(x)
            y = tf.image.flip_up_down(y[..., tf.newaxis])[..., 0]
        k = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        x = tf.image.rot90(x, k=k)
        y = tf.image.rot90(y[..., tf.newaxis], k=k)[..., 0]
        brightness_delta = tf.random.uniform((), -0.08, 0.08)
        x = tf.clip_by_value(x + brightness_delta, 0.0, 1.0)
        noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.01)
        x = tf.clip_by_value(x + noise, 0.0, 1.0)
        return x, y

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=1000, seed=SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # ── Build + compile ────────────────────────────────────────────────────
    model = build_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, 5))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=make_combined_loss(water_weight),
        metrics=[iou_metric, f1_metric],
    )
    print(f"Trainable params: {model.count_params():,}")

    ckpt_path = os.path.join(out_dir, "disastershield_best.keras")
    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True,
                        monitor="val_iou_metric", mode="max", verbose=1),
        EarlyStopping(patience=12, monitor="val_iou_metric", mode="max",
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_iou_metric", factor=0.5, patience=5,
                          min_lr=1e-6, mode="max", verbose=1),
    ]

    t0 = time.time()
    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=epochs, callbacks=callbacks, verbose=2)
    train_seconds = time.time() - t0

    # ── History CSV ────────────────────────────────────────────────────────
    hist = history.history
    epochs_run = len(hist["loss"])
    with open(os.path.join(out_dir, "training_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = list(hist.keys())
        w.writerow(["epoch"] + keys)
        for e in range(epochs_run):
            w.writerow([e + 1] + [hist[k][e] for k in keys])

    # ── Evaluate on held-out test (best weights restored) ──────────────────
    test_results = model.evaluate(test_dataset, verbose=2)
    test_loss, test_iou, test_f1 = (float(test_results[0]),
                                    float(test_results[1]),
                                    float(test_results[2]))

    final_train_iou = float(hist["iou_metric"][-1])
    final_val_iou = float(hist["val_iou_metric"][-1])
    best_val_iou = float(max(hist["val_iou_metric"]))
    train_val_gap = final_train_iou - final_val_iou

    # ── Inference timing: all test patches in ONE batch ────────────────────
    _ = model.predict(X_test, batch_size=X_test.shape[0], verbose=0)  # warm-up
    t_inf = time.time()
    test_preds = model.predict(X_test, batch_size=X_test.shape[0], verbose=0)
    inference_seconds_all_test = time.time() - t_inf

    # ── Per-patch test IoU -> worst cases ──────────────────────────────────
    test_pred_bin = (test_preds[..., 1] > 0.5).astype(np.int32)
    patch_ious = per_patch_iou(y_test, test_pred_bin)
    worst_idx = np.argsort(patch_ious)[:3]
    worst = [{"patch_index": int(i),
              "iou": float(patch_ious[i]),
              "water_frac_true": float((y_test[i] == 1).mean())}
             for i in worst_idx]

    # ── Prediction figure (validation, notebook cell 12 style) ─────────────
    n_samples = min(4, X_val.shape[0])
    vpred = model.predict(X_val[:n_samples], verbose=0)
    vpred_bin = (vpred[..., 1] > 0.5).astype(int)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    axes[0, 0].set_title("RGB (B4-B3-B2)")
    axes[0, 1].set_title("Ground Truth")
    axes[0, 2].set_title("Prediction")
    for i in range(n_samples):
        rgb_vis = np.clip(X_val[i][:, :, [2, 1, 0]], 0, 1)
        axes[i, 0].imshow(rgb_vis); axes[i, 0].axis("off")
        axes[i, 1].imshow(y_val[i], cmap="Blues", vmin=0, vmax=1); axes[i, 1].axis("off")
        axes[i, 2].imshow(vpred_bin[i], cmap="Blues", vmin=0, vmax=1); axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "predictions.png"), dpi=150)
    plt.close()

    # ── metrics.json ───────────────────────────────────────────────────────
    metrics = {
        "mode": "smoke" if args.smoke else "full",
        "utc_timestamp": ts,
        "seed": SEED,
        "tensorflow_version": tf.__version__,
        "gpu_used": len(tf.config.list_physical_devices("GPU")) > 0,
        "data_dir": DATA_DIR,
        "spatial": pixel_info,
        "patch_counts_per_year": {str(k): v for k, v in per_year_counts.items()},
        "n_train_patches_used": int(X_train.shape[0]),
        "n_val_patches": int(X_val.shape[0]),
        "n_test_patches": int(X_test.shape[0]),
        "water_fraction": {"train": wf_train, "val": wf_val, "test": wf_test},
        "water_weight": water_weight,
        "epochs_requested": epochs,
        "epochs_run": epochs_run,
        "train_seconds": train_seconds,
        "inference_seconds_all_test_patches": inference_seconds_all_test,
        "n_test_patches_inference": int(X_test.shape[0]),
        "final_train_iou": final_train_iou,
        "final_val_iou": final_val_iou,
        "best_val_iou": best_val_iou,
        "train_val_gap": train_val_gap,
        "test_iou": test_iou,
        "test_f1": test_f1,
        "test_loss": test_loss,
        "worst_test_patches": worst,
        "output_files": {
            "metrics_json": os.path.join(out_dir, "metrics.json"),
            "training_history_csv": os.path.join(out_dir, "training_history.csv"),
            "predictions_png": os.path.join(out_dir, "predictions.png"),
            "class_balance_png": os.path.join(out_dir, "class_balance.png"),
            "model_keras": ckpt_path,
        },
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n==== DONE ====")
    print("TEST IoU:", test_iou, "| TEST F1:", test_f1, "| test loss:", test_loss)
    print("metrics.json:", os.path.join(out_dir, "metrics.json"))
    print("RESULT_DIR:", out_dir)


if __name__ == "__main__":
    main()
