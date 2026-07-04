"""
DisasterShield Phase 2 - Feni 10 m flood segmentation (S1 change detection, v2 window-matched).

Inputs: data/processed/feni_{X,y}_{train,val,test}_v2.npy
Channels (4): [VV_flood, VH_flood, VV_pre, VH_pre] in dB. Input shape (64,64,4). NO NDWI.

Normalization: per-channel mean/std computed on TRAIN split ONLY (radar dB -> standardize;
NO /10000, NO clip to [0,1]). Stats saved to run metrics.json AND data/processed/norm_stats_v2.json.
Augmentation: H/V flips + 90 deg rotations + Gaussian noise (stddev 0.05 in standardized units).
NO brightness jitter (radar is calibrated).

Same U-Net / Dice+weighted-CE loss / IoU+F1 metrics / ModelCheckpoint+EarlyStopping(12,restore_best)
+ReduceLROnPlateau / batch 16 / --smoke,--epochs. Outputs -> results/<UTC-timestamp>/.
"""
import os, json, time, random, argparse, datetime, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROC = r"d:\Hamim\DisasterShield\data\processed"
# 4-channel (v1/v2) or 6-channel physics-informed (v3) input; resolved from the data shape.
CHANNELS_4 = ["VV_flood", "VH_flood", "VV_pre", "VH_pre"]
CHANNELS_6 = ["VV_flood", "VH_flood", "VV_pre", "VH_pre", "dVV", "dVH"]
CHANNELS = CHANNELS_4  # default; overridden in main() from X_train.shape[-1]
PATCH_SIZE = 64
BATCH_SIZE = 16
L2_REG = 1e-4
NOISE_STD = 0.05  # standardized units


def load_split(tag, sp):
    X = np.load(os.path.join(PROC, f"feni_X_{sp}_{tag}.npy")).astype(np.float32)
    y = np.load(os.path.join(PROC, f"feni_y_{sp}_{tag}.npy")).astype(np.float32)
    return X, y


# ─── Model (identical architecture to Phase 1) ──────────────────────────────
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


def build_unet(input_shape=(64, 64, 4), num_classes=2):
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
    return Model(inputs, outputs, name="DisasterShield_Feni_UNet")


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
    out = []
    for i in range(y_true.shape[0]):
        yt = (y_true[i] == 1).astype(np.float32)
        yp = pred_bin[i].astype(np.float32)
        inter = float((yt * yp).sum())
        union = float(yt.sum() + yp.sum() - inter)
        out.append((inter + 1e-6) / (union + 1e-6))
    return np.array(out)


def main():
    global CHANNELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke: cap train to 200 patches, force 2 epochs.")
    args = ap.parse_args()
    epochs = 2 if args.smoke else args.epochs
    TAG = args.tag

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(r"d:\Hamim\DisasterShield\results", ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Mode: {'SMOKE' if args.smoke else 'FULL'} | tag={TAG} | epochs={epochs}")
    print("TensorFlow:", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

    # ── Load data ──────────────────────────────────────────────────────────
    X_train, y_train = load_split(TAG, "train")
    X_val, y_val = load_split(TAG, "val")
    X_test, y_test = load_split(TAG, "test")
    print(f"Loaded  train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    # Resolve channel names from the data (4-ch v1/v2 or 6-ch physics-informed v3).
    n_ch = X_train.shape[-1]
    CHANNELS = CHANNELS_6 if n_ch == 6 else CHANNELS_4
    assert len(CHANNELS) == n_ch, f"channel mismatch: data has {n_ch}, names {CHANNELS}"
    print(f"Input channels ({n_ch}): {CHANNELS}")

    # ── Normalization: per-channel mean/std on FULL TRAIN only ──────────────
    # A tiny fraction of pixels are NaN (SAR invalid pixels inside otherwise-valid
    # patches). Compute stats ignoring NaN, then fill NaN with the channel mean so
    # standardized value == 0 (neutral) and no NaN leaks into training.
    flat = X_train.reshape(-1, X_train.shape[-1])
    n_nonfinite_train = int((~np.isfinite(flat)).sum())
    means = np.nanmean(flat, axis=0)
    stds = np.nanstd(flat, axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    norm_stats = {"channels": CHANNELS, "tag": TAG, "seed": SEED,
                  "computed_on": "train split only (nanmean/nanstd)",
                  "mean": [float(m) for m in means],
                  "std": [float(s) for s in stds],
                  "nonfinite_pixels_filled_with_mean": True,
                  "n_nonfinite_train_channel_values": n_nonfinite_train}
    with open(os.path.join(PROC, f"norm_stats_{TAG}.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)
    print("Norm means:", norm_stats["mean"])
    print("Norm stds :", norm_stats["std"])
    print("Non-finite train channel-values filled:", n_nonfinite_train)

    def standardize(X):
        X = np.where(np.isfinite(X), X, means).astype(np.float32)  # fill NaN with channel mean
        return ((X - means) / stds).astype(np.float32)
    X_train = standardize(X_train)
    X_val = standardize(X_val)
    X_test = standardize(X_test)

    # ── Smoke cap (AFTER stats computed on full train) ──────────────────────
    if args.smoke and X_train.shape[0] > 200:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(X_train.shape[0], 200, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"  [smoke] train capped to {X_train.shape[0]} patches")

    # ── Class balance + water_weight (from train flood fraction) ────────────
    wf_train = float((y_train == 1).mean())
    wf_val = float((y_val == 1).mean())
    wf_test = float((y_test == 1).mean())
    water_weight = float(np.clip((1 - wf_train) / wf_train, 2, 15)) if wf_train > 0 else 8.0
    print(f"flood frac  train {wf_train*100:.2f}%  val {wf_val*100:.2f}%  test {wf_test*100:.2f}%"
          f" | water_weight={water_weight:.3f}")

    # ── Augmentation: flips + rot90 + Gaussian noise; NO brightness ─────────
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
        x = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=NOISE_STD)  # standardized units
        return x, y

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=1000, seed=SEED)
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = build_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, len(CHANNELS)))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=make_combined_loss(water_weight),
                  metrics=[iou_metric, f1_metric])
    print(f"Trainable params: {model.count_params():,}")

    ckpt_path = os.path.join(out_dir, "feni_unet_best.keras")
    last_ckpt_path = os.path.join(out_dir, "feni_unet_last.keras")
    csv_path = os.path.join(out_dir, "training_history.csv")
    callbacks = [
        # best model: saved only when val-IoU improves
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_iou_metric", mode="max", verbose=1),
        # last model: saved EVERY epoch -> an interrupted run loses at most one epoch
        ModelCheckpoint(last_ckpt_path, save_best_only=False, save_freq="epoch", verbose=0),
        EarlyStopping(patience=12, monitor="val_iou_metric", mode="max", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_iou_metric", factor=0.5, patience=5, min_lr=1e-6, mode="max", verbose=1),
        # append history to disk AFTER EVERY EPOCH (partial history survives interruption)
        CSVLogger(csv_path, append=True),
    ]

    t0 = time.time()
    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=epochs, callbacks=callbacks, verbose=2)
    train_seconds = time.time() - t0

    hist = history.history
    epochs_run = len(hist["loss"])
    sec_per_epoch = train_seconds / max(epochs_run, 1)
    # training_history.csv is written per-epoch by CSVLogger (survives interruption).

    test_results = model.evaluate(test_dataset, verbose=2)
    test_loss, test_iou, test_f1 = float(test_results[0]), float(test_results[1]), float(test_results[2])

    final_train_iou = float(hist["iou_metric"][-1])
    final_val_iou = float(hist["val_iou_metric"][-1])
    best_val_iou = float(max(hist["val_iou_metric"]))
    train_val_gap = final_train_iou - final_val_iou

    # ── Inference timing: all test patches ─────────────────────────────────
    _ = model.predict(X_test[:BATCH_SIZE], verbose=0)  # warm-up
    t_inf = time.time()
    test_preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    inference_seconds_all_test = time.time() - t_inf

    test_pred_bin = (test_preds[..., 1] > 0.5).astype(np.int32)
    patch_ious = per_patch_iou(y_test, test_pred_bin)
    worst_idx = np.argsort(patch_ious)[:3]
    worst = [{"patch_index": int(i), "iou": float(patch_ious[i]),
              "flood_frac_true": float((y_test[i] == 1).mean()),
              "pred_flood_frac": float(test_pred_bin[i].mean())} for i in worst_idx]

    # ── Prediction figure: VV_flood(std) | GT | Pred for 4 val samples ─────
    n_samples = min(4, X_val.shape[0])
    vpred = model.predict(X_val[:n_samples], verbose=0)
    vpred_bin = (vpred[..., 1] > 0.5).astype(int)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    axes[0, 0].set_title("VV_flood (standardized)")
    axes[0, 1].set_title("Ground Truth")
    axes[0, 2].set_title("Prediction")
    for i in range(n_samples):
        axes[i, 0].imshow(X_val[i][:, :, 0], cmap="gray"); axes[i, 0].axis("off")
        axes[i, 1].imshow(y_val[i], cmap="Blues", vmin=0, vmax=1); axes[i, 1].axis("off")
        axes[i, 2].imshow(vpred_bin[i], cmap="Blues", vmin=0, vmax=1); axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "predictions.png"), dpi=150)
    plt.close()

    known_label_noise = (
        "v2 flood composite (Aug 18-26 2024) matches the UNOSAT S1_20240818_20240826 label window "
        "exactly, so the v1 composite-window caveat (v1 spans Aug 20-Sep 5, wider than the label) does "
        "NOT apply to this run. Residual negative-class noise: ~92.45% of pre-flood permanent water "
        "(VV<-16 dB, Step 2c) falls outside UNOSAT polygons, i.e. permanent open water is genuinely "
        "non-flood in the negatives."
    )

    metrics = {
        "mode": "smoke" if args.smoke else "full",
        "phase": "phase2_feni", "tag": TAG,
        "utc_timestamp": ts, "seed": SEED,
        "tensorflow_version": tf.__version__,
        "gpu_used": len(tf.config.list_physical_devices("GPU")) > 0,
        "channels": CHANNELS, "input_shape": [PATCH_SIZE, PATCH_SIZE, len(CHANNELS)],
        "flood_input": f"Feni_S1_Flood_{'18to26Aug2024' if TAG in ('v2', 'v3') else 'Aug2024'}_10m.tif",
        "normalization": norm_stats,
        "n_train_patches_used": int(X_train.shape[0]),
        "n_val_patches": int(X_val.shape[0]),
        "n_test_patches": int(X_test.shape[0]),
        "flood_fraction": {"train": wf_train, "val": wf_val, "test": wf_test},
        "water_weight": water_weight,
        "epochs_requested": epochs, "epochs_run": epochs_run,
        "train_seconds": train_seconds, "seconds_per_epoch": sec_per_epoch,
        "inference_seconds_all_test_patches": inference_seconds_all_test,
        "n_test_patches_inference": int(X_test.shape[0]),
        "final_train_iou": final_train_iou, "final_val_iou": final_val_iou,
        "best_val_iou": best_val_iou, "train_val_gap": train_val_gap,
        "test_iou": test_iou, "test_f1": test_f1, "test_loss": test_loss,
        "worst_test_patches": worst,
        "known_label_noise": known_label_noise,
        "output_files": {
            "metrics_json": os.path.join(out_dir, "metrics.json"),
            "training_history_csv": os.path.join(out_dir, "training_history.csv"),
            "predictions_png": os.path.join(out_dir, "predictions.png"),
            "model_keras": ckpt_path,
            "model_last_keras": last_ckpt_path,
            "norm_stats_json": os.path.join(PROC, f"norm_stats_{TAG}.json"),
        },
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n==== DONE ====")
    print(f"epochs_run={epochs_run} sec/epoch={sec_per_epoch:.1f}")
    print("TEST IoU:", test_iou, "| TEST F1:", test_f1, "| test loss:", test_loss)
    print("RESULT_DIR:", out_dir)


if __name__ == "__main__":
    main()
