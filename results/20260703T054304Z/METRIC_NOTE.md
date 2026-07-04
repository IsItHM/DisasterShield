# METRIC NOTE — metrics in this folder are NON-CANONICAL

The `test_iou` / `test_f1` fields in `metrics.json` here are **Keras batch-averaged**
metrics (the mean of the per-batch IoU/F1 that `model.evaluate` reports over the test
batches). IoU is nonlinear, so the mean of per-batch IoUs is **not** the pooled pixel-level
IoU — batches with few flood pixels drag the average down. These fields are therefore
**NON-CANONICAL** and must not be compared against the global pooled numbers.

- This folder is the **v2** (4-channel S1 change-detection) run — the **final model**. Its
  batch-averaged `test_iou` = **0.6052** (`metrics.json`) is on the batch-averaged scale; the
  same weights score global pooled test IoU **0.7216 / F1 0.8383 @ threshold 0.65**.

Canonical numbers (global pooled pixel-level IoU/F1, val-tuned threshold) are in
`results/20260703T165223Z/threshold_fair_v3.csv`.

**Canonical final model: v2, test IoU 0.7216 / F1 0.8383 @ threshold 0.65.**
Weights: `results/20260703T054304Z/feni_unet_best.keras`.
