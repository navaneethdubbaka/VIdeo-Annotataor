# Accuracy Metrics

This document provides quantitative accuracy bounds for each annotation type produced by the VLA Data Collection Annotator. The values represent expected performance under typical operating conditions (indoor lighting, camera at operator eye level, hand fully visible). Conditions that degrade performance — occlusion, poor lighting, extreme hand angles — are noted for each annotation type.

---

## How to Read This Document

Each section provides:
- **Primary metric(s):** the standard measure of accuracy for that annotation type
- **Expected bounds:** values derived from published MediaPipe benchmarks and analytical error propagation
- **Quality thresholds:** a three-tier classification (Good / Acceptable / Poor) to support pass/fail decisions

Measured values from specific evaluation runs should be recorded in the **Measured** column; those cells are left blank here as placeholders.

---

## Summary Table

| Annotation | Primary metric | Good | Acceptable | Poor | Measured |
|-----------|---------------|------|-----------|------|---------|
| Raw landmarks XY | MAE (mm) | < 10 | 10–20 | > 20 | |
| Raw landmarks Z (depth) | MAE (mm) | < 25 | 25–40 | > 40 | |
| PCK @ 20 mm | % correct | > 85% | 70–85% | < 70% | |
| Palm roll / pitch | MAE (°) | < 8 | 8–15 | > 15 | |
| Palm yaw | MAE (°) | < 12 | 12–20 | > 20 | |
| End-effector XY | MAE (mm) | < 8 | 8–20 | > 20 | |
| End-effector Z | MAE (mm) | < 25 | 25–45 | > 45 | |
| MCP joint angles | MAE (°) | < 10 | 10–18 | > 18 | |
| PIP joint angles | MAE (°) | < 15 | 15–22 | > 22 | |
| DIP joint angles | MAE (°) | < 18 | 18–25 | > 25 | |
| Grasp type (7-class) | Macro F1 | > 0.80 | 0.65–0.80 | < 0.65 | |
| Finger aperture | MAE (mm) | < 6 | 6–12 | > 12 | |
| Contact state | Accuracy | > 88% | 75–88% | < 75% | |
| EE speed | MAE (m/s) | < 0.10 | 0.10–0.25 | > 0.25 | |
| EE acceleration | MAE (m/s²) | < 2 | 2–6 | > 6 | |

---

## Metric Definitions

| Term | Meaning |
|------|---------|
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and ground-truth values |
| **RMSE** (Root Mean Squared Error) | Square root of the mean squared difference; penalises large errors more than MAE |
| **PCK @ N mm** (Percentage Correct Keypoints) | Fraction of landmark predictions that fall within N millimetres of ground truth |
| **F1 score (macro)** | Harmonic mean of precision and recall, averaged equally across all classes regardless of frequency |
| **Accuracy** | Fraction of frames where the predicted class matches the ground-truth class |

---

## 1. Raw 3D Landmarks

### Published baseline

These figures come from Google's published MediaPipe Hand Landmarker benchmarks evaluated on the Interhand 2.6M dataset.

| Metric | Published value | Condition |
|--------|----------------|-----------|
| PCK @ 20 mm | 84–92% | Good lighting, frontal view, full visibility |
| Median landmark error | 8–12 mm | XY plane |
| Fingertip error | 10–18 mm | XY plane |
| Depth (Z) error | 20–30 mm | Monocular camera — estimated from appearance |
| Wrist error | 5–8 mm | Most stable landmark |

Depth accuracy is fundamentally limited by monocular estimation: the Z coordinate is inferred from learned shape priors, not measured geometrically. This means depth errors are typically 2–3× larger than the in-plane (XY) errors, and they cannot be reduced without adding a depth sensor.

### Expected error by landmark group

| Group | Landmarks | XY MAE | Z MAE | Notes |
|-------|-----------|--------|-------|-------|
| Wrist | 0 | 5–8 mm | 15–25 mm | Most stable; anchors the hand frame |
| Knuckles (MCP) | 5, 9, 13, 17 | 7–12 mm | 18–28 mm | Good in-plane accuracy |
| PIP joints | 6, 10, 14, 18 | 8–14 mm | 20–30 mm | |
| DIP joints | 7, 11, 15, 19 | 9–15 mm | 22–32 mm | |
| Fingertips | 4, 8, 12, 16, 20 | 10–18 mm | 25–40 mm | Worst under occlusion |

### Common causes of degraded accuracy

- Partial hand occlusion (fingers behind objects or the other hand)
- Extreme orientations (palm facing completely away from camera)
- Hand at the edge of the video frame
- Fast motion causing motion blur (at 30 fps: > ~0.5 m/s)

---

## 2. Palm Orientation (Roll / Pitch / Yaw)

### Error propagation

A 10 mm error in the middle MCP position (lm 9) — the landmark that defines the Y-axis of the palm frame — translates to an angular error of approximately:

> arctan(10 mm / 80 mm) ≈ 7°

where 80 mm is a typical wrist-to-MCP distance. This gives a practical upper bound for orientation error under typical landmark noise.

Yaw is the least accurate of the three angles because it is most sensitive to depth (Z) errors, which are the largest component of landmark uncertainty.

### Expected accuracy

| Angle | MAE | RMSE | Notes |
|-------|-----|------|-------|
| Roll | 5–8° | 7–12° | Uses stable wrist and middle MCP landmarks |
| Pitch | 5–8° | 7–12° | Moderately reliable |
| Yaw | 8–15° | 12–20° | Sensitive to depth error in the Z axis |

### Gimbal lock zone

When `|pitch_deg|` exceeds 80°, yaw values become unreliable (set to 0° by construction) and roll absorbs the full rotation. The effective orientation accuracy in this region is undefined. Frames meeting this criterion should be excluded from quantitative evaluation of orientation.

---

## 3. Height-Normalised Coordinates

Normalisation is a deterministic multiplication; it adds no new errors. All landmark errors scale by the same factor:

> MAE_normalised = MAE_raw × (robot height ÷ operator height)

For a scale factor of 0.73 (120 cm robot, 165 cm operator), a raw XY MAE of 10 mm becomes a normalised MAE of approximately 7.3 mm.

**Systematic limitation:** The isotropic scale does not account for differences in working height between the operator's table and the robot's workspace. This can introduce a systematic positional offset of 2–5 cm for height ratios in the range 0.6–1.0, depending on the physical setup.

---

## 4. End-Effector Position

The end-effector position is the wrist (landmark 0). Its accuracy is therefore identical to the wrist row in Section 1.

| Axis | MAE (Good) | MAE (Acceptable) | MAE (Poor) |
|------|-----------|-----------------|-----------|
| XY plane | < 8 mm | 8–20 mm | > 20 mm |
| Z (depth) | < 25 mm | 25–45 mm | > 45 mm |

---

## 5. Finger Joint Angles

### Error propagation

Joint angle error depends on the accuracy of the three landmarks that define it and on the physical length of the finger segment. For a PIP segment of ~25 mm and a 10 mm landmark error:

> arctan(10 / 25) ≈ 22° (worst case)

In practice, MediaPipe noise is correlated across adjacent landmarks (they move together rather than independently), which reduces the effective error considerably below this theoretical worst case.

### Expected accuracy by joint type

| Joint type | MAE | RMSE | Notes |
|-----------|-----|------|-------|
| MCP (all fingers) | 5–10° | 8–15° | Uses the wrist as the proximal anchor; relatively stable |
| PIP (all fingers) | 8–15° | 12–20° | Mid-finger; moderate sensitivity to noise |
| IP (thumb) | 8–15° | 12–20° | Comparable to PIP |
| DIP (all fingers) | 10–20° | 15–25° | Shortest segments; highest noise sensitivity |

Higher DIP error compared to MCP error is expected and normal. For tasks where fine fingertip detail matters (e.g. precision assembly), consider down-weighting DIP annotations or supplementing with a glove sensor.

### Per-joint measured values (to be completed)

| Column | MAE (°) | RMSE (°) | R² |
|--------|---------|---------|---|
| `thumb_mcp` | | | |
| `thumb_ip` | | | |
| `idx_mcp` | | | |
| `idx_pip` | | | |
| `idx_dip` | | | |
| `mid_mcp` | | | |
| `mid_pip` | | | |
| `mid_dip` | | | |
| `ring_mcp` | | | |
| `ring_pip` | | | |
| `ring_dip` | | | |
| `pinky_mcp` | | | |
| `pinky_pip` | | | |
| `pinky_dip` | | | |

---

## 6. Grasp Classification

### Overall expected accuracy

| Metric | Expected | Condition |
|--------|---------|-----------|
| Overall accuracy | 75–85% | Balanced class distribution, good lighting |
| Macro F1 | 0.70–0.80 | Equal weight across all 7 classes |
| Weighted F1 | 0.78–0.88 | Weighted by class frequency |

### Per-class expected performance

| Class | Expected precision | Expected recall | Main confusion |
|-------|------------------|----------------|---------------|
| `open` | 0.85–0.95 | 0.85–0.95 | Rarely confused |
| `power` | 0.80–0.92 | 0.80–0.92 | Rarely confused |
| `pinch` | 0.70–0.85 | 0.70–0.85 | With `tripod` |
| `tripod` | 0.65–0.80 | 0.65–0.80 | With `pinch` |
| `lateral` | 0.55–0.75 | 0.60–0.78 | With `hook` |
| `hook` | 0.55–0.73 | 0.55–0.72 | With `lateral` |
| `unknown` | — | — | Transition frames |

The two most common confusions are:
- **Pinch vs. tripod:** the middle finger curl threshold is shared at 0.70; near this boundary the class is ambiguous
- **Lateral vs. hook:** both have the same finger curl pattern; they differ only by whether the thumb tip is within 3 cm of the index MCP, which is sensitive to small positional errors

### Confusion matrix (to be completed)

| Ground truth → Predicted | open | pinch | tripod | power | lateral | hook | unknown |
|--------------------------|------|-------|--------|-------|---------|------|---------|
| **open** | | | | | | | |
| **pinch** | | | | | | | |
| **tripod** | | | | | | | |
| **power** | | | | | | | |
| **lateral** | | | | | | | |
| **hook** | | | | | | | |
| **unknown** | | | | | | | |

### Finger aperture

The aperture error is propagated from fingertip landmark errors. With a per-tip error of ~10–18 mm, the two-tip aperture error adds approximately in quadrature:

> σ_aperture ≈ √(σ_tip² + σ_tip²) = σ_tip × √2 ≈ 14–25 mm worst case

Expected MAE in practice: **3–6 mm** (correlated noise reduces this below the worst case).

### Contact state

| Metric | Expected |
|--------|---------|
| Overall accuracy | 80–90% |
| Main error mode | `partial` ↔ `closed` near the 0.02 m threshold |

---

## 7. Velocity and Acceleration

### Noise amplification by finite differencing

Finite differencing amplifies position noise. For a landmark position noise of σ_pos and frame interval Δt:

| Derivative | Noise formula | Example (30 fps, skip=1, σ_pos = 10 mm) |
|-----------|--------------|----------------------------------------|
| Velocity | σ_pos × √2 / Δt | ~0.43 m/s |
| Acceleration | σ_pos × 2 / Δt² | ~18 m/s² |

These are the theoretical noise floors for **unsmoothed** output. Increasing the frame stride (e.g. `--skip 3`) reduces velocity noise approximately 3-fold and acceleration noise approximately 9-fold, at the cost of lower temporal resolution.

### Expected accuracy with post-processing smoothing

| Processing | Speed MAE | Acceleration MAE |
|-----------|-----------|-----------------|
| Raw (no smoothing) | 0.10–0.40 m/s | 2–15 m/s² |
| 3-frame median filter | 0.05–0.15 m/s | 1–5 m/s² |
| 5-frame Gaussian filter | 0.03–0.10 m/s | 0.5–3 m/s² |

### Observed values from sample data

From the sewing-machine task (1,322 annotated frames):

| Metric | Observed |
|--------|---------|
| Maximum speed | ~0.19 m/s |
| Maximum acceleration | ~3.3 m/s² |
| Speed during steady hold | < 0.02 m/s |
| Acceleration at direction reversal | 1.5–3.3 m/s² |

These values are physically consistent with slow, controlled fabric manipulation.

---

## 8. Task Labels

The equal-segmentation algorithm has zero algorithmic error when task transitions are uniformly distributed across the video. For non-uniform tasks, the worst-case step-boundary timing error is:

> max boundary error = step duration ÷ 2

For a 30-second video with 3 steps (10 s each): the boundary may be off by up to ±5 seconds at either transition. The semantic accuracy of the text labels depends entirely on the operator's input at recording time.
