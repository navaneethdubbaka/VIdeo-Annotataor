# Evaluation Methodology

This document describes the methodology used to assess the accuracy of annotations produced by the VLA Data Collection Annotator. It explains how ground truth is collected, how comparisons between predicted and ground-truth values are structured, and how statistical rigour is maintained across different annotation types.

---

## Overview

Because the annotator derives measurements from monocular video using a learned pose-estimation model, accuracy must be assessed empirically rather than guaranteed analytically. The evaluation process follows three broad phases:

1. **Ground-truth collection** — gathering an independent, high-accuracy reference for the same hand motion
2. **Comparison** — aligning the annotator's outputs with the ground truth in time and computing error metrics
3. **Reporting** — aggregating metrics across frames and sequences, with appropriate statistical treatment

---

## 1. Ground Truth Collection Methods

Four methods are available, listed from highest to lowest accuracy:

### 1.1 Optical Motion Capture

**Spatial accuracy:** ±0.3–1.0 mm | **Temporal accuracy:** sub-millisecond

Optical motion capture (systems such as OptiTrack or Vicon) tracks retroreflective markers placed on the hand to sub-millimetre precision. It is the gold standard for evaluating 3D landmark positions, joint angles, and velocity.

**Marker placement:** Ten markers are attached to the hand dorsum, positioned at the wrist, the four MCP knuckles, and the five fingertips. This provides direct ground truth for the most structurally important landmarks.

**Synchronisation:** A hardware trigger fires a flash LED visible in both the RGB camera frame and the motion capture recording. Both streams are aligned to the frame of the flash, achieving synchronisation within one video frame (≈33 ms at 30 fps).

**Limitation:** Markers on the palm dorsum cannot capture volar (palm-side) landmarks directly when the hand faces the camera. Positions for volar landmarks may be inferred from the kinematic model, introducing an additional ~3–8 mm uncertainty for those points.

---

### 1.2 Depth Camera Cross-Validation

**Spatial accuracy:** ±5–15 mm at 0.3–1.5 m range

A structured-light or time-of-flight depth camera (such as Intel RealSense D435 or Microsoft Azure Kinect) provides 3D wrist position without markers. The wrist is localised in 2D and its depth is read from the depth map, then reprojected to 3D world coordinates using the camera's calibrated intrinsics.

This method is well suited for validating end-effector position (`ee_xyz`) and detecting systematic biases (such as a constant Z offset). It is treated as a noisy reference rather than a true gold standard, and its own measurement uncertainty (±5–15 mm) must be accounted for in the analysis.

---

### 1.3 Inertial Measurement Unit (IMU) Cross-Validation

**Orientation accuracy:** ±1–3° | **Acceleration accuracy:** ±0.05 m/s²

A 9-DOF IMU (accelerometer + gyroscope + magnetometer) mounted on the hand dorsum, aligned with the palm normal, provides independent orientation measurements. A sensor-fusion filter (Madgwick or Mahony) integrates all three sensing axes into a quaternion orientation that is then converted to ZYX Euler angles for direct comparison with `roll_deg`, `pitch_deg`, `yaw_deg`.

The same IMU provides linear acceleration that can be compared with `ee_accel`, after subtracting the 9.81 m/s² gravitational component.

**Limitation:** Orientation drift accumulates over time without periodic magnetometer recalibration. Evaluation sequences should be kept under 5 minutes between drift corrections.

---

### 1.4 Manual Frame-by-Frame Labelling

**Spatial accuracy:** ±10–25 mm (after 2D-to-3D projection) | **Classification accuracy:** human-level

A human annotator reviews the video frame by frame and records:
- The 2D pixel position of the wrist (later projected to 3D using depth)
- The grasp type (one of the seven defined classes)
- The contact state

This method requires no specialised hardware but is labour-intensive (~2–5 hours per 1,000 frames). It is the primary method for evaluating grasp classification and contact state accuracy.

**Two-annotator protocol:** Each sequence is labelled by at least two independent annotators. Agreement is measured using Cohen's kappa coefficient:

| Annotation type | Target κ | Minimum acceptable κ |
|----------------|---------|---------------------|
| Grasp type (7-class) | > 0.75 | > 0.60 |
| Contact state (3-class) | > 0.80 | > 0.65 |
| Micro-step boundary (±1 s) | > 0.70 | > 0.55 |

Disagreements are resolved by consensus discussion or by a third annotator as tiebreaker.

---

## 2. Evaluation Datasets

Where recording custom evaluation sequences is not feasible, the following publicly available datasets provide pre-existing ground truth compatible with the annotator's output format:

| Dataset | Size | Ground truth type | Suitable for evaluating |
|---------|------|-------------------|------------------------|
| Interhand 2.6M | 2.6M frames | 2D + 3D keypoints | Raw landmarks |
| FreiHAND | 32K images | 3D MANO model fit | Landmarks, orientation |
| HO-3D v3 | 77K frames | MANO + object pose | Landmarks, grasp type |
| GRAB | 1.6M frames | MANO + contact labels | Grasp, contact state, velocity |
| FPHA (First-Person Hand Action) | 105K frames | Glove-sensor joint angles | Joint angles, task labels |

For landmark and orientation accuracy, **Interhand 2.6M** or **FreiHAND** are recommended. For grasp classification, **HO-3D** or **GRAB** provide the most relevant labels.

---

## 3. Recording Protocol for Custom Evaluation Sequences

### Camera setup

- Position the camera at the operator's eye level, angled down 15–30° toward the workspace
- Maintain ≥ 500 lux illumination with diffuse, non-directional light sources
- Use a plain, neutral-coloured background (dark grey or green felt works well)
- Record at 1080p resolution and ≥ 30 fps

### Coverage requirements

| Session | Grasp types included | Minimum duration |
|---------|---------------------|-----------------|
| Session 1 | open, pinch, tripod | 3 × 30 s + transitions |
| Session 2 | power, lateral, hook | 3 × 30 s + transitions |
| Session 3 | Naturalistic manipulation task | ≥ 2 minutes continuous |

For generalisability, record at least two operators (covering different hand sizes and skin tones) and both left and right hands where possible.

---

## 4. Evaluation Protocols

### 4.1 Landmark Position Accuracy

**Ground truth required:** Motion capture or depth camera wrist positions

1. Load the annotator's CSV output and the ground-truth position file
2. Align the two streams by matching timestamps (nearest-neighbour within a 50 ms tolerance)
3. For each matched frame and each landmark, compute the Euclidean distance between the predicted and ground-truth positions
4. Report:
   - **MAE** (mean absolute error) in mm, separately for the XY plane and for Z
   - **RMSE** in mm
   - **PCK @ 10 mm, 20 mm, 30 mm** (percentage of landmark predictions within each distance threshold)
5. Aggregate results by landmark group (wrist, knuckles, fingertips)

---

### 4.2 Palm Orientation Accuracy

**Ground truth required:** IMU-derived ZYX Euler angles, or mocap-derived orientation

1. Convert ground-truth orientation to the same ZYX Euler convention used by the annotator
2. Align streams by timestamp
3. Exclude frames where `|pitch_deg| > 80°` (gimbal lock region)
4. For each angle (roll, pitch, yaw), compute the signed angular difference with wraparound correction (the shortest path around the ±180° boundary)
5. Report MAE and RMSE per angle, and the total rotation error as a single quaternion magnitude

---

### 4.3 Finger Joint Angle Accuracy

**Ground truth required:** MANO model fit to motion-capture landmarks, or FPHA glove-sensor angles

1. Apply the same geometric formula used by the annotator (dot-product angle at vertex B) to the ground-truth landmark positions, using the identical triplet table from the generation stage. This produces reference angle values using the same definition — eliminating any formula bias.
2. Align annotator and reference angle sequences by timestamp
3. For each of the 14 joints, compute the signed angle difference
4. Report MAE, RMSE, and R² (coefficient of determination) per joint
5. Aggregate by joint type (MCP, PIP, DIP) to identify systematic trends

---

### 4.4 Grasp Classification Accuracy

**Ground truth required:** Manual frame labels from ≥ 2 annotators

1. Manually label every frame of the evaluation video for grasp type
2. Verify inter-annotator agreement (Cohen's κ ≥ 0.75 before proceeding)
3. Run the annotator on the same video; extract the `grasp_type` column
4. Align labels by frame index
5. Compute:
   - **Confusion matrix** (7 × 7, rows = ground truth, columns = predicted)
   - **Per-class precision, recall, and F1**
   - **Macro F1** (equal weight across all 7 classes)
   - **Weighted F1** (weight by class frequency)
   - **Overall accuracy**
6. Record the `unknown` rate; if > 20%, consider threshold recalibration (see Section 6)
7. Apply **leave-one-video-out cross-validation** when evaluating across multiple recordings: never split at the frame level, as consecutive frames are highly correlated

---

### 4.5 Velocity Accuracy

**Ground truth required:** Motion-capture wrist trajectory

1. Apply the same first-order finite difference formula to the motion-capture wrist positions, using the same frame rate and stride settings as the annotator. This produces a reference velocity using the same computational definition.
2. Align annotator and reference velocity sequences by timestamp
3. Compute per-frame speed error: `|ee_speed_pred − ee_speed_ref|`
4. Report MAE and RMSE for speed magnitude, and the mean cosine similarity between the predicted and reference 3D velocity vectors (a measure of directional accuracy)
5. Optionally repeat after applying a 3-frame median filter to both sequences, to assess smoothed accuracy

---

## 5. Statistical Framework

### Sample size

A minimum of **500 frames** per metric is required for stable 95% confidence intervals. For grasp classification, this corresponds to approximately 70+ examples per class (assuming roughly uniform distribution).

### Confidence intervals

Bootstrap resampling (10,000 samples) is used to construct 95% confidence intervals for all reported MAE and RMSE values. Both the point estimate and the interval are reported:

> Wrist MAE 3D: **10.2 mm  [9.4, 11.1 mm]** (95% CI)

### Significance testing

When comparing performance across conditions (e.g. different lighting, different operators):
- **Paired t-test** when error distributions are approximately normal
- **Wilcoxon signed-rank test** when normality cannot be assumed (velocity and acceleration distributions are typically right-skewed)

### Effect size

Cohen's *d* is reported alongside p-values to characterise the practical magnitude of any differences:

| Cohen's *d* | Interpretation |
|------------|---------------|
| < 0.2 | Negligible |
| 0.2–0.5 | Small |
| 0.5–0.8 | Medium |
| > 0.8 | Large |

### Multi-sequence consistency

All evaluations are run on at least **three independent video sequences** (different operators, scenes, and lighting conditions). Per-sequence metrics are aggregated as mean ± standard deviation to characterise both typical performance and variability.

---

## 6. Threshold Calibration

### When to recalibrate grasp thresholds

Recalibration is warranted when overall macro F1 falls below 0.75 on a representative evaluation set. The procedure:

1. Construct a validation set of manually labelled frames covering all 7 grasp classes
2. Vary each curl-ratio threshold independently across the range 0.50–0.95 in steps of 0.05
3. For each threshold combination, compute macro F1 on the validation set
4. Select the combination that maximises macro F1
5. Validate on a held-out test set (not used during calibration) to confirm the improvement generalises

### When to recalibrate contact state thresholds

1. Collect manually labelled contact state annotations (`closed`, `partial`, `open`)
2. Plot the precision-recall curve for the `closed` class across a range of aperture thresholds
3. Identify the threshold that maximises the F1 score for the task-specific balance between false positives and false negatives
4. Repeat for the `open` / `not-open` split

---

## 7. Known Failure Modes

| Failure mode | Root cause | Detection | Recommended response |
|-------------|-----------|-----------|---------------------|
| Large Z error (> 40 mm) | Monocular depth ambiguity | Z MAE > 30 mm in depth-camera cross-validation | Supplement Z with a depth camera |
| RPY instability | Gimbal lock (`\|pitch\| > 80°`) | Flag frames where `abs(pitch_deg) > 80` | Exclude from orientation analysis; consider quaternion representation |
| Velocity spikes | Single-frame landmark jump (~10 mm) | `ee_speed > 0.8 m/s` for manual tasks | Apply 3-frame median filter before use |
| Grasp `unknown` rate > 20% | Curl ratios near classification boundary | Count `grasp_type == "unknown"` | Recalibrate thresholds (Section 6) |
| EE trajectory systematic offset | Incorrect `op_height_cm` argument | Compare normalised vs. raw trajectory extents | Verify height argument at recording time |
| All Z coordinates near zero | Camera nearly perpendicular to hand plane | `max(z0..z20) < 0.005 m` | Adjust camera angle or discard sequence |

---

## 8. Evaluation Tiers

### Minimal evaluation — approximately 3 hours

Suitable for a quick quality check during dataset collection:

1. Record 60 seconds of hand motion covering open, pinch, power, lateral, and hook postures plus transitions
2. Manually label `grasp_type` for every frame (one annotator)
3. Run the annotator; compare `grasp_type` column against labels
4. Report overall accuracy and macro F1
5. Visually inspect 20 randomly sampled frames: compare annotated joint angles against the video still image
6. Record the `unknown` rate; flag if above 20%

### Full evaluation — approximately 2–4 days

Suitable for publication-quality reporting or before deploying a training dataset:

1. Record 10 or more sequences across multiple operators, grasp types, lighting conditions, and scenes
2. Collect motion capture or depth camera ground truth for position accuracy (Protocol 4.1)
3. Collect IMU ground truth for orientation accuracy (Protocol 4.2)
4. Obtain manual grasp labels from two independent annotators; confirm κ ≥ 0.75 (Protocol 4.4)
5. Run all five evaluation protocols (Sections 4.1–4.5)
6. Apply the full statistical framework: bootstrap confidence intervals, significance tests, multi-sequence mean ± std (Section 5)
7. Populate all measured values in the [accuracy_metrics.md](accuracy_metrics.md) table
8. Recalibrate thresholds for any metric below the "Acceptable" level (Section 6)
