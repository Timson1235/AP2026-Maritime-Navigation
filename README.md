# Maritime Obstacle Detection for Unmanned Surface Vehicles

## Repository Link

[https://github.com/Timson1235/AP2026-Maritime-Navigation](https://github.com/Timson1235/AP2026-Maritime-Navigation)

## Description

Unmanned surface vehicles (USVs) need a reliable vision system to detect obstacles on open
water — other boats, buoys, swimmers, and debris — in order to navigate safely. This project
studies maritime obstacle detection on the **LaRS** (Lakes, Rivers and Seas) dataset and
benchmarks three object detectors from different architecture families on the same data:

- **YOLOv8** — one-stage CNN detector (speed-oriented)
- **Faster R-CNN** — two-stage CNN detector (classic baseline)
- **RF-DETR** — transformer detector with a DINOv2 backbone (accuracy-oriented)

LaRS provides panoptic instance masks only for its train and validation splits, so we derive
detection bounding boxes from the mask extents. Because several distinct objects sometimes
share one segment (producing a single oversized box), we apply a **model-assisted 4-fold
cross-validation relabeling** step that removes ghost and merged/oversized boxes and adds
high-confidence missing labels. We then compare the detectors on accuracy and inference speed,
analyse their errors by object size and class, and apply Explainable-AI methods (D-RISE,
Grad-CAM++, EigenCAM, and RF-DETR cross-attention) to see where each model "looks."

### Task Type

Object Detection (Computer Vision)

### Results Summary

All models were fine-tuned from COCO-pretrained weights on our scene-level re-split of LaRS
(train 2,102 / validation 198 / test 503 images) and evaluated on the held-out test split
(1,605 boxes after relabeling). Speed is the maximum sustained inference throughput on LaRS
test images, measured on a single NVIDIA A40 GPU.

#### Best Model Performance
- **Best Model:** RF-DETR (Large variant, DINOv2 backbone) — best Optuna trial
- **Evaluation Metric:** mAP@[.50:.95] (COCO-style), reported alongside mAP@.50,
  precision/recall/F1 (at the best-F1 confidence threshold), and inference FPS
- **Final Performance:** **34.1 mAP@[.50:.95]**, 58.6 mAP@.50, 81.2% precision,
  67.7% recall, F1 0.74 — at **27 FPS (real time)**

#### Model Comparison
| Model | mAP@.50:.95 | mAP@.50 | Precision | Recall | F1 | Speed |
|-------|:-----------:|:-------:|:---------:|:------:|:----:|:-----:|
| **RF-DETR** (transformer) | **0.341** | **0.586** | **0.812** | **0.677** | **0.738** | 27.2 FPS |
| Faster R-CNN (two-stage CNN) | 0.235 | 0.436 | 0.729 | 0.578 | 0.645 | 20.3 FPS |
| YOLOv8m (one-stage CNN) | 0.234 | 0.422 | 0.708 | 0.601 | 0.650 | **49.8 FPS** |

- **Improvement over the baselines:** RF-DETR adds **+10.5 mAP@[.50:.95]** (≈ +45% relative)
  over the stronger CNN baseline while still running in real time.
- **Best alternative model:** YOLOv8m — essentially tied with Faster R-CNN on accuracy but
  ~2.5× faster (49.8 FPS), making it the better choice when compute or frame rate is the
  priority.

#### Key Insights
- **Label quality was the single biggest factor on the scores** — more than the choice of
  architecture. The dominant error for every model is *missing* objects (predicting
  background) rather than confusing one class with another, which points to residual label
  noise and small-object difficulty rather than weak classification.
- **Small objects are the main open challenge**, and a safety concern: a missed buoy or
  swimmer is the most dangerous error, and bounding-box areas in LaRS reach down to a single
  pixel. The classes are also strongly imbalanced (boat/ship ≈ 64% of instances, float ≈ 0.2%).
- **RF-DETR is the recommended model for a USV:** best accuracy, real-time speed, the most
  temporally stable behaviour on live video, and a permissive **Apache-2.0** licence for
  commercial deployment (YOLOv8 is AGPL-3.0). XAI attention maps show RF-DETR attends to the
  objects themselves rather than to water reflections or glare — exactly the maritime failure
  mode we were concerned about.
- **Deployment relevance:** frame-by-frame predictions flicker, so a real USV would benefit
  from temporal smoothing or tracking across frames (not explored here).

## Documentation

1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics (EDA)](1_DatasetCharacteristics/exploratory_data_analysis.ipynb)**
3. **[Data Preprocessing](2_DataPreprocessing/)** — scene-level split, cross-validation
   relabeling, and offline augmentation
4. **[Models, Training & Evaluation](3_Model/)** — per-model training, Optuna hyperparameter
   search, and the side-by-side [`evaluate_all_models.ipynb`](3_Model/evaluate_all_models.ipynb)
5. **[Explainability (XAI)](5_XAI/XAI.ipynb)**
6. **[Presentation](4_Presentation/README.md)**

For a full end-to-end walkthrough — from the raw LaRS download through preprocessing,
training, hyperparameter search, evaluation, and the live-video demo — see
**[DATA_PIPELINE.md](DATA_PIPELINE.md)**.

## Cover Image

![Project Cover Image](CoverImage/cover_image.png)
