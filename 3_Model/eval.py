"""
Headless eval mirroring evaluation.ipynb's metric logic.
Usage:
    python eval.py <predictions.json> <model_id>
Updates ../runs/model_results.csv with the resulting row (mAP@50, mAP@50:95,
mAP@75, mAP@50_agnostic, P/R/F1 at best-F1 threshold, per-class AP@50).
"""
import sys
import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv

PREDICTIONS_FILE = Path(sys.argv[1])
MODEL_ID         = sys.argv[2]

GT_ANNOTATIONS   = Path(__file__).parent / "../Data/lars_processed/test/_annotations.coco.json"
RESULTS_CSV      = Path(__file__).parent / "../runs/model_results.csv"
IOU_THRESH       = 0.5

# ── Load GT ──────────────────────────────────────────────────────────────────
ann = json.load(open(GT_ANNOTATIONS))
img_meta   = {img["id"]: img for img in ann["images"]}
ann_by_img = {}
for a in ann["annotations"]:
    ann_by_img.setdefault(a["image_id"], []).append(a)

categories  = ann["categories"]
CLASS_IDS   = [c["id"]   for c in categories]
CLASS_NAMES = [c["name"] for c in categories]
id_to_idx   = {cid: i for i, cid in enumerate(CLASS_IDS)}
N_CLASSES   = len(CLASS_IDS)

img_id_order = list(img_meta.keys())

all_gt_dets = []
for img_id in img_id_order:
    gt_anns = ann_by_img.get(img_id, [])
    if gt_anns:
        gt_xyxy = np.array([[a["bbox"][0], a["bbox"][1],
                              a["bbox"][0] + a["bbox"][2],
                              a["bbox"][1] + a["bbox"][3]] for a in gt_anns], dtype=float)
        gt_cids = np.array([a["category_id"] for a in gt_anns], dtype=int)
    else:
        gt_xyxy = np.empty((0, 4), dtype=float)
        gt_cids = np.array([], dtype=int)
    all_gt_dets.append(sv.Detections(xyxy=gt_xyxy, class_id=gt_cids))

# ── Helpers ──────────────────────────────────────────────────────────────────
def box_iou(a, b):
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter    = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a   = (a[2] - a[0]) * (a[3] - a[1])
    area_b   = (b[2] - b[0]) * (b[3] - b[1])
    union    = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def match_detections(pred_det, gt_det, iou_thresh=0.5):
    pred_boxes   = pred_det.xyxy       if len(pred_det) > 0 else np.empty((0, 4))
    pred_classes = pred_det.class_id   if len(pred_det) > 0 else np.array([], dtype=int)
    pred_scores  = (pred_det.confidence
                    if pred_det.confidence is not None and len(pred_det) > 0
                    else np.ones(len(pred_det)))
    gt_boxes     = gt_det.xyxy         if len(gt_det) > 0 else np.empty((0, 4))
    gt_classes   = gt_det.class_id     if len(gt_det) > 0 else np.array([], dtype=int)
    if len(pred_boxes) == 0:
        return [], list(range(len(gt_boxes))), []
    if len(gt_boxes) == 0:
        return [], [], list(range(len(pred_boxes)))
    order        = np.argsort(-pred_scores)
    pred_boxes   = pred_boxes[order]
    pred_classes = pred_classes[order]
    pred_scores  = pred_scores[order]
    matched_gt, matched_pred = {}, {}
    for pi, pb in enumerate(pred_boxes):
        best_iou, best_gi = iou_thresh, -1
        for gi in range(len(gt_boxes)):
            if gi in matched_gt: continue
            iou = box_iou(pb, gt_boxes[gi])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0:
            matched_gt[best_gi] = pi
            matched_pred[pi]    = best_gi
    matched = list(matched_gt.keys())
    fn = [i for i in range(len(gt_boxes))  if i not in matched_gt]
    fp = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    return matched, fn, fp

# ── Compute mAP (using ALL predictions, no threshold) ────────────────────────
def to_0idx(xyxy, cids, scores=None):
    mapped = np.array([id_to_idx.get(int(c), -1) for c in cids], dtype=int)
    m = mapped >= 0
    conf = scores[m] if scores is not None else np.ones(m.sum())
    return sv.Detections(xyxy=xyxy[m], class_id=mapped[m], confidence=conf)

gt_sv = []
for d in all_gt_dets:
    if len(d) == 0:
        gt_sv.append(sv.Detections(xyxy=np.empty((0,4), dtype=float),
                                   class_id=np.array([], dtype=int)))
    else:
        mapped = np.array([id_to_idx.get(int(c), -1) for c in d.class_id], dtype=int)
        m = mapped >= 0
        gt_sv.append(sv.Detections(xyxy=d.xyxy[m], class_id=mapped[m]))

raw = json.load(open(PREDICTIONS_FILE))
by_img = {}
for p in raw:
    by_img.setdefault(p["image_id"], []).append(p)

pred_sv_all = []
for img_id in img_id_order:
    ps = by_img.get(img_id, [])
    if ps:
        xyxy   = np.array([[p["bbox"][0], p["bbox"][1],
                            p["bbox"][0]+p["bbox"][2],
                            p["bbox"][1]+p["bbox"][3]] for p in ps], dtype=float)
        cids   = np.array([p["category_id"] for p in ps], dtype=int)
        scores = np.array([p["score"]        for p in ps], dtype=float)
        pred_sv_all.append(to_0idx(xyxy, cids, scores))
    else:
        pred_sv_all.append(sv.Detections(xyxy=np.empty((0,4), dtype=float),
                                         class_id=np.array([], dtype=int),
                                         confidence=np.array([], dtype=float)))

metric = sv.metrics.MeanAveragePrecision()
metric.update(pred_sv_all, gt_sv)
res = metric.compute()
ap_map = {int(cls): res.ap_per_class[i, 0] for i, cls in enumerate(res.matched_classes)}

# Class-agnostic
def strip_class(d):
    if len(d) == 0:
        return sv.Detections(xyxy=np.empty((0,4),dtype=float),
                             class_id=np.array([],dtype=int),
                             confidence=d.confidence if d.confidence is not None else None)
    conf = d.confidence if d.confidence is not None else np.ones(len(d))
    return sv.Detections(xyxy=d.xyxy, class_id=np.zeros(len(d), dtype=int), confidence=conf)

metric_agn = sv.metrics.MeanAveragePrecision()
metric_agn.update([strip_class(d) for d in pred_sv_all],
                  [strip_class(d) for d in gt_sv])
res_agn = metric_agn.compute()

# ── P / R / F1 threshold sweep ────────────────────────────────────────────────
thresholds = np.arange(0.05, 0.96, 0.05)
precisions, recalls, f1s = [], [], []
for t in thresholds:
    preds_t = {}
    for p in raw:
        if p["score"] >= t:
            preds_t.setdefault(p["image_id"], []).append(p)
    tp_t = fp_t = fn_t = 0
    for i, img_id in enumerate(img_id_order):
        ps = preds_t.get(img_id, [])
        if ps:
            xyxy   = np.array([[p["bbox"][0], p["bbox"][1],
                                p["bbox"][0]+p["bbox"][2],
                                p["bbox"][1]+p["bbox"][3]] for p in ps], dtype=float)
            cids   = np.array([p["category_id"] for p in ps], dtype=int)
            scores = np.array([p["score"]        for p in ps], dtype=float)
            pred_d = sv.Detections(xyxy=xyxy, class_id=cids, confidence=scores)
        else:
            pred_d = sv.Detections(xyxy=np.empty((0,4),dtype=float),
                                   class_id=np.array([],dtype=int),
                                   confidence=np.array([],dtype=float))
        matched, fn, fp = match_detections(pred_d, all_gt_dets[i], IOU_THRESH)
        tp_t += len(matched); fp_t += len(fp); fn_t += len(fn)
    prec = tp_t / (tp_t + fp_t) if tp_t + fp_t > 0 else 0.0
    rec  = tp_t / (tp_t + fn_t) if tp_t + fn_t > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
    precisions.append(prec); recalls.append(rec); f1s.append(f1)

best_f1_idx    = int(np.argmax(f1s))
best_f1_thresh = float(thresholds[best_f1_idx])

# ── Print + upsert into CSV ──────────────────────────────────────────────────
print(f"\n=== {MODEL_ID} ===")
print(f"mAP@50:95       = {res.map50_95:.4f}")
print(f"mAP@50          = {res.map50:.4f}")
print(f"mAP@75          = {res.map75:.4f}")
print(f"mAP@50_agnostic = {res_agn.map50:.4f}")
print(f"Best F1 @ thr={best_f1_thresh:.2f}: P={precisions[best_f1_idx]:.4f}  R={recalls[best_f1_idx]:.4f}  F1={f1s[best_f1_idx]:.4f}")
print("Per-class AP@50:")
for idx, name in enumerate(CLASS_NAMES):
    ap = ap_map.get(idx, float("nan"))
    print(f"  {name:<18} {ap:.4f}")

row = {
    "model":           MODEL_ID,
    "mAP@50:95":       round(float(res.map50_95), 4),
    "mAP@50":          round(float(res.map50),    4),
    "mAP@75":          round(float(res.map75),    4),
    "mAP@50_agnostic": round(float(res_agn.map50), 4),
    "precision":       round(float(precisions[best_f1_idx]), 4),
    "recall":          round(float(recalls[best_f1_idx]),    4),
    "F1":              round(float(f1s[best_f1_idx]),        4),
    "best_thresh":     round(best_f1_thresh, 2),
    "iou_thresh":      IOU_THRESH,
    "evaluated_at":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
}
for idx, name in enumerate(CLASS_NAMES):
    row[f"AP@50_{name}"] = round(float(ap_map.get(idx, float("nan"))), 4)

if RESULTS_CSV.exists():
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["model"] != MODEL_ID]
else:
    df = pd.DataFrame()

df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# Preserve fps/latency columns if they existed before (they're set separately)
for col in ["fps","latency_mean_ms","latency_std_ms"]:
    if col not in df.columns:
        df[col] = np.nan

df.to_csv(RESULTS_CSV, index=False)
print(f"\nUpdated row in {RESULTS_CSV}")
