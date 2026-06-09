"""
video_inference.py

Run live detection on a video with one of the three LARS detectors
(Faster R-CNN, RF-DETR, YOLOv8) and write an annotated MP4.

Usage
-----
    python video_inference.py --video clip.mp4 --model yolo
    python video_inference.py --video clip.mp4 --model rfdetr --conf 0.5
    python video_inference.py --video clip.mp4 --model frcnn \
        --output out.mp4 --start-frame 300 --max-frames 600 --display

Defaults
--------
- Weights are read from ../model_showcase/ (the symlinks the demo notebook uses).
- Confidence thresholds match the evaluation-tuned values from the XAI notebook.
- Output path defaults to "<video stem>_<model>_annotated.mp4" next to the input.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str((_HERE / "../3_Model").resolve()))

CLASS_NAMES = [
    "Boat/ship", "Row boats", "Paddle board", "Buoy",
    "Swimmer", "Animal", "Float", "Other",
]
N_CLASSES = len(CLASS_NAMES)

DEFAULT_CONF = {"frcnn": 0.7, "rfdetr": 0.4, "yolo": 0.3}
DEFAULT_WEIGHTS = {
    "frcnn":  _HERE / "../model_showcase/fasterrcnn_trial002.pth",
    "rfdetr": _HERE / "../model_showcase/rfdetr_trial004.pth",
    "yolo":   _HERE / "../model_showcase/yolo_exp7.pt",
}
# Max sustained inference FPS measured on LARS test images
# (see model_showcase/model_results.csv).
MODEL_MAX_FPS = {"frcnn": 20.3, "rfdetr": 27.2, "yolo": 49.8}

# Longest-side image dimension each model was trained at. Frames are resized
# to this before inference so the benchmark FPS actually applies (otherwise
# 4K input swamps everything). Aspect ratio is preserved.
MODEL_INFER_SIZE = {"frcnn": 1280, "rfdetr": 784, "yolo": 1024}


# ---------------------------------------------------------------------------
# Model adapters — each returns a callable predict(frame_rgb_u8) -> (boxes, cls, scores)
# ---------------------------------------------------------------------------

def _load_frcnn(weights: Path, device: torch.device):
    from train_fasterrcnn import build_model
    import torchvision.transforms.functional as TF

    model = build_model("base", num_classes=N_CLASSES + 1)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    @torch.no_grad()
    def predict(frame_rgb):
        t = TF.to_tensor(frame_rgb).to(device)
        out = model([t])[0]
        return (
            out["boxes"].cpu().numpy(),
            (out["labels"].cpu().numpy() - 1),   # 1-indexed → 0-indexed
            out["scores"].cpu().numpy(),
        )
    return predict


def _load_rfdetr(weights: Path, device: torch.device,
                 variant: str = "base", resolution: int = 784):
    from rfdetr import RFDETRBase, RFDETRLarge
    from PIL import Image

    ModelCls = RFDETRLarge if variant == "large" else RFDETRBase
    model = ModelCls(
        resolution=resolution, num_classes=N_CLASSES,
        pretrain_weights=str(weights),
    )
    model.optimize_for_inference()

    def predict(frame_rgb):
        # rfdetr.predict takes a PIL image; threshold filtering happens later.
        dets = model.predict(Image.fromarray(frame_rgb), threshold=0.01)
        if len(dets) == 0:
            return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
        return (
            np.asarray(dets.xyxy),
            np.asarray(dets.class_id).astype(int),
            np.asarray(dets.confidence),
        )
    return predict


def _load_yolo(weights: Path, device: torch.device):
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def predict(frame_rgb):
        res = model.predict(
            source=frame_rgb, imgsz=1024, conf=0.01,
            device=device, verbose=False,
        )[0]
        if len(res.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
        return (
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.cls.cpu().numpy().astype(int),
            res.boxes.conf.cpu().numpy(),
        )
    return predict


LOADERS = {"frcnn": _load_frcnn, "rfdetr": _load_rfdetr, "yolo": _load_yolo}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

# Fixed colour per class (BGR for cv2)
_PALETTE = [
    ( 60,  76, 231), (113, 179,  60), ( 33, 162, 245), (199, 173,  41),
    (180,  60, 200), ( 22, 205, 255), (255, 100, 100), (128, 128, 128),
]
def _colour(cls_id: int) -> tuple[int, int, int]:
    return _PALETTE[cls_id % len(_PALETTE)]


def draw_detections(frame_bgr, boxes, cls, scores, conf_thr):
    h, w = frame_bgr.shape[:2]
    for box, c, s in zip(boxes, cls, scores):
        if s < conf_thr:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        col = _colour(int(c))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), col, 2)
        name = CLASS_NAMES[int(c)] if 0 <= int(c) < N_CLASSES else f"id={int(c)}"
        label = f"{name} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr


def draw_hud(frame_bgr, *, model_name, conf_thr, fps_inst, fps_avg,
             frame_idx, total_frames, n_kept, out_fps, src_fps, skip):
    lines = [
        f"model: {model_name}   conf>={conf_thr:.2f}",
        f"infer: {fps_inst:5.1f} FPS  (avg {fps_avg:5.1f})   "
        f"out {out_fps:5.2f} FPS / src {src_fps:5.2f} FPS  (skip={skip})",
        f"src frame {frame_idx}/{total_frames}   dets: {n_kept}",
    ]
    y = 20
    for line in lines:
        cv2.putText(frame_bgr, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 1, cv2.LINE_AA)
        y += 22


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--video", required=True, type=Path, help="Input video file")
    p.add_argument("--model", required=True, choices=list(LOADERS), help="Which detector to run")
    p.add_argument("--weights", type=Path, default=None,
                   help="Override weights path; defaults to ../model_showcase/<model>")
    p.add_argument("--output", type=Path, default=None,
                   help="Annotated MP4 output path; defaults to <video>_<model>_annotated.mp4")
    p.add_argument("--conf", type=float, default=None,
                   help="Confidence threshold for drawing; per-model default if omitted")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Skip this many frames before starting inference")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many processed frames")
    p.add_argument("--display", action="store_true",
                   help="Also show a cv2 window (needs an X display); press 'q' to quit")
    p.add_argument("--no-write", action="store_true",
                   help="Skip writing the annotated MP4 (display-only mode)")
    p.add_argument("--target-fps", type=float, default=None,
                   help="Cap output FPS. Default is the model's max sustained FPS "
                        "from model_results.csv. Source frames are dropped via "
                        "integer skip so output FPS = source FPS / k.")
    p.add_argument("--infer-size", type=int, default=None,
                   help="Longest-side resize before inference. "
                        "Default matches the model's training resolution.")
    p.add_argument("--output-size", type=int, default=None,
                   help="Longest-side resize for the WRITTEN/annotated video. "
                        "Decoupled from --infer-size: detections are computed at "
                        "infer-size, then scaled up and drawn on this larger frame "
                        "for a crisp output. Default: min(source longest side, 1920).")
    p.add_argument("--device", default=None,
                   help="Torch device override (e.g. 'cuda:0', 'cpu')")
    p.add_argument("--rfdetr-variant", choices=["base", "large"], default="base",
                   help="RF-DETR model variant the weights were trained with")
    p.add_argument("--rfdetr-resolution", type=int, default=784,
                   help="RF-DETR model input resolution (must match training)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.video.exists():
        sys.exit(f"video not found: {args.video}")

    weights = args.weights or DEFAULT_WEIGHTS[args.model]
    if not Path(weights).exists():
        sys.exit(f"weights not found: {weights}")

    conf_thr = args.conf if args.conf is not None else DEFAULT_CONF[args.model]
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    print(f"[setup] model={args.model}  device={device}  weights={weights}")
    print(f"[setup] conf>={conf_thr:.2f}")
    if args.model == "rfdetr":
        predict = _load_rfdetr(weights, device,
                               variant=args.rfdetr_variant,
                               resolution=args.rfdetr_resolution)
    else:
        predict = LOADERS[args.model](weights, device)

    # Warm up to absorb cuDNN auto-tune / JIT cost — otherwise the first "live"
    # inference takes seconds and consumes a huge chunk of video budget.
    print("[setup] warming up ...", end=" ", flush=True)
    warm = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(2):
        predict(warm)
    print("done")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {src_w}x{src_h} @ {src_fps:.2f} fps   total frames: {total}")

    # Match output FPS to the model's max sustained inference rate (integer
    # source-frame skip). Caller can override with --target-fps.
    target_fps = args.target_fps if args.target_fps is not None else MODEL_MAX_FPS[args.model]
    skip       = max(1, int(np.ceil(src_fps / target_fps)))
    out_fps    = src_fps / skip

    # Two independent resolutions:
    #   infer_size  — longest side fed to the model (its training res); caps detection quality.
    #   output_size — longest side of the WRITTEN video; only affects visual sharpness.
    # Detections are computed at infer_size then scaled to the output frame for drawing.
    infer_size = args.infer_size if args.infer_size is not None else MODEL_INFER_SIZE[args.model]
    output_size = args.output_size if args.output_size is not None else min(max(src_w, src_h), 1920)

    def _fit(longest):
        if max(src_w, src_h) > longest:
            s = longest / max(src_w, src_h)
            return int(round(src_w * s)), int(round(src_h * s))
        return src_w, src_h

    out_w, out_h = _fit(output_size)   # written frame
    inf_w, inf_h = _fit(infer_size)    # frame fed to the model
    # Box coords come back in inference-frame pixels; scale up to the output frame.
    box_scale_x = out_w / inf_w
    box_scale_y = out_h / inf_h
    print(f"[plan ] model max ~{MODEL_MAX_FPS[args.model]:.1f} FPS  →  "
          f"keep every {skip} source frame(s)  →  output {out_fps:.2f} FPS"
          f"  (~{total // skip} frames)")
    print(f"[plan ] infer at {inf_w}x{inf_h} (size {infer_size})  →  "
          f"write at {out_w}x{out_h} (size {output_size})")

    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        print(f"[video] starting at frame {args.start_frame}")

    writer = None
    if not args.no_write:
        out_path = args.output or args.video.with_name(
            f"{args.video.stem}_{args.model}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))
        if not writer.isOpened():
            sys.exit(f"cannot open output writer: {out_path}")
        print(f"[out  ] {out_path}")

    fps_window = deque(maxlen=30)
    t_run_start = time.time()
    src_idx     = args.start_frame
    written     = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            # Drop source frames that fall between our retained timestamps.
            if (src_idx - args.start_frame) % skip != 0:
                src_idx += 1
                continue

            # Both frames are resized from the ORIGINAL source so neither is
            # degraded by the other (e.g. infer_size > output_size must still
            # see full source detail, not an upscaled output frame).
            src_frame = frame_bgr
            # Inference frame: model's training resolution.
            if (inf_w, inf_h) != (src_w, src_h):
                inf_bgr = cv2.resize(src_frame, (inf_w, inf_h),
                                     interpolation=cv2.INTER_AREA)
            else:
                inf_bgr = src_frame
            # Output frame (what gets annotated + written) at output resolution.
            if (out_w, out_h) != (src_w, src_h):
                frame_bgr = cv2.resize(src_frame, (out_w, out_h),
                                       interpolation=cv2.INTER_AREA)
            else:
                frame_bgr = src_frame
            frame_rgb = cv2.cvtColor(inf_bgr, cv2.COLOR_BGR2RGB)

            # Time only the model call — honest, input-resolution-independent FPS.
            t0 = time.time()
            boxes, cls, scores = predict(frame_rgb)
            dt = time.time() - t0

            keep = scores >= conf_thr
            n_kept = int(keep.sum())

            # Scale boxes from inference-frame pixels up to the output frame.
            boxes_out = boxes[keep].copy()
            if len(boxes_out):
                boxes_out[:, [0, 2]] *= box_scale_x
                boxes_out[:, [1, 3]] *= box_scale_y
            draw_detections(frame_bgr, boxes_out, cls[keep], scores[keep], 0.0)

            fps_window.append(dt)
            fps_inst = 1.0 / max(dt, 1e-6)
            fps_avg  = len(fps_window) / max(sum(fps_window), 1e-6)
            draw_hud(frame_bgr,
                     model_name=args.model, conf_thr=conf_thr,
                     fps_inst=fps_inst, fps_avg=fps_avg,
                     frame_idx=src_idx, total_frames=total, n_kept=n_kept,
                     out_fps=out_fps, src_fps=src_fps, skip=skip)

            if writer is not None:
                writer.write(frame_bgr)
            if args.display:
                cv2.imshow(f"video_inference [{args.model}]", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[run  ] interrupted by user")
                    break

            written += 1
            if written % 30 == 0:
                print(f"[run  ] src frame {src_idx:>6}  written {written}"
                      f"  inst {fps_inst:5.1f} FPS  avg {fps_avg:5.1f}  dets {n_kept}")
            if args.max_frames is not None and written >= args.max_frames:
                print(f"[run  ] stopped at --max-frames={args.max_frames}")
                break
            src_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    wall = time.time() - t_run_start
    print(f"[done ] wrote {written} frames at {out_fps:.2f} FPS  "
          f"(skip={skip}, src_fps={src_fps:.2f}) in {wall:.1f}s  "
          f"infer avg {written/max(wall,1e-6):.1f} FPS")


if __name__ == "__main__":
    main()
