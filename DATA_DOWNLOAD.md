# Data Download

This project uses the **LaRS** (Lakes, Rivers and Seas) maritime obstacle
detection benchmark. The dataset is not included in this repository due to its
size — download it from the official page:

**https://lojzezust.github.io/lars-dataset/**

The project uses the panoptic instance masks of the **train** and **validation**
splits, from which detection bounding boxes are derived. After downloading, see
[DATA_PIPELINE.md](DATA_PIPELINE.md) for the full preprocessing steps
(scene-level re-split, cross-validation relabeling, and the COCO/YOLO format
conversions) that turn the raw download into the `Data/lars_processed` and
`Data/lars_relabeled` layouts used by the training scripts.
