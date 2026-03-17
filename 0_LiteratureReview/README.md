# Literature Review

Approaches or solutions that have been tried before on similar projects.

---

## Source 1: ODExAI: A Comprehensive Object Detection Explainable AI Evaluation

**Link:** https://arxiv.org/abs/2504.19249

**Objective:**  
To establish a standardized evaluation framework for assessing Explainable AI (XAI) methods applied to bounding-box object detection models.

**Methods:**  
Benchmarked several XAI methods (D-RISE, D-CLOSE, EigenCAM) across prominent object detectors, specifically Faster R-CNN and YOLO.

**Outcomes:**  
Region-based XAI methods showed superior faithfulness to the model's decision process for bounding boxes, while CAM-based methods effectively highlighted specific object pixels.

**Relation to the Project:**  
Provides a baseline methodology and evaluation metrics to test Faster R-CNN, YOLO, and XAI approaches for pure object detection. :contentReference[oaicite:0]{index=0}

---

## Source 2: Deformable DETR: Deformable Transformers for End-to-End Object Detection

**Link:** https://arxiv.org/abs/2010.04159

**Objective:**  
To address the slow convergence and poor small-object performance of the original DETR model for bounding box detection.

**Methods:**  
Replaced global Transformer attention with deformable attention modules that focus on a small set of relevant sampling points.

**Outcomes:**  
Achieved better performance on small objects compared to Faster R-CNN and original DETR, while requiring about **10× fewer training epochs**. :contentReference[oaicite:1]{index=1}

**Relation to the Project:**  
Serves as the foundational architecture for modern bounding-box Transformers used to benchmark against YOLO and Faster R-CNN.

---

## Source 3: MiM-ISTD: Mamba-in-Mamba for Efficient Infrared Small Target Detection

**Link:** https://arxiv.org/abs/2403.02148

**Objective:**  
To apply State Space Models (SSMs), specifically the Mamba architecture, to detect very small objects while maintaining global context efficiently.

**Methods:**  
Introduces a **Mamba-in-Mamba (MiM)** architecture that processes image patches as sequences to capture long-range dependencies and local features simultaneously.

**Outcomes:**  
Achieved strong detection accuracy with improved computational efficiency compared to CNN and Transformer approaches, particularly for small targets. :contentReference[oaicite:2]{index=2}

**Relation to the Project:**  
Fulfills the requirement to explore a Mamba-based model for bounding boxes, demonstrating its efficiency in small-object detection tasks.

---

## Source 4: Impact of Surface Reflections in Maritime Obstacle Detection

**Link:** https://arxiv.org/abs/2410.08713

**Objective:**  
To investigate how surface reflections negatively affect bounding-box detectors in maritime environments.

**Methods:**  
Evaluated several detectors (YOLO variants, Faster R-CNN, Transformer-based detectors) on maritime image datasets containing reflection artifacts.

**Outcomes:**  
Surface reflections significantly reduced detection accuracy (mAP decreased by up to **9.6 points**). A proposed heatmap-based filtering method reduced false positives caused by reflections by **about 35%**. :contentReference[oaicite:3]{index=3}

**Relation to the Project:**  
Directly analyzes how YOLO, Faster R-CNN, and Transformer models perform in maritime environments similar to the LaRS dataset.

---

## Source 5: 3rd Workshop on Maritime Computer Vision (MaCVi) 2025 – Challenge Results

**Link:** https://macvi.org/workshop/macvi25/challenges

**Objective:**  
To benchmark state-of-the-art algorithms for maritime obstacle detection in autonomous surface vehicles.

**Methods:**  
Participants trained models such as advanced YOLO variants and Transformer-based RT-DETR on maritime datasets like LaRS.

**Outcomes:**  
Transformer architectures achieved higher accuracy on very small objects (e.g., buoys), while YOLO-based models maintained superior real-time inference speed. :contentReference[oaicite:4]{index=4}

**Relation to the Project:**  
Provides a current benchmark comparison of YOLO and Transformer-based object detectors on maritime datasets similar to LaRS.