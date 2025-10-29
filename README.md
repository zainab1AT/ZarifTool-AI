# ZarifTool AI & Computer Vision

## Overview
This repository contains the AI and computer vision components of the ZarifTool application — the core system responsible for posture detection and analysis.
Using advanced pose estimation and keypoint detection techniques, the AI identifies postural issues such as kyphosis, lordosis, and forward head posture, providing data-driven recommendations to users.

## Technologies Used

YOLOv8 – for custom pose estimation and keypoint detection

MediaPipe – for baseline body landmark detection and experimentation

PyTorch – for model fine-tuning, training, and integration

OpenCV – for image preprocessing and visualization

## Rationale

The AI system was designed to ensure accurate, fast, and medically relevant posture analysis.
Each technology played a specific role:

YOLOv8 handled custom-trained keypoints for precise spine and pelvis detection.

MediaPipe provided a baseline for quick body landmark detection.

PyTorch supported additional keypoints and improved accuracy from side-view images.

## Model Details
YOLOv8 Pose Estimation

Trained on ~2,500 manually annotated images

Detected 7 posture-related keypoints:

Left Ear, Right Ear

Lower Neck

Thoracic Spine (Upper Back)

Lumbar Spine (Lower Back)

Pelvic Back, Pelvic Front

Used transfer learning with yolov8n-pose.pt base model

Average inference time: ~3 seconds per image

The extracted keypoints are used for:

Measuring spinal alignment

Detecting neck displacement

Evaluating pelvic tilt and rotation

## Outcome

The combined use of YOLOv8, MediaPipe, and PyTorch delivered:

High precision and recall in posture keypoint detection

Fast inference for near real-time assessment

Reliable diagnosis of postural deviations

## How to Run

1️⃣ Installation

* pip install ultralytics
* pip install mediapipe
* pip install opencv-python
* pip install torch torchvision torchaudio

2️⃣ Run the Model

* python main.py
