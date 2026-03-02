# Iris Detection – Real-Time Eye State & Keypoint Detection

## Overview

This project implements a **real-time eye and iris detection system** using **Python and TensorFlow**. The model detects human faces, localizes the **left and right eyes**, and classifies each eye as **open or closed**. The system is designed to demonstrate a complete computer vision workflow, from **manual data annotation** to **model training and real-time inference**.

This project also highlights strong experience in **data annotation, keypoint labeling, dataset structuring, and quality control**, making it relevant for large-scale annotation tasks and AI dataset preparation.

---

## Key Features

* Real-time face and eye detection
* Localization of:

  * **Right eye** (🟢 Green dot)
  * **Left eye** (🔴 Red dot)
* Eye state classification: **Open / Closed**
* Custom annotated dataset with eye keypoints
* Multi-task deep learning model (classification + regression)

---

## Dataset

### Data Collection

* Images collected manually from public sources (Google Images)
* Includes both **open-eye** and **closed-eye** samples

### Annotation

* Annotation tool: **LabelMe**
* Each image annotated with:

  * Left eye keypoint (x, y)
  * Right eye keypoint (x, y)
  * Eye state label:

    * Open = 1
    * Closed = 0

### Dataset Structure

```
data/
├── train/
├── val/
└── test/
```

Each sample includes:

* Face image
* Label file containing:

  * Left & right eye coordinates
  * Binary eye-state label

---

## Data Augmentation

Data augmentation was applied using **Albumentations** to improve model robustness:

* Random crop
* Brightness / contrast adjustment
* Horizontal flip
* Gamma correction

---

## Model Architecture

### Base Model

* **VGG16** (pretrained on ImageNet)
* Used as a feature extractor

### Custom Heads

* **Eye State Classification Head**

  * Binary classification (open / closed)
* **Eye Position Regression Head**

  * Predicts left and right eye keypoints (x, y)

### Loss Functions

* Binary Cross-Entropy Loss → Eye state classification
* Mean Squared Error (MSE) → Keypoint regression

---

## Training

### Training Command

```
python train.py \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --train_dir ./data/train \
  --val_dir ./data/val \
  --model vgg16
```

### Training Details

* Optimized for both classification and localization tasks
* Validation used to monitor overfitting

---

## Real-Time Detection

Run real-time eye detection using a webcam or image input:

```
python detect_realtime.py
```

### Output Visualization

* Face detected
* 🔴 Red dot → Left eye
* 🟢 Green dot → Right eye
* Labels displayed:

  * Right Eye: Open / Closed
  * Left Eye: Open / Closed
<img width="676" height="321" alt="image" src="https://github.com/user-attachments/assets/d6e8469d-9f56-4ebb-bd0c-73d1cd8ddc4a" />
---

## Requirements

* Python 3.8+
* TensorFlow / Keras
* OpenCV
* Albumentations
* LabelMe
* NumPy
* Matplotlib

### Install Dependencies

```
pip install -r requirements.txt
```

---

## Future Improvements

* Track eyelid closure duration (e.g., driver drowsiness detection)
* Expand dataset to handle occlusion and glasses
* Convert model to mobile-friendly formats (TFLite / ONNX)
* Improve performance on challenging lighting conditions

---

## Acknowledgements

* VGG16 architecture
* LabelMe for annotation
* Albumentations for data augmentation

---

## Author

**Uma**

This project demonstrates end-to-end experience in **data annotation, computer vision pipelines, and real-time deep learning systems**.

