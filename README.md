# Iris-Detection
Deep Iris Detection Model using Python and Tensorflow
This project implements a real-time eye detection and classification system using a custom-trained VGG16 model. It detects open/closed eyes and localizes the right and left eyes as colored points (red for right eye, green for left eye) on human faces.

Key Features
Real-time face & eye detection.

Localizes right eye (🔴) and left eye (🟢) with dot markers.

Classifies each eye as open or closed.

Uses a custom dataset of face images annotated with eye keypoints.

Fine-tuned VGG16 model with dual-task output:

Eye state classification (binary)

Eye position prediction (keypoint coordinates)

Dataset
Collected manually from Google images (open and closed eyes).

Annotated using LabelMe for:

Right eye (x, y)

Left eye (x, y)

Eye state (open = 1, closed = 0)

Augmentations applied with Albumentations:

Random crop

Brightness/contrast

Horizontal flip

Gamma correction

Split into:

train/

val/

test/

Each sample includes:

The face image.

A label file with:

Left and right eye keypoints.

Binary eye state.

Model Architecture
Base Model: VGG16 (pretrained on ImageNet, feature extractor).

Custom Heads:

Eye State Classification (open/closed).

Eye Position Regression (left & right eye keypoints).

Loss Functions:
Binary Cross-Entropy Loss for eye state classification.

Mean Squared Error (MSE) for keypoint (dot) regression.

Training
bash
Copy
Edit
python train.py \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --train_dir ./data/train \
  --val_dir ./data/val \
  --model vgg16
Real-Time Detection
Detect and visualize eyes using webcam or image input.

Green dot: Left eye

Red dot: Right eye

Label for each eye: Open or Closed

bash
Copy
Edit
python detect_realtime.py
Example Output
<img width="676" height="321" alt="image" src="https://github.com/user-attachments/assets/d6e8469d-9f56-4ebb-bd0c-73d1cd8ddc4a" />

Face detected

🔴 Red dot on right eye

🟢 Green dot on left eye

Labels: Right Eye: Open, Left Eye: Closed

Requirements
Python 3.8+

TensorFlow or Keras

OpenCV

Albumentations

LabelMe

NumPy, Matplotlib

bash
Copy
Edit
pip install -r requirements.txt

Future Improvements
Add eyelid closure duration tracking (e.g., for driver drowsiness detection).

Expand dataset to handle occlusion and glasses.

Convert to mobile-friendly format (TFLite or ONNX).

Acknowledgements
VGG16 architecture by Simonyan & Zisserman

Annotation using LabelMe

Data augmentation via Albumentations

<img width="676" height="321" alt="image" src="https://github.com/user-attachments/assets/d6e8469d-9f56-4ebb-bd0c-73d1cd8ddc4a" />
