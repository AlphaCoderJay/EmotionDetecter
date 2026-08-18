# EmotionDetecter
# 😐 Emotion Detector

A real-time facial emotion classification project built with **PyTorch, OpenCV, and MTCNN**.

The project combines a custom **ResNet-style convolutional neural network** with face detection to recognize emotions from faces in webcam footage or video files.

The model was trained from scratch and then integrated into a real-time computer vision pipeline that displays both the predicted emotion and the probability distribution across all classes.

> **Note:** This project predicts the emotion labels represented in its training dataset. Facial expressions are not a definitive measurement of someone's internal emotional state, so predictions should be treated as model classifications rather than objective readings of emotion.

---

## Overview

The complete pipeline looks like this:

```text
Webcam / Video
      │
      ▼
   OpenCV
      │
      ▼
   MTCNN
      │
      ▼
Face Detection
      │
      ▼
Crop Each Face
      │
      ▼
Resize + Normalize
      │
      ▼
Custom ResNet
      │
      ▼
6 Emotion Classes
      │
      ├── Prediction + Confidence
      │
      └── Probability Distribution
```

The application can process **multiple faces in the same frame**, with an individual prediction displayed over each detected face.

---

# Emotions

The model was trained to classify six emotion labels:

| Class | Emotion   |
| ----: | --------- |
|     0 | Angry     |
|     1 | Fearful   |
|     2 | Happy     |
|     3 | Neutral   |
|     4 | Sad       |
|     5 | Surprised |

The model outputs six logits which are converted into probabilities using a softmax function during inference.

---

# Model Architecture

Instead of using a pretrained ResNet, I implemented a **ResNet-style architecture from scratch in PyTorch**.

The network consists of four stages of residual blocks.

```text
Input: 3 × 64 × 64
        │
        ▼
3×3 Conv
64 Channels
        │
        ▼
Layer 1
64 Channels
        │
        ▼
Layer 2
128 Channels
        │
        ▼
Layer 3
256 Channels
        │
        ▼
Layer 4
512 Channels
        │
        ▼
Global Average Pooling
        │
        ▼
Dropout
        │
        ▼
Linear Layer
512 → 6
        │
        ▼
Emotion Prediction
```

---

# Residual Blocks

The core building block follows the idea behind ResNet residual learning.

Each block contains:

```text
Input
 │
 ├───────────────────────┐
 │                       │
 ▼                       │
3×3 Convolution           │
 │                       │
 ▼                       │
BatchNorm                 │
 │                       │
 ▼                       │
ReLU                      │
 │                       │
 ▼                       │
3×3 Convolution           │
 │                       │
 ▼                       │
BatchNorm                 │
 │                       │
 └────────── + ───────────┘
             │
             ▼
            ReLU
```

When the input and output dimensions differ, a `1×1` convolution with the appropriate stride is used on the identity path.

This allows the network to progressively increase its feature dimensions while reducing spatial resolution.

---

# Network Configuration

The four residual stages use:

```text
Layer 1: 64 channels
Layer 2: 128 channels
Layer 3: 256 channels
Layer 4: 512 channels
```

Spatial downsampling occurs between stages using a stride of `2`.

The final feature representation is reduced using:

```python
nn.AdaptiveAvgPool2d((1, 1))
```

This produces a fixed-size representation regardless of the spatial dimensions before the pooling layer.

---

# Training

The model was trained for:

```text
Epochs: 100
Batch Size: 64
Optimizer: AdamW
Learning Rate: 3 × 10⁻⁴
Weight Decay: 1 × 10⁻⁴
Loss: Cross Entropy
```

The learning rate was controlled using cosine annealing:

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_epochs
)
```

---

# Data Augmentation

The training pipeline applies several augmentations to improve generalization.

```python
RandomResizedCrop(
    64,
    scale=(0.8, 1.0)
)

Resize((64, 64))

RandomHorizontalFlip()

RandomRotation(15)

RandomErasing(p=0.3)
```

Images are then converted to tensors and normalized:

```text
Mean = (0.5, 0.5, 0.5)
Std  = (0.5, 0.5, 0.5)
```

The test set uses deterministic preprocessing:

```text
Resize → Tensor → Normalize
```

---

# MixUp

Training also uses **MixUp** augmentation.

Two images are combined using a randomly sampled interpolation coefficient:

```text
Mixed Image =
λ × Image A +
(1 − λ) × Image B
```

The corresponding labels are mixed when calculating the loss:

```python
loss = (
    lam * loss_function(outputs, labels_a)
    + (1 - lam) * loss_function(outputs, labels_b)
)
```

The MixUp parameter was:

```text
alpha = 0.2
```

This was used as an additional regularization technique during training.

---

# Mixed Precision Training

The model uses PyTorch automatic mixed precision:

```python
with autocast(device_type=device):
    outputs = net(inputs)
    loss = ...
```

and gradient scaling:

```python
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

This allows compatible operations to use lower precision during training while using gradient scaling to maintain numerical stability.

---

# Face Detection

The trained emotion classifier is combined with **MTCNN** for face detection.

```python
mtcnn = MTCNN(
    keep_all=True,
    device=device
)
```

For every video frame:

```text
Frame
 ↓
BGR → RGB
 ↓
MTCNN
 ↓
Face Bounding Boxes
 ↓
Crop Faces
 ↓
Emotion Classifier
```

The detector can return multiple faces, allowing the application to classify several people in the same frame.

A configurable confidence threshold is also available:

```bash
python app.py --confidence 0.9
```

The default threshold is:

```text
0.9
```

---

# Real-Time Inference

For every detected face, the application:

1. Extracts the face bounding box.
2. Clamps the coordinates to the image dimensions.
3. Crops the face.
4. Resizes it to `64 × 64`.
5. Normalizes the image.
6. Passes it through the ResNet.
7. Applies softmax.
8. Selects the highest-probability emotion.
9. Draws the prediction onto the video.

Example:

```text
┌──────────────────────────────┐
│                              │
│       ┌─────────────┐        │
│       │    FACE     │        │
│       │             │        │
│       └─────────────┘        │
│       Happy: 94.23%          │
│                              │
└──────────────────────────────┘
```

---

# Probability Visualization

Alongside the main video window, the application displays a second window containing the probability distribution for all six emotions.

```text
Happy:     ███████████████████ 94.2%
Neutral:   ██                  3.1%
Surprised: █                    1.4%
Sad:       █                    0.8%
Angry:                         0.3%
Fearful:                       0.2%
```

This provides more information than simply showing the top prediction.

It also makes it possible to see when the model is uncertain between multiple classes.

---

# Command Line Arguments

The application supports command-line arguments using `argparse`.

### Video Source

```bash
python app.py --video 0
```

`0` uses the default webcam.

A video file can also be supplied:

```bash
python app.py --video path/to/video.mp4
```

### Detection Confidence

```bash
python app.py --confidence 0.9
```

The default confidence threshold is:

```text
0.9
```

### Example

```bash
python app.py --video 0 --confidence 0.9
```

Press **Q** to exit the application.

---

# Model Checkpoint

The trained model is saved together with the class mapping:

```python
model_data = {
    "model_state_dict": net.state_dict(),
    "class_to_idx": train_data.class_to_idx
}
```

The resulting checkpoint is:

```text
emotiondetecter_model.pth
```

This allows the trained network to be loaded without retraining.

---

# Dataset

The model was trained using an emotion image dataset organized using PyTorch's `ImageFolder`.

The expected structure is:

```text
emotiondata/
│
├── train/
│   ├── Angry/
│   ├── Fearful/
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   └── Surprised/
│
└── test/
    ├── Angry/
    ├── Fearful/
    ├── Happy/
    ├── Neutral/
    ├── Sad/
    └── Surprised/
```

`ImageFolder` automatically creates the class-to-index mapping from the directory structure.

---

# Technologies

* **Python**
* **PyTorch**
* **Torchvision**
* **OpenCV**
* **MTCNN / facenet-pytorch**
* **NumPy**
* **PIL**
* **CUDA**

---

# What I Learned

This project was a major step in my computer vision work because it moved beyond simply training an image classifier.

I learned and implemented:

* Convolutional neural networks
* ResNet architecture
* Residual connections
* Identity/downsample paths
* Batch normalization
* Global average pooling
* Data augmentation
* Random erasing
* MixUp
* AdamW
* Cosine learning-rate scheduling
* Mixed-precision training
* Model checkpointing
* Softmax probability distributions
* Face detection
* Real-time computer vision
* OpenCV video processing
* Multi-face detection
* Integrating multiple ML models into a single pipeline

The biggest part of the project was combining **two separate computer vision systems**:

```text
MTCNN
Face Detection
     +
ResNet
Emotion Classification
     ↓
Real-Time Emotion Detector
```

Rather than giving the classifier an entire image, the face detector first finds the relevant regions and passes those crops into the emotion model.

---

# Limitations

There are several important limitations to this project.

### Emotion Is Not Directly Observable

The model classifies **facial-expression labels** from the dataset. It does not directly measure a person's internal emotional state.

### Dataset Bias

The model can inherit biases and limitations from the training dataset.

### Limited Classes

Only six emotion categories are represented.

### Real-World Conditions

Performance can degrade with:

* Poor lighting
* Extreme head angles
* Occlusion
* Blurry faces
* Very small faces
* Unusual camera angles
* Expressions that differ from the training data

### CPU Inference

The current application explicitly uses:

```python
device = "cpu"
```

so inference is performed on the CPU.

---

# Possible Improvements

Future versions could include:

* GPU inference
* Better face tracking between frames
* Temporal modeling across multiple frames
* More robust face preprocessing
* Confidence thresholding for emotion predictions
* Grad-CAM visualizations
* A larger and more diverse dataset
* Better evaluation metrics such as precision, recall, and F1
* Model quantization for faster CPU inference
* Comparing the custom ResNet against pretrained architectures
* Adding an "uncertain" prediction when the model's confidence is low

---

# Project Structure

```text
Emotion-Detector/
│
├── model.py
├── app.py
├── utils.py
├── emotiondetecter_model.pth
└── README.md
```

---

# Running the Project

Install the dependencies:

```bash
pip install torch torchvision opencv-python numpy facenet-pytorch
```

Then run the webcam version:

```bash
python app.py
```

Or explicitly specify the webcam:

```bash
python app.py --video 0
```

For a video file:

```bash
python app.py --video path/to/video.mp4
```

Adjust the face detection confidence if needed:

```bash
python app.py --confidence 0.9
```

Press **Q** to close the application.

---

## Author

Built with **PyTorch and OpenCV** as a computer vision project focused on learning how CNN architectures can be trained and combined with real-time face detection to create an end-to-end computer vision application.
