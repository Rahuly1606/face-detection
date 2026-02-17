<div align="center">

# 🎯 Face Detection & Embedding System

**YOLOv8 + FaceNet for Real-Time Face Recognition**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat-square)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-00FFFF.svg?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

</div>

## 📌 Overview

End-to-end face detection and embedding system combining **YOLOv8** for fast face detection and **FaceNet** for generating 512-dimensional face embeddings. Trained on WIDER FACE dataset.

**Key Features:**
- 🎯 Custom YOLOv8 face detector (83% precision, 66% mAP)
- 🧠 512-dim L2-normalized embeddings (FaceNet/VGGFace2)
- ⚡ GPU-accelerated (~50ms per image)
- 📦 Production-ready with batch processing

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **mAP@50** | 66.0% |
| **Precision** | 83.4% |
| **Recall** | 57.6% |
| **Model Size** | 5.94 MB |
| **Inference** | ~50ms (GPU) |

Validated on WIDER FACE - detected 14 faces in test image (confidence: 59.7% - 87.4%)

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Rahuly1606/face-detection.git
cd face-detection
pip install ultralytics facenet-pytorch opencv-python torch torchvision
```

### Basic Usage

```python
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_detector = YOLO('face_detector_best.pt')
embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Detect faces
results = face_detector('photo.jpg')

# Generate embeddings for each face
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    face_crop = image[y1:y2, x1:x2]
    embedding = embedding_model(preprocess(face_crop))  # 512-dim vector
```

### Using FaceDetectionSystem Class

```python
from face_detection_system import FaceDetectionSystem

system = FaceDetectionSystem('face_detector_best.pt')

# Process single image
results = system.process_image('group_photo.jpg', output_dir='./faces')
print(f"Detected {results['num_faces']} faces")

# Batch process
batch_results = system.process_batch(['img1.jpg', 'img2.jpg'])
```

---

## 📁 Project Structure

```
face_detection/
├── face_detection_training.ipynb    # Training & inference notebook
├── wider_face_annotations/          # WIDER FACE annotations
├── WIDER_train/                     # Training images (12,880)
├── WIDER_val/                       # Validation images (3,226)
└── face_detector_best.pt           # Trained model weights
```

---

## 🎓 Training

The notebook includes complete pipeline:
1. Parse WIDER FACE annotations
2. Convert to YOLO format
3. Train YOLOv8 (50 epochs, batch 16)
4. Validate and export model

**Training Config:**
- Base: YOLOv8n
- Input: 640×640
- Optimizer: AdamW
- Time: ~2-3 hours (T4 GPU)

---

## 💡 Use Cases

- **Face Recognition** - Attendance, security systems
- **Photo Organization** - Auto-tag people in galleries
- **Identity Verification** - Compare face embeddings
- **Access Control** - Real-time face detection
- **Social Media** - Face tagging and search

---

## 🔧 Output Format

```json
{
  "num_faces": 3,
  "faces": [
    {
      "bbox": [120, 80, 220, 180],
      "confidence": 0.874,
      "embedding": [0.123, -0.456, ...],  // 512 dimensions
      "saved_path": "faces/face_0.jpg"
    }
  ]
}
```

---

## 🌐 Kaggle Deployment

1. Upload `face_detection_training.ipynb`
2. Add WIDER FACE dataset as input
3. Enable GPU accelerator
4. Run all cells
5. Download trained model

---

## 🙏 Acknowledgments

- [WIDER FACE Dataset](http://shuoyang1213.me/WIDERFACE/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FaceNet PyTorch](https://github.com/timesler/facenet-pytorch)
- [VGGFace2](https://github.com/ox-vgg/vgg_face2)

---

## 📧 Contact

Project Link: [https://github.com/Rahuly1606/face-detection](https://github.com/Rahuly1606/face-detection)

---

**⭐ Star this repo if you find it useful!**
