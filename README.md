# 🎯 Face Detection & Embedding System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete end-to-end pipeline for **face detection** and **face embedding generation** using state-of-the-art deep learning models. Train a custom YOLOv8 face detector on WIDER FACE dataset and generate 512-dimensional face embeddings using FaceNet.

![Face Detection Demo](https://user-images.githubusercontent.com/placeholder.png)

## 🌟 Features

- ✅ **YOLOv8-based Face Detection** - Fast and accurate face localization
- ✅ **FaceNet Embeddings** - Generate 512-dim L2-normalized face embeddings
- ✅ **WIDER FACE Training** - Train on industry-standard face detection dataset
- ✅ **GPU Accelerated** - Automatic GPU detection and utilization
- ✅ **Batch Processing** - Process multiple images efficiently
- ✅ **JSON Export** - Save embeddings and metadata in structured format
- ✅ **Kaggle Ready** - Fully compatible with Kaggle notebooks
- ✅ **Production Ready** - Clean, modular, and reusable code

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **mAP50** | 66.0% |
| **Precision** | 83.4% |
| **Recall** | 57.6% |
| **Model Size** | 5.94 MB |
| **Inference Speed** | ~50ms per image (GPU) |

## 🚀 Quick Start

### Installation

```bash
pip install ultralytics facenet-pytorch opencv-python torch torchvision
```

### Usage

```python
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import torch

# Load trained face detector
face_detector = YOLO('face_detector_best.pt')

# Load embedding model
embedding_model = InceptionResnetV1(pretrained='vggface2').eval()

# Detect faces and generate embeddings
results = face_detector('group_photo.jpg')

# Process detections
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    face_crop = img[int(y1):int(y2), int(x1):int(x2)]
    # Generate embedding...
```

## 📁 Project Structure

```
face_detection/
├── face_detection_training.ipynb  # Complete training & inference notebook
├── wider_face_annotations/        # WIDER FACE annotation files
│   └── wider_face_split/
│       ├── wider_face_train_bbx_gt.txt
│       └── wider_face_val_bbx_gt.txt
├── WIDER_train/                   # Training images
├── WIDER_val/                     # Validation images
├── WIDER_test/                    # Test images
└── README.md
```

## 🎓 Training

The notebook includes a complete training pipeline:

1. **Parse WIDER FACE Annotations** - Handle complex annotation format
2. **Convert to YOLO Format** - Normalize bounding boxes
3. **Train YOLOv8** - 50 epochs with early stopping
4. **Validate Model** - Check metrics and performance
5. **Export Weights** - Save best model for inference

### Training Configuration

```python
epochs = 50
batch_size = 16
image_size = 640
patience = 10  # Early stopping
```

### Dataset

- **Training samples**: 5,000 images
- **Validation samples**: 1,000 images
- **Source**: [WIDER FACE Dataset](http://shuoyang1213.me/WIDERFACE/)

## 🔍 Inference Pipeline

### 1. Face Detection
```python
system = FaceDetectionSystem('face_detector_best.pt')
results = system.process_image('group_photo.jpg', output_dir='./faces')
```

### 2. Output Structure

```json
{
  "num_faces": 14,
  "faces": [
    {
      "bbox": [120, 80, 220, 180],
      "confidence": 0.874,
      "embedding": [0.123, -0.456, ...],  // 512-dim vector
      "saved_path": "faces/face_0.jpg"
    }
  ]
}
```

### 3. Batch Processing

```python
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
batch_results = process_image_batch(image_paths)
```

## 📈 Key Components

### FaceDetectionSystem Class

```python
class FaceDetectionSystem:
    def __init__(self, detector_path, device='cuda')
    def detect_faces(self, image_path, conf_threshold=0.5)
    def generate_embedding(self, face_crop)
    def process_image(self, image_path, output_dir=None)
```

### Functions

- `parse_wider_annotations()` - Parse WIDER FACE format
- `convert_to_yolo_format()` - Convert annotations to YOLO
- `detect_and_embed_faces()` - Complete detection + embedding pipeline
- `process_image_batch()` - Batch processing
- `visualize_detections()` - Visualize detection results

## 🎯 Use Cases

- **Face Recognition Systems** - Generate embeddings for face matching
- **Attendance Systems** - Detect and recognize faces in group photos
- **Security Applications** - Real-time face detection and tracking
- **Photo Organization** - Auto-tag people in photo galleries
- **Identity Verification** - Compare face embeddings for authentication

## 📊 Results & Metrics

Training converges well with consistent improvement:

```
Epoch  Box Loss  Precision  Recall   mAP50
  46    1.505     0.830     0.574    0.656
  47    1.498     0.834     0.571    0.658
  48    1.501     0.834     0.575    0.659
  49    1.480     0.832     0.576    0.660
  50    1.478     0.834     0.576    0.660
```

## 🛠️ Technical Details

### Model Architecture
- **Detector**: YOLOv8n (Nano variant)
- **Embedding**: InceptionResnetV1 (VGGFace2 pretrained)
- **Input Size**: 640x640 (detection), 160x160 (embedding)
- **Output**: 512-dimensional L2-normalized embeddings

### Requirements
```
ultralytics>=8.0.0
facenet-pytorch>=2.5.0
opencv-python>=4.5.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pillow>=9.0.0
pyyaml>=6.0
```

## 🌐 Kaggle Deployment

This notebook is optimized for Kaggle:

1. Upload `face_detection_training.ipynb`
2. Add WIDER FACE dataset as input
3. Enable GPU accelerator
4. Run all cells
5. Download trained model and results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [WIDER FACE Dataset](http://shuoyang1213.me/WIDERFACE/) - Face detection benchmark
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [FaceNet PyTorch](https://github.com/timesler/facenet-pytorch) - Face recognition library
- [VGGFace2](https://github.com/ox-vgg/vgg_face2) - Face recognition dataset

## 📧 Contact

Project Link: [https://github.com/Rahuly1606/face-detection](https://github.com/Rahuly1606/face-detection)

## 🔮 Future Enhancements

- [ ] Real-time webcam face detection
- [ ] Face tracking across video frames
- [ ] Multi-face comparison and clustering
- [ ] REST API deployment
- [ ] Docker containerization
- [ ] Mobile optimization (ONNX export)
- [ ] Face quality assessment
- [ ] Age and gender prediction

---

**⭐ If you find this project useful, please consider giving it a star!**
