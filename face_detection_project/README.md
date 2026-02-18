# Face Detection System - YOLOv8

Production-ready face detection system using YOLOv8 trained model.

## Project Structure

```
face_detection_project/
├── main.py                     # Main application script
├── face_detector_best.pt       # Trained YOLOv8 model weights
├── input_images/               # Place your input images here
├── output_images/              # Annotated images saved here
├── results.json                # Detection results in JSON format
└── requirements.txt            # Python dependencies
```

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify GPU support (optional but recommended):**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Usage

### Basic Usage

Place your images in the `input_images/` folder and run:

```bash
python main.py --input input_images --output output_images
```

### Advanced Usage

```bash
python main.py --input <input_folder> --output <output_folder> --conf <threshold> --results <json_file>
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | `input_images` | Path to input images folder |
| `--output` | `-o` | `output_images` | Path to save annotated images |
| `--model` | `-m` | `face_detector_best.pt` | Path to model weights |
| `--conf` | `-c` | `0.25` | Confidence threshold (0.0-1.0) |
| `--results` | `-r` | `results.json` | Path to output JSON file |

### Examples

**Process images with custom confidence threshold:**
```bash
python main.py --input my_photos --output results --conf 0.5
```

**Use different model weights:**
```bash
python main.py --model custom_model.pt --input images --output detected
```

**Specify custom results file:**
```bash
python main.py --input images --output results --results detections.json
```

## Output

### 1. Annotated Images
- Saved in the specified output folder
- Green bounding boxes around detected faces
- Confidence scores displayed above each box

### 2. JSON Results File
Contains detection data for each image:
```json
[
  {
    "image_name": "photo1.jpg",
    "faces_detected": 2,
    "bounding_boxes": [
      {
        "x1": 120.5,
        "y1": 80.3,
        "x2": 250.7,
        "y2": 210.4,
        "confidence": 0.95
      }
    ]
  }
]
```

### 3. Console Summary
- Progress for each image
- Total images processed
- Total faces detected
- Average faces per image

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)

## Features

- ✅ Automatic GPU detection and usage
- ✅ Batch processing of multiple images
- ✅ Configurable confidence threshold
- ✅ Detailed JSON output with bounding box coordinates
- ✅ Error handling for corrupted images
- ✅ Progress tracking and statistics
- ✅ Modular and maintainable code structure

## Troubleshooting

**Issue: "Model file not found"**
- Ensure `face_detector_best.pt` is in the project directory
- Or specify the correct path using `--model`

**Issue: "No valid images found"**
- Check that your input folder contains supported image formats
- Verify the input path is correct

**Issue: Slow processing**
- Install CUDA-compatible PyTorch for GPU acceleration
- Visit: https://pytorch.org/get-started/locally/

## Performance

- **With GPU:** ~50-100 images/second (depending on image size and GPU)
- **CPU only:** ~5-10 images/second

## License

This project uses the YOLOv8 model from Ultralytics (AGPL-3.0 license).
