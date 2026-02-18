#!/usr/bin/env python3
"""
Face Detection System using YOLOv8
Processes images from input folder and saves annotated results with detection data.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import torch
from ultralytics import YOLO


def load_model(model_path: str) -> YOLO:
    """
    Load the YOLOv8 face detection model.
    
    Args:
        model_path: Path to the model weights file
        
    Returns:
        Loaded YOLO model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = YOLO(model_path)
    model.to(device)
    
    print(f"Model loaded successfully from: {model_path}")
    return model


def process_image(model: YOLO, image_path: str, conf_threshold: float = 0.25) -> Tuple[any, List[Dict]]:
    """
    Process a single image and detect faces.
    
    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Tuple of (annotated image, list of detection dictionaries)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Run inference
    results = model(img, conf=conf_threshold, verbose=False)
    
    # Extract detections
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            
            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": round(confidence, 3)
            })
            
            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Face {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
            cv2.putText(img, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return img, detections


def process_folder(model: YOLO, input_folder: str, output_folder: str, 
                   conf_threshold: float = 0.25) -> List[Dict]:
    """
    Process all images in a folder.
    
    Args:
        model: Loaded YOLO model
        input_folder: Path to input images folder
        output_folder: Path to save output images
        conf_threshold: Confidence threshold for detections
        
    Returns:
        List of detection results for all images
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Get all image files
    image_files = [f for f in os.listdir(input_folder) 
                   if Path(f).suffix.lower() in valid_extensions]
    
    if not image_files:
        print(f"No valid images found in {input_folder}")
        return []
    
    print(f"\nFound {len(image_files)} images to process")
    print("-" * 50)
    
    all_results = []
    total_faces = 0
    processed_count = 0
    
    for idx, filename in enumerate(image_files, 1):
        try:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Process image
            annotated_img, detections = process_image(model, input_path, conf_threshold)
            
            # Save annotated image
            cv2.imwrite(output_path, annotated_img)
            
            # Store results
            result = {
                "image_name": filename,
                "faces_detected": len(detections),
                "bounding_boxes": detections
            }
            all_results.append(result)
            
            total_faces += len(detections)
            processed_count += 1
            
            print(f"[{idx}/{len(image_files)}] {filename}: {len(detections)} face(s) detected")
            
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] Error processing {filename}: {str(e)}")
            continue
    
    print("-" * 50)
    print(f"\nProcessing complete!")
    print(f"Images processed: {processed_count}/{len(image_files)}")
    print(f"Total faces detected: {total_faces}")
    print(f"Average faces per image: {total_faces/processed_count:.2f}" if processed_count > 0 else "N/A")
    
    return all_results


def save_results(results: List[Dict], output_file: str) -> None:
    """
    Save detection results to JSON file.
    
    Args:
        results: List of detection results
        output_file: Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Face Detection System using YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input input_images --output output_images
  python main.py --input ../test_images --output ../results --conf 0.5
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='input_images',
        help='Path to input images folder (default: input_images)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output_images',
        help='Path to output images folder (default: output_images)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='face_detector_best.pt',
        help='Path to model weights (default: face_detector_best.pt)'
    )
    
    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    
    parser.add_argument(
        '--results', '-r',
        type=str,
        default='results.json',
        help='Path to output JSON file (default: results.json)'
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.exists(args.input):
        print(f"Error: Input folder not found: {args.input}")
        return
    
    print("=" * 50)
    print("Face Detection System - YOLOv8")
    print("=" * 50)
    
    try:
        # Load model
        model = load_model(args.model)
        
        # Process all images
        results = process_folder(model, args.input, args.output, args.conf)
        
        # Save results to JSON
        if results:
            save_results(results, args.results)
        else:
            print("\nNo results to save.")
        
        print("\n" + "=" * 50)
        print("Processing completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
