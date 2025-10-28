#!/usr/bin/env python3

import cv2 as cv
import numpy as np
from ultralytics import YOLO
import easyocr
from pathlib import Path
import argparse
import re
import os

def init_ocr(langs=['en'], use_gpu=True):
    """Initialize EasyOCR reader."""
    print(f"Initializing EasyOCR reader (langs={langs}, gpu={use_gpu})...")
    return easyocr.Reader(langs, gpu=use_gpu)

def is_license_plate_text(text):
    """Check if text looks like a license plate."""
    if not text:
        return False
    
    # Clean the text
    text_clean = text.replace(' ', '').replace('-', '').replace('_', '').replace('[', '').replace(']', '').upper()
    
    # License plate patterns - very flexible matching
    if len(text_clean) >= 3:
        # Must contain both digits and letters
        has_digits = any(c.isdigit() for c in text_clean)
        has_letters = any(c.isalpha() for c in text_clean)
        
        if has_digits and has_letters:
            # Various license plate patterns
            # 3 digits + 2 letters (like 537-UG)
            if re.match(r'^\d{3}[A-Z]{2}$', text_clean):
                return True
            # 4 digits + 2 letters
            if re.match(r'^\d{4}[A-Z]{2}$', text_clean):
                return True
            # 3 letters + 4 digits (like KET-9559)
            if re.match(r'^[A-Z]{3}\d{4}$', text_clean):
                return True
            # 3 letters + 3 digits (like KET-3088)
            if re.match(r'^[A-Z]{3}\d{4}$', text_clean):
                return True
            # 3 digits + 3 letters (like 491-N3)
            if re.match(r'^\d{3}[A-Z]{1,3}$', text_clean):
                return True
            # Mixed pattern with at least 4 characters total
            if len(text_clean) >= 4 and has_digits and has_letters:
                return True
    
    return False

def detect_license_plates_improved(input_image_path, output_image_path=None, vehicle_model_path='yolov8s.pt', ocr_langs=['en'], ocr_gpu=True):
    """Improved license plate detection with better accuracy."""
    
    # Initialize models
    print(f"Loading vehicle model from '{vehicle_model_path}'")
    vehicle_model = YOLO(vehicle_model_path)
    
    # Initialize OCR
    reader = init_ocr(ocr_langs, use_gpu=ocr_gpu)
    
    # Load image
    frame = cv.imread(str(input_image_path))
    if frame is None:
        raise RuntimeError(f"Could not load image: {input_image_path}")
    
    # Create a clean copy for output
    output_frame = frame.copy()
    
    vehicles_coco_ids = [2, 3, 5]  # car, motorcycle, bus/truck
    license_plates_found = []
    
    # Detect vehicles first
    print("Detecting vehicles...")
    detections = vehicle_model(frame)[0]
    
    vehicle_regions = []
    for d in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = d
        
        if float(score) < 0.3:
            continue
        if int(class_id) not in vehicles_coco_ids:
            continue
            
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        
        vehicle_regions.append((x1i, y1i, x2i, y2i, score))
        
        # Draw vehicle bounding box
        cv.rectangle(output_frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv.putText(output_frame, f"Vehicle {score:.2f}", (x1i, y1i-10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    print(f"Found {len(vehicle_regions)} vehicles")
    
    # Now do OCR on the entire image to find all text
    print("Reading all text in the image...")
    try:
        all_text_results = reader.readtext(frame)
        print(f"OCR found {len(all_text_results)} text regions")
        
        for i, (bbox, text, confidence) in enumerate(all_text_results):
            print(f"  {i+1}: '{text}' (confidence: {confidence:.2f})")
            
    except Exception as e:
        print(f"OCR error: {e}")
        all_text_results = []
    
    # Filter for license plates
    license_plates = []
    for bbox, text, confidence in all_text_results:
        # Check if text looks like a license plate
        if is_license_plate_text(text) and confidence > 0.2:  # Lower confidence threshold
            license_plates.append((bbox, text, confidence))
            print(f"License plate detected: '{text}' (confidence: {confidence:.2f})")
    
    # Draw license plate bounding boxes with RED color
    for bbox, text, confidence in license_plates:
        # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        # Draw RED bounding box around license plate
        cv.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        
        # Add clean text label with RED color
        clean_text = text.replace('[', '').replace(']', '').replace('_', '')
        label = f"License Plate: {clean_text} ({confidence:.2f})"
        cv.putText(output_frame, label, (int(x1), int(y1)-10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        license_plates_found.append({
            'text': clean_text,
            'confidence': float(confidence),
            'bbox': [int(x1), int(y1), int(x2), int(y2)]
        })
    
    # Save the clean annotated image
    if output_image_path is not None:
        cv.imwrite(output_image_path, output_frame)
        print(f"Clean annotated image saved to: {output_image_path}")
    
    return license_plates_found

def process_all_images_improved(input_dir='inputs', output_dir='outputs', vehicle_model_path='yolov8s.pt', ocr_gpu=True):
    """Process all images with improved license plate detection."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files to process")
    
    # Process each image
    total_plates_found = 0
    for image_file in sorted(image_files):
        print(f"\n{'='*50}")
        print(f"Processing: {image_file.name}")
        print(f"{'='*50}")
        
        # Create output filename
        output_filename = f"improved_detected_{image_file.stem}.jpg"
        output_file_path = output_path / output_filename
        
        try:
            plates = detect_license_plates_improved(
                str(image_file),
                output_image_path=str(output_file_path),
                vehicle_model_path=vehicle_model_path,
                ocr_langs=['en'],
                ocr_gpu=ocr_gpu
            )
            
            print(f"Found {len(plates)} license plates in {image_file.name}")
            for plate in plates:
                print(f"  - {plate['text']} (confidence: {plate['confidence']:.2f})")
            
            total_plates_found += len(plates)
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Batch processing complete!")
    print(f"Processed {len(image_files)} images")
    print(f"Found {total_plates_found} license plates total")
    print(f"Output images saved in: {output_dir}")
    print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(description='Improved License Plate Detection')
    parser.add_argument('--input-dir', default='inputs', help='Input directory containing images')
    parser.add_argument('--output-dir', default='outputs', help='Output directory for results')
    parser.add_argument('--vehicle-model', default='yolov8s.pt', help='YOLOv8 weights for vehicle detection')
    parser.add_argument('--ocr-gpu', action='store_true', help='Use GPU for EasyOCR')
    
    args = parser.parse_args()
    
    process_all_images_improved(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vehicle_model_path=args.vehicle_model,
        ocr_gpu=args.ocr_gpu
    )

if __name__ == '__main__':
    main()
