import os
import argparse
import cv2
import pandas as pd
from ultralytics import YOLO

# Get the directory where this file resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
local_model_path = os.path.join(BASE_DIR, "yolov8n.pt")
MODEL_PATH = local_model_path if os.path.exists(local_model_path) else "yolov8n.pt"

model = YOLO(MODEL_PATH)

def detect_objects(image_path):
    # Perform Detection
    results = model(image_path)

    # Draw Bounding Boxes
    annotated_image = results[0].plot()

    detected_objects = []

    # Get Object Details
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        object_name = model.names[class_id]
        confidence = float(box.conf[0])

        detected_objects.append({
            "Object": object_name,
            "Confidence": round(confidence * 100, 2)
        })

    total_objects = len(detected_objects)

    return annotated_image, detected_objects, total_objects

def main():
    parser = argparse.ArgumentParser(description="Detect and recognize objects in an image using YOLOv8.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--output", "-o",
        help="Path to save the annotated image (optional)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at '{args.image}'")
        return
        
    print(f"Running object detection on: {args.image}")
    annotated_image, detected, total = detect_objects(args.image)
    print(f"Total Objects Detected: {total}")
    
    if total > 0:
        df = pd.DataFrame(detected)
        print(df.to_string(index=False))
        
    if args.output:
        cv2.imwrite(args.output, annotated_image)
        print(f"Saved annotated image to: {args.output}")

if __name__ == "__main__":
    main()
