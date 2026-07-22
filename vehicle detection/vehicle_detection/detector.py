import os
import argparse
import cv2
from ultralytics import YOLO

# Get the directory where this file resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

model = YOLO(MODEL_PATH)

vehicle_classes = ["car", "bus", "truck", "motorcycle", "bicycle"]

def detect_vehicles(image_path):
    results = model(image_path)
    annotated_image = results[0].plot()

    vehicle_count = 0
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]

        if label in vehicle_classes:
            vehicle_count += 1

    return annotated_image, vehicle_count

def main():
    parser = argparse.ArgumentParser(description="Detect vehicles in an image using YOLOv8.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--output", "-o",
        help="Path to save the annotated image (optional)"
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display the annotated image in a window"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at '{args.image}'")
        return
        
    print(f"Running vehicle detection on: {args.image}")
    annotated_image, count = detect_vehicles(args.image)
    print(f"Total Vehicles Detected: {count}")
    
    if args.output:
        cv2.imwrite(args.output, annotated_image)
        print(f"Saved annotated image to: {args.output}")
        
    if args.show:
        cv2.imshow("Vehicle Detection", annotated_image)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
