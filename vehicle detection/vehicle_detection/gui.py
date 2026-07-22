import streamlit as st
from PIL import Image
import cv2
import os
import sys
import tempfile

# Add parent directory of vehicle_detection to sys.path to allow absolute imports when run directly as a script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from vehicle_detection.detector import detect_vehicles

def main():
    st.set_page_config(page_title="Vehicle Detection", page_icon="🚗")

    st.title("🚗 Vehicle Detection using YOLOv8")
    st.write("Upload an image to detect vehicles.")

    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Create temp folder for storing the uploaded file
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, uploaded_file.name)

        # Save uploaded image
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.subheader("Original Image")
        st.image(image_path, use_container_width=True)

        # Detect vehicles
        result_image, vehicle_count = detect_vehicles(image_path)

        st.subheader("Detected Image")

        # Convert BGR to RGB
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        st.image(result_image, use_container_width=True)

        st.success(f"Total Vehicles Detected: {vehicle_count}")

def run_app():
    import streamlit.web.cli as stcli
    gui_path = os.path.abspath(__file__)
    sys.argv = ["streamlit", "run", gui_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
