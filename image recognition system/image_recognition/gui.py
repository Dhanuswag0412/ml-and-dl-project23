import streamlit as st
import cv2
import pandas as pd
import os
import sys
import tempfile

# Add parent directory of image_recognition to sys.path to allow absolute imports when run directly as a script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from image_recognition.detector import detect_objects

def main():
    st.set_page_config(
        page_title="Image Classification and Object Recognition",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Deep Learning Based Image Classification and Object Recognition System")
    st.write("Upload an image to detect and recognize objects using YOLOv8.")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, uploaded_file.name)

        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.subheader("Original Image")
        st.image(image_path, use_container_width=True)

        result_image, detected_objects, total_objects = detect_objects(image_path)

        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        st.subheader("Detected Image")
        st.image(result_image, use_container_width=True)

        st.success(f"Total Objects Detected : {total_objects}")

        if len(detected_objects) > 0:
            st.subheader("Detected Objects")
            df = pd.DataFrame(detected_objects)
            st.dataframe(df, use_container_width=True)

            output_path = os.path.join(temp_dir, "detected_image.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 Download Detected Image",
                    data=file,
                    file_name="Detected_Image.jpg",
                    mime="image/jpeg"
                )

def run_app():
    import streamlit.web.cli as stcli
    gui_path = os.path.abspath(__file__)
    sys.argv = ["streamlit", "run", gui_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
