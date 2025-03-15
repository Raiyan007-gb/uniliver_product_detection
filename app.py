# streamlit_app.py
import streamlit as st
import requests
import cv2
import numpy as np
from pathlib import Path
import pandas as pd

# Set page config
st.set_page_config(
    page_title="UBL Product Detection",
    page_icon="🔍",
    layout="wide"
)

API_URL = "http://localhost:8000/detect/"  # Update this if your API runs on a different host/port

def get_accuracy_percentage(avg_confidence):
    """
    Convert average confidence to an accuracy percentage.
    """
    if avg_confidence > 0.8:
        return 90 + (avg_confidence - 0.8) * 50
    elif avg_confidence > 0.6:
        return 80 + (avg_confidence - 0.6) * 50
    elif avg_confidence > 0.4:
        return 70 + (avg_confidence - 0.4) * 50
    else:
        return max(50, 50 + (avg_confidence - 0.25) * 133)

def get_accuracy_label(avg_confidence):
    """
    Convert average confidence to a human-readable label.
    """
    if avg_confidence > 0.8:
        return "Excellent"
    elif avg_confidence > 0.6:
        return "Good"
    elif avg_confidence > 0.4:
        return "Moderate"
    else:
        return "Low"

def draw_detections(image, detections):
    """Draw detections on the image"""
    img = image.copy()
    for det in detections["detections"]:
        bbox = det["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img, (x, y - text_size[1] - 10), (x + text_size[0], y), (0, 255, 0), -1)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return img

def main():
    # Title and description
    st.title("UBL Product Detection System")
    
    # Model selection
    st.sidebar.title("Model Selection")
    current_model = st.sidebar.radio(
        "Select Detection Model",
        ["Default Detection", "UBL/NON-UBL Classification", "Shelf - Horlicks", "Shelf - Nido"],
        key="model_selection"
    )
    
    # Handle model switching
    if 'previous_model' not in st.session_state:
        st.session_state.previous_model = current_model
    
    if current_model != st.session_state.previous_model:
        model_key = 'ubl_classifier' if current_model == "UBL/NON-UBL Classification" else \
                   'shelf_horlicks' if current_model == "Shelf - Horlicks" else \
                   'shelf_nido' if current_model == "Shelf - Nido" else 'default'
        try:
            response = requests.post(f"http://localhost:8000/switch-model/{model_key}")
            if response.status_code == 200:
                st.success(f"Switched to {current_model} model")
            else:
                st.error("Failed to switch model")
        except Exception as e:
            st.error(f"Error switching model: {str(e)}")
        st.session_state.previous_model = current_model
    
    # Model description based on selection
    if current_model == "Default Detection":
        st.markdown("""
        ### Model Description
        The model is trained to detect and classify various Unilever Bangladesh Limited (UBL) products in retail environments, 
        distinguishing them separately. The model focuses on sachets, specifically:
        - Horlicks (CTSL 18ml)
        - Clear Men's Shampoo (5ml)
        - Dove Conditioner (IRP DOLCE 7ml)
        """)
    elif current_model == "UBL/NON-UBL Classification":
        st.markdown("""
        ### Model Description
        This model is specifically trained to classify products as either UBL or NON-UBL.
        It helps in identifying and distinguishing Unilever Bangladesh Limited products from other products.
        """)
    elif current_model == "Shelf - Horlicks":
        st.markdown("""
        ### Model Description
        This model is specialized in detecting Horlicks products on retail shelves.
        It provides accurate detection and classification of Horlicks products in various shelf arrangements.
        """)
    else:  # Shelf - Nido
        st.markdown("""
        ### Model Description
        This model is specialized in detecting Nido products on retail shelves.
        It provides accurate detection and classification of Nido products in various shelf arrangements.
        """)

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Product Detection")
        st.markdown("""
        **Capabilities:**
        - Detect and classify UBL sachet products
        - Process single or multiple images
        - Generate detailed analysis reports
        - Provide accuracy metrics per detection
        - UBL/NON-UBL product Classification
        """)

        # Initialize session state
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = None
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'file_uploader_key' not in st.session_state:
            st.session_state.file_uploader_key = "file_uploader_0"

        # File uploader
        uploaded_files = st.file_uploader(
            "Drag and drop images here",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'],
            key=st.session_state.file_uploader_key
        )
        
        st.session_state.uploaded_files = uploaded_files

        # Create a container for buttons
        button_col1, button_col2 = st.columns([1, 1])
        
        with button_col1:
            analyze_button = st.button("Analyze Images", type="primary", use_container_width=True)
        
        with button_col2:
            if uploaded_files and st.button("❌ Clear All", key="clear_button", use_container_width=True):
                # Reset relevant session state variables
                st.session_state.uploaded_files = None
                st.session_state.results = None
                st.session_state.analysis_complete = False
                # Correctly increment the numeric part of the key
                current_number = int(st.session_state.file_uploader_key.split('_')[2])  # Extract the number
                st.session_state.file_uploader_key = f"file_uploader_{current_number + 1}"
                st.rerun()

        if st.session_state.uploaded_files and analyze_button:
            files = [("files", (f.name, f, f.type)) for f in st.session_state.uploaded_files]
            
            with st.spinner("Processing images..."):
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    st.session_state.results = response.json()
                    st.session_state.analysis_complete = True
                    st.success(f"Successfully processed {len(st.session_state.uploaded_files)} images!")
                else:
                    st.error(f"Error: {response.text}")

    with col2:
        if st.session_state.results and st.session_state.analysis_complete:
            if st.session_state.analysis_complete:
                st.success("Analysis complete!")
                st.session_state.analysis_complete = False
            
            st.markdown("### Analysis Results")
            results = st.session_state.results
            summary = results["summary"]

            # Summary metrics
            st.markdown("### Analysis Summary")
            scol1, scol2 = st.columns(2)
            with scol1:
                st.markdown(f"**Detection Quality:** {summary['detection_quality']}")
            with scol2:
                st.markdown(f"**Overall Accuracy:** {summary['overall_accuracy']:.1f}%")

            # Object counts table
            st.markdown("### Object Counts")
            summary_df_data = []
            for class_name, count in summary["total_objects_by_class"].items():
                class_confidences = []
                for detection_data in results["detections"].values():
                    for det in detection_data["detections"]:
                        if det["class_name"] == class_name:
                            class_confidences.append(det["confidence"])
                avg_class_conf = sum(class_confidences) / len(class_confidences) if class_confidences else 0.0
                class_accuracy = get_accuracy_percentage(avg_class_conf)
                summary_df_data.append({
                    "Product": class_name,
                    "Count": count,
                    "Accuracy": f"{class_accuracy:.1f}%"
                })
            summary_df = pd.DataFrame(summary_df_data)
            st.table(summary_df)

            # Processed images
            st.markdown("### Processed Images")
            img_cols = st.columns(2)
            for idx, (file, (img_name, detection_data)) in enumerate(zip(st.session_state.uploaded_files, results["detections"].items())):
                file.seek(0)
                img_array = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_with_dets = draw_detections(img, detection_data)
                
                with img_cols[idx % 2]:
                    st.image(img_with_dets, caption=img_name, use_container_width=True)
                
                # Detailed results for each image
                st.markdown(f"#### {img_name}")
                st.write(f"**Accuracy:** {detection_data['estimated_accuracy']:.1f}%")
                st.write(f"**Quality:** {detection_data['quality']}")
                
                df_data = []
                for class_name, count in detection_data["class_counts"].items():
                    class_confidences = [
                        det["confidence"] 
                        for det in detection_data["detections"] 
                        if det["class_name"] == class_name
                    ]
                    avg_class_conf = sum(class_confidences) / len(class_confidences) if class_confidences else 0.0
                    class_accuracy = get_accuracy_percentage(avg_class_conf)
                    df_data.append({
                        "Product": class_name,
                        "Count": count,
                        "Accuracy": f"{class_accuracy:.1f}%"
                    })
                df = pd.DataFrame(df_data)
                st.table(df)

if __name__ == "__main__":
    main()