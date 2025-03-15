import streamlit as st
import requests
import os
import json
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="UBL Product Detection",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("UBL Product Detection System")

st.markdown("""
### Model Description
The model is trained to detect and classify various Unilever Bangladesh Limited (UBL) products in retail environments, 
distinguishing them separately. The model (UBLModel_v2.pt) focuses on sachets, specifically:
- Horlicks (CTSL 18ml)
- Clear Men's Shampoo (5ml)
- Dove Conditioner (IRP DOLCE 7ml)
""")

# Create two columns for the layout
col1, col2 = st.columns([1, 1])

with col1:
    # Use Case 1 Button with description
    st.markdown("### Use Case 1: Product Detection")
    st.markdown("""
    **Capabilities:**
    - Detect and classify UBL sachet products
    - Process single or multiple images
    - Generate detailed analysis reports
    - Provide accuracy metrics per detection
    """)
    
    # Initialize session state variables
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # File uploader outside button condition
    uploaded_files = st.file_uploader(
        "Drag and drop images here",
        accept_multiple_files=True,
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'],
        key='file_uploader'
    )
    
    # Update session state
    st.session_state.uploaded_files = uploaded_files
    
    # Analyze button
# Previous code remains the same...

# Previous code remains the same...

    # Analyze button
    analyze_button = st.button("Analyze Images", type="primary", use_container_width=True)

    if st.session_state.uploaded_files and analyze_button:
        # Save uploaded files temporarily
        temp_paths = []
        for file in uploaded_files:
            temp_path = os.path.join(os.getcwd(), file.name)
            with open(temp_path, 'wb') as f:
                f.write(file.getbuffer())
            temp_paths.append(temp_path)
        
        # Call detect-images endpoint
        response = requests.post(
            "http://localhost:8000/detect-images",
            data={"image_paths": temp_paths, "return_images": "true"}
        )
        
        # Clean up temp files
        for path in temp_paths:
            os.remove(path)
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.result = result
            st.session_state.output_dir = result['output_directory']
            st.session_state.analysis_complete = True
            
            # Display results for all images
            if 'result' in st.session_state and 'output_dir' in st.session_state:
                st.success(f"Successfully processed {len(temp_paths)} images!")
                
                # Display summary for all images
                summary_files = [f for f in os.listdir(st.session_state.output_dir) if f.endswith('.txt')]
                if summary_files:
                    summary_txt = os.path.join(st.session_state.output_dir, summary_files[0])
                    if os.path.exists(summary_txt):
                        with open(summary_txt, 'r') as f:
                            summary_content = f.read()
                        
                        # Display overall summary
                        st.markdown("### Overall Summary")
                        overall_quality = next((line for line in summary_content.split('\n') 
                                              if 'Detection Quality:' in line), '')
                        overall_accuracy = next((line for line in summary_content.split('\n') 
                                               if 'Accuracy:' in line), '')
                        st.markdown(f"**{overall_quality}**")
                        st.markdown(f"**{overall_accuracy}**")
                        
                        # Display detailed results for each image
                        st.markdown("### Detailed Results")
                        for image_path in temp_paths:
                            st.markdown(f"#### {os.path.basename(image_path)}")
                            # Display image-specific results here
                            # (You can add more detailed results for each image)
        else:
            st.error(f"Error: {response.text}")

# Display results in the second column
with col2:
    if 'result' in st.session_state and 'output_dir' in st.session_state:
        if st.session_state.analysis_complete:
            st.success("Analysis complete!")
            st.session_state.analysis_complete = False
        st.markdown("### Analysis Results")
        
        # Read and display the summary
        summary_files = [f for f in os.listdir(st.session_state.output_dir) if f.endswith('.txt')]
        
        if summary_files:
            summary_txt = os.path.join(st.session_state.output_dir, summary_files[0])
            
            if os.path.exists(summary_txt):
                with open(summary_txt, 'r') as f:
                    summary_content = f.read()
                
                # Extract and display key metrics
                quality_line = next((line for line in summary_content.split('\n') 
                                  if 'Detection Quality:' in line), '')
                accuracy_line = next((line for line in summary_content.split('\n') 
                                   if 'Accuracy:' in line), '')
                
                # Create a styled table for analysis results
                st.markdown("### Analysis Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{quality_line}**")
                with col2:
                    st.markdown(f"**{accuracy_line}**")
                
                # Display object counts in a table
                st.markdown("### Object Counts")
                
                # Initialize class counts
                class_counts = {
                    'ClearMen5ml': 0,
                    'DoveConditionerIRPDOLCE7ml': 0,
                    'HorlicksCTSL18ml': 0
                }
                
                for line in summary_content.split('\n'):
                    for class_name in class_counts.keys():
                        if class_name in line:
                            try:
                                count = int(line.split(':')[1].strip())
                                class_counts[class_name] += count
                            except:
                                pass
                
                # Create a DataFrame for better visualization
                import pandas as pd
                df = pd.DataFrame([
                    {"Product": "Clear Men's Shampoo (5ml)", "Count": class_counts['ClearMen5ml']},
                    {"Product": "Dove Conditioner (IRP DOLCE 7ml)", "Count": class_counts['DoveConditionerIRPDOLCE7ml']},
                    {"Product": "Horlicks (CTSL 18ml)", "Count": class_counts['HorlicksCTSL18ml']}
                ])
                st.table(df)
        
        # Display processed images in a grid
        st.markdown("### Processed Images")
        image_files = [f for f in os.listdir(st.session_state.output_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Create image grid
        cols = st.columns(2)
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(st.session_state.output_dir, image_file)
            with cols[idx % 2]:
                st.image(image_path, caption=image_file, use_container_width=True)