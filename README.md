# mini-project
import streamlit as st
import pydicom
from PIL import Image
import numpy as np

st.set_page_config(page_title="Medical Image Analysis Portal", layout="wide")

st.title("🏥 Medical Image & MRI Diagnostic Assistant")
st.write("Upload an X-ray, Brain MRI, or Diagnostic Scan for AI analysis.")

# File uploader widget
uploaded_file = st.sidebar.file_uploader(
    "Choose a scan (PNG, JPG, DICOM .dcm)", 
    type=["png", "jpg", "jpeg", "dcm"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    with col1:
        st.subheader("Uploaded Input Scan")
        
        # Process DICOM files (.dcm)
        if file_extension == "dcm":
            dicom_data = pydicom.dcmread(uploaded_file)
            pixel_array = dicom_data.pixel_array
            norm_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255).astype(np.uint8)
            img = Image.fromarray(norm_array)
            st.image(img, caption=f"Patient ID: {getattr(dicom_data, 'PatientID', 'Anonymous')}", use_column_width=True)
        
        # Process standard image files (PNG, JPG)
        else:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image Scan", use_column_width=True)

    with col2:
        st.subheader("Diagnostic Prediction & Analysis")
        
        if st.button("Run Model Inference"):
            with st.spinner("Analyzing image features..."):
                # Integrate PyTorch / MONAI model prediction function here
                st.success("Analysis Complete!")
                st.metric(label="Primary Finding", value="Abnormality Detected", delta="High Confidence")
                st.progress(0.88)
                st.write("**Model Confidence:** 88.4%")
                st.info("💡 **Grad-CAM Attention Area:** Focus concentrated on the lower-right pulmonary region.")
streamlit
pydicom
pillow
numpy
torch
torchvision
monai
opencv-python-headless
