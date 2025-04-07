import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Histogram Equalization
def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

# Simulated PSO Enhancement (mock for now)
def pso_enhancement_mock(img):
    enhanced = cv2.convertScaleAbs(img, alpha=1.4, beta=20)
    return enhanced

# Streamlit UI
st.title("Underwater Image Enhancement Comparison")

uploaded_file = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    he_img = histogram_equalization(image_bgr)
    pso_img = pso_enhancement_mock(image_bgr)

    he_img_rgb = cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB)
    pso_img_rgb = cv2.cvtColor(pso_img, cv2.COLOR_BGR2RGB)

    st.subheader("Comparison")
    st.image([image, he_img_rgb, pso_img_rgb], 
             caption=["Original", "Histogram Equalization", "PSO Enhanced"], 
             use_column_width=True)
