import streamlit as st
import cv2 as cv
from utility import apply_histogram_equalization, rghs, NUCE
from PIL import Image
import numpy as np

st.title("Underwater Image Enhancement")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Resize image for processing if needed
    resized_image = cv.resize(image_np, (256, 256))

    # Apply enhancements
    he_img = apply_histogram_equalization(resized_image)
    rghs_img = rghs(resized_image)

    with st.spinner("Enhancing using PSO..."):
        nuce_img = NUCE(resized_image)

    # Display all 3 enhancements in a row
    st.subheader("Enhanced Images")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(he_img, caption="Histogram Equalization", use_column_width=True)

    with col2:
        st.image(rghs_img, caption="RGHS", use_column_width=True)

    with col3:
        st.image(nuce_img, caption="NUCE (PSO)", use_column_width=True)


