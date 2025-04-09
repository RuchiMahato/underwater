import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utility import NUCE, apply_histogram_equalization, rghs

st.set_page_config(page_title="Underwater Enhancement Comparison", layout="wide")

st.title("🌊 Underwater Image Enhancement Comparison")

uploaded_file = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Histogram Equalization")
        he_img = apply_histogram_equalization(image)
        st.image(he_img, use_column_width=True)

    with col2:
        st.markdown("### RGHS")
        rghs_img = rghs(image)
        st.image(rghs_img, use_column_width=True)

    with col3:
        st.markdown("### NUCE (PSO-based)")
        nuce_img = NUCE(image)
        st.image(nuce_img, use_column_width=True)

    st.markdown("---")
    st.subheader("📝 Observation")
    st.write("""
    - **Histogram Equalization**: Improves contrast but may introduce artifacts or unnatural colors.
    - **RGHS**: Enhances brightness but may not correct color cast well.
    - **NUCE (PSO-based)**: Offers better color correction and balanced contrast, resulting in a more visually pleasing image.
    """)
else:
    st.info("Upload an underwater image to begin.")

