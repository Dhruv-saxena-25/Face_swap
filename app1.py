import os
import sys

import logging
import streamlit as st 

from face_app.face_app import FaceAPP



logger = logging.getLogger(__name__)


IMG_FOLDER = os.path.join("static", "images")

output_path = os.path.join(IMG_FOLDER, 'modefied.jpg')



ROOT = os.getcwd()
source_path = os.path.join(ROOT, 'images', 'src_img.png')
dest_path = os.path.join(ROOT, 'images', 'dst_img.jpg')



st.markdown(
    """
<style>
    div[data-testid="stSidebarUserContent"] {
        padding: 2rem 1.5rem;
    }
    div[data-testid="stAppViewBlockContainer"]{
        max-width: 66rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.title("Upload Images")

uploaded_image1 = st.sidebar.file_uploader("Upload Source Image")
uploaded_image2 = st.sidebar.file_uploader("Upload Destination Image")

if uploaded_image1 and uploaded_image2:
        # Save uploaded images to uploads folder
        image1_path = os.path.join(source_path, uploaded_image1.name)
        image2_path = os.path.join(dest_path, uploaded_image2.name)

        logger.info('Image accepted form user')

        if st.button("Swap Image"):

            face_app = FaceAPP(source_path, dest_path, output_path)
            face_app.run()    
            st.header("Generated Image")
            display_image = os.path.join("./static\images", 'modefied.jpg')
            st.image(display_image)

