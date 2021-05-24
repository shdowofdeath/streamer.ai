import streamlit as st
import numpy as np
from PIL import Image

st.title('הפרוייקט של AI')
st.balloons()
# st.write("בבקשה להעלות תמונה  ")

img_file_buffer = st.file_uploader("בבקשה להעלות תמונה  ")
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image) # if you want to pass it to OpenCV
    st.image(image, caption="The caption", use_column_width=True)
