import streamlit as st
import numpy as np
from PIL import Image
import cv2 

st.title('Storganizer')
st.balloons()
# st.write("בבקשה להעלות תמונה  ")

def averagecolor(image):
    return np.mean(image, axis=(0, 1))


def predict(image):
    img = cv2.imread(image)
    img_features = averagecolor(img)
    calculated_distances = []
    for card in (trainX2):
        calculated_distances.append(np.linalg.norm(img_features-card))
    prediction =  trainY2[np.argmin(calculated_distances)]
    
    print (filename + ": " + prediction)
    filenames.append(filename)
    predictedY.append(prediction)

img_file_buffer = st.file_uploader("בבקשה להעלות תמונה  ")
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image) # if you want to pass it to OpenCV
    st.image(image, caption="The caption", use_column_width=True)
    predict(img_array)
