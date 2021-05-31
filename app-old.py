import streamlit as st
import numpy as np
from PIL import Image , ImageOps
import cv2 
from tensorflow import keras
import tensorflow as tf
st.title('Storganizer')
st.balloons()
# st.write("בבקשה להעלות תמונה  ")

#תיוג צבעים
def averagecolor(image):
    return np.mean(image, axis=(0, 1))


def predict(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    model = keras.models.load_model("model")
    prediction = model.predict(data)
    st.write(np.argmax(prediction))

def predict2(image_array):
    model = tf.keras.models.load_model("elhays-test-modle.h5")
    #st.write(model.summary())
    print(model.summary())
    prediction = model.predict(image_array)
    st.write(np.argmax(prediction))


def predictor(image):
    predictedY = []
    img = cv2.imread(image)
    img_features = averagecolor(img)
    calculated_distances= []
    for card in (trainX2):
       calculated_distances.append(np.linalg.norm(img_features-card))
    prediction =  trainY2[np.argmin(calculated_distances)]
    print ('Color:', prediction)
    filenames.append(filename)
    #plt.imshow(img)

    x=[]
    img = image.load_img(filename, color_mode = "grayscale", target_size=(28, 28))
    img = image.img_to_array(img)
    img = img.reshape(28, 28)
    img = img.astype('float32')
    img = (255-img)/255.0
    x.append(img)
    x=np.array(x)
    result = np.argmax(model.predict(x), axis=-1)
    #model.save('elhays-test-modle.h5')
    for i in range(len(result)):
      print('Type:', lbls[result[i]])

img_file_buffer = st.file_uploader("בבקשה להעלות תמונה  ")
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image) # if you want to pass it to OpenCV
    st.image(image, caption="The caption", use_column_width=True)
    #predict(image)
    #predict2(img_array)
    predictor(image)
