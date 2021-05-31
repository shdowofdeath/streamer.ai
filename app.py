import streamlit as st
import numpy as np
from PIL import Image , ImageOps
from keras.preprocessing import image
import cv2 
from tensorflow import keras
import tensorflow as tf
import os
st.title('Storganizer')
st.balloons()
# st.write("בבקשה להעלות תמונה  ")



def predictor(nameos):
    def averagecolor(image):
        return np.mean(image, axis=(0, 1))
    trainX2 = []
    trainY2 = []
    
    path = "Clothes2/"
    for label in ('Black', 'Blue', 'Brown', 'Green', 'Orange', 'Pink','Red', 'White', 'Yellow'):
    #     print ("Loading training images for the label: "+label)
        
        #Load all images inside the subfolder
        for filename in os.listdir(path+"/"+label+"/"): 
            img = cv2.imread(path+label+"/"+filename)
            img_features = averagecolor(img)
            trainX2.append(img_features)
            trainY2.append(label)
            
    path = "test"
    filenames = []
    predictedY = []
    for filename in os.listdir(path+"/"): 
        img = cv2.imread(path+"/"+filename)
        img_features = averagecolor(img)
        calculated_distances = []
        for card in (trainX2):
            calculated_distances.append(np.linalg.norm(img_features-card))
        prediction =  trainY2[np.argmin(calculated_distances)]
        
    #     print (filename + ": " + prediction)
        filenames.append(filename)
        predictedY.append(prediction)
    model = tf.keras.models.load_model("elhays-test-modle.h5")
    lbls = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    #filename = "Buttoned.jpg"
    filename = nameos
    predictedY = []
    img = cv2.imread(filename)
    img_features = averagecolor(img)
    calculated_distances= []
    for card in (trainX2):
        calculated_distances.append(np.linalg.norm(img_features-card))
    prediction =  trainY2[np.argmin(calculated_distances)]
    print ('Color:', prediction)
    filenames.append(filename)
    # plt.imshow(img)
    
    x=[]
    img = image.load_img(filename, color_mode = "grayscale", target_size=(28, 28))
    img = image.img_to_array(img)
    img = img.reshape(28, 28)
    img = img.astype('float32')
    img = (255-img)/255.0
    x.append(img)
    x=np.array(x)
    result = np.argmax(model.predict(x), axis=-1)
    for i in range(len(result)):
     print('Type:', lbls[result[i]])
     st.write('Type:', lbls[result[i]])
     st.write('Color:', prediction)
    st.image(img, caption="The caption", use_column_width=True)
import uuid
img_file_buffer = st.file_uploader("בבקשה להעלות תמונה  ")
if img_file_buffer is not None:
    image_local = Image.open(img_file_buffer)
    nameos  = uuid.uuid1()
    nameos = str(nameos)+'.jpg'
    im1 = image_local.save(nameos)
    #img_array = np.array(image_local) # if you want to pass it to OpenCV
    #st.image(image, caption="The caption", use_column_width=True)
    predictor(nameos)
