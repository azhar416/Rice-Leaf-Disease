import streamlit as st
import numpy as np
import pandas as pd
import cv2
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import random
from rembg import remove

st.set_page_config(layout="wide")

def clahe_enc(img):
  img_hsv = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2HSV)
  h, s, v = cv2.split(img_hsv)
  clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8,8))
  v = clahe.apply(v)
  img_hsv = np.dstack((h,s,v))
  img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
  return img_bgr
    
def denoise(img):
    denoise_img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,15)
    return denoise_img

def remove_bg(img):
    rembg_img = remove(img)
    return rembg_img

def draw_contours(img, pred):
    denoise_img = denoise(img.copy())
    # enhance_img = clahe_enc(denoise_img)
    
    if pred == 0:
        rm_img = remove_bg(denoise_img)
        low = np.array([0, 40, 150])
        upp = np.array([28, 100, 255])
    elif pred == 1:
        rm_img = remove_bg(denoise_img)
        low = np.array([4, 40, 118])
        upp = np.array([30, 189, 255])
    elif pred == 2:
        rm_img = remove_bg(denoise_img)
        low = np.array([2, 20, 100])
        upp = np.array([23, 255, 200])
    elif pred == 3:
        rm_img = denoise_img
        low = np.array([13, 77, 160])
        upp = np.array([27, 255, 255])
    
    hsv_img = cv2.cvtColor(rm_img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img, low, upp)
    masked_img = img.copy()
    masked_img[mask != 0] = [0, 0, 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    drawed = cv2.drawContours(img.copy(), contours, -1, (255,0,0), 2)
    return drawed

st.header('Rice Leaf Disease Prediction')
st.subheader('Daffa Muhamad Azhar')
uploaded_file = st.file_uploader('Upload', type=['jpg', 'png', 'jpeg', 'tif'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    pred_image = cv2.resize(image.copy(), (300, 300))

    model = load_model('..\..\Model\Model.hdf5')
    datagen = ImageDataGenerator()
    image_array = np.array(pred_image)
    image_array = image_array.reshape(1, 300, 300, 3)
    image_array = image_array.astype('float32')
    generator = datagen.flow(image_array, batch_size=1)

    prob = model.predict(generator)
    prediction = np.argmax(prob,axis=1)
    
    drawn_img = draw_contours(image, prediction)

    label = [
        'Bacterialblight',
        'Blast',
        'Brownspot',
        'Tungro',
    ]

    col1,col2 = st.columns(2)
    with col1:
        st.header('Uploaded Image')
        st.image(drawn_img)
    with col2:
        st.header('Prediction')
        st.subheader(label[prediction[0]])