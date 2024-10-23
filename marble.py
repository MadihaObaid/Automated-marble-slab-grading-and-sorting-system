import streamlit as st
import numpy as np
from keras.models import load_model
import keras.utils
import keras.utils
from PIL import Image, ImageOps
import cv2
from keras.applications.imagenet_utils import preprocess_input

# Load the trained LSTM model
model = load_model('MarbleModel.tf')

st.markdown('<h1 style="color:gray;"> Image classification model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies marble into following categories</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> crack, dot, good, joint </h3>', unsafe_allow_html=True)

upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
  im= Image.open(upload)
  img= np.asarray(im)
  image= cv2.resize(img,(48, 48))
  img= preprocess_input(image)
  img= np.expand_dims(img, 0)
  c1.header('Input Image')
  c1.image(im)
  c1.write(img.shape)

 # prediction on model
  vgg_preds = model.predict(img)
  vgg_pred_classes = np.argmax(vgg_preds, axis=1)
  c2.header('Output')
  c2.subheader('Predicted class :')
  labels_str = ['crack', 'dot', 'good', 'joint']
  c2.write(vgg_pred_classes[0])
  c2.write("The marble is {}".format(labels_str[vgg_pred_classes[0]]))
