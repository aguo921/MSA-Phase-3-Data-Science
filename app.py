import streamlit as st

from tensorflow.keras.models import load_model
import tensorflow as tf

import cv2
from PIL import Image

import numpy as np

st.title('Image classification')

model = load_model("output/model3")

def probability(model, data):
    reshaped = data.reshape(1,32,32,3)  # reshape data into model input shape
    logit = model.predict(reshaped)  # predict logit values from input
    prob = tf.nn.softmax(logit)  # convert from logit to probability
    return np.array(prob).reshape(2,)

def print_results(prob):
    if prob[0] > 0.5:
        return "This image is most likely not an airplane with {:.2f} percent confidence.".format(prob[0]*100)
    else:
        return "This image is most likely an airplane with {:.2f} percent confidence.".format(prob[1]*100)

def evaluate_image(model, image):
    image = np.asarray(image.resize((32, 32)))  # resize image
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # remove 4th channel
    prob = probability(model, image)  # get probability
    return print_results(prob)  # display results



uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img)

    st.write(evaluate_image(model, img))
    