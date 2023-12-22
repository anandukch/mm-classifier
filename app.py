import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("./models/model1.h5")

st.title("Image Upload")
uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    st.image(uploaded_file, width=200)
    with open("image.jpg", "wb") as f:
        f.write(uploaded_file.read())


def process_ip(image):
    img = cv2.imread(str(image))
    return np.array([cv2.resize(img, (180, 180))]) / 255


classes = ["mammooty", "mohanlal"]

if st.button("Classify"):
    prediction = model.predict(process_ip("./image.jpg"))
    print(prediction)
    index = np.argmax(prediction)
    st.title("the person is " + classes[index])
