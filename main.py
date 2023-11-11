import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the model
model = tf.keras.models.load_model("model.h5")


# Set the title of the app
st.title("Image Upload")
# Create a file uploader widget
uploaded_file = st.file_uploader("Upload an image")

# Check if the file has been uploaded
if uploaded_file is not None:
    # Display the image
    st.image(uploaded_file,width=200)

    # Save the image to a file
    with open("image.jpg", "wb") as f:
        f.write(uploaded_file.read())

def process_ip(image):
    img=cv2.imread(str(image))
    return np.array([cv2.resize(img,(180,180))])/255

mm_dict={
    0:'mammooty',
    1:'mohanlal'
}

if st.button("Predict"):
    prediction = model.predict(process_ip('./image.jpg'))
    print(prediction)
    index = np.argmax(prediction)
    st.title("the person is "+mm_dict[index])