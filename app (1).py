import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Image preprocessing function for the Autoencoder model
def preprocess_image1(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(image).astype(np.float32)
    image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
    image = image / 255.0
    image = np.reshape(image, (1, 160, 160, 1))
    return image

# Image preprocessing function for the GANs model
def preprocess_image2(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image).astype(np.float32)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = image / 255.0
    image = np.reshape(image, (1, 256, 256, 3))
    return image

# Loading The Autoencoder model
json_file1 = open('model.json', 'r')
loaded_model_json1 = json_file1.read()
json_file1.close()
model1 = tf.keras.models.model_from_json(loaded_model_json1)
model1.load_weights("model.h5")
print("Loaded model 1 from disk")

# Loading The GANs model
json_file2 = open('model2.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
model2 = tf.keras.models.model_from_json(loaded_model_json2)
model2.load_weights("model2.h5")
print("Loaded model 2 from disk")


model1.compile(loss='binary_crossentropy',
               optimizer='rmsprop', metrics=['accuracy'])


@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('path_to_model')


def main():
    st.title(":blue[A]-:red[E]:orange[Y]:red[E]")
    st.subheader('The point where :blue[Logic] meets :red[Art] ')
    st.caption('Powerd By : Ahmed Hatem, Ziyad Elshazly, Rahma Mahmoud')
    st.subheader('', divider='orange')
    uploaded_file = st.file_uploader(
        "Choose an image that you want to colorize", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        width = st.slider('What is the width in pixels?', 0, 350, 350)
        st.image(image, caption='Uploaded Image', width =width,)

        # Preprocess the image
        image_array = np.array(image)
        preprocessed_image1 = preprocess_image1(image_array)
        preprocessed_image2 = preprocess_image2(image_array)

        # Send the image to the model for processing
        processed_image_output1 = model1.predict(preprocessed_image1)
        processed_image_output2 = (model2(preprocessed_image2,training = True))
        print(type(processed_image_output2))
        processed_image_output2 = np.array(processed_image_output2)
        print (type(processed_image_output2))

        # Display the processed image
        st.image(processed_image_output1[0], caption='Auto Encoder',
                width =width,)
        st.image(processed_image_output2[0], clamp=True,  channels='BGR',caption='GANs',
                width =width,)


if __name__ == '__main__':
    main()
