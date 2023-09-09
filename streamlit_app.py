import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import streamlit as st
import tensorflow as tf

# Sequential model (Conv2D + MaxPool)
MODEL1 = tf.keras.models.load_model("Omdena_model1.h5", compile=False)
# Mobilenet-v2 
MODEL4 = tf.keras.models.load_model("Omdena_model4.h5", compile=False)

CLASS_NAMES = ['Cescospora', 'Healthy', 'Miner', 'Phoma', 'Rust']

def predict():
    upload = st.file_uploader("Upload your image here...", type=['png', 'jpeg', 'jpg'])
    
    if upload is not None:
        image = Image.open(upload)
        image = image.crop((left, top, right, bottom))
        newsize = (256, 256)
        image = image.resize(newsize)
        image = np.asarray(image)
        img_batch = np.expand_dims(image, 0)
        
        predictions1 = MODEL1.predict(img_batch)
        predictions4 = MODEL4.predict(img_batch)

        predicted_class1 = CLASS_NAMES[np.argmax(predictions1[0])]
        confidence1 = np.max(predictions1[0])

        predicted_class4 = CLASS_NAMES[np.argmax(predictions4[0])]
        confidence4 = np.max(predictions4[0])
        
        return {"class1": predicted_class1, "confidence1": float(confidence1), "class4": predicted_class4, "confidence4": float(confidence4)}
    else:
        return {"class1": "No Image", "confidence1": 0, "class4": "No Image", "confidence4": "No Image"}

predicted_output = predict()
st.write("Predicion from baseline CNN model (183877 parameters): ", predicted_output['class1'])
st.write("Predicion from Mobilenet-v2 (2667589 parameters): ", predicted_output['class4'])


