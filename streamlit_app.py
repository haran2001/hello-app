import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import streamlit as st
import tensorflow as tf

MODEL = tf.keras.models.load_model("model.h5", compile=False)
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

def predict():
    upload = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
    
    if upload is not None:
        im = Image.open(upload)
        img = np.asarray(im)
        img_batch = np.expand_dims(img, 0)

        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {"class": predicted_class, "confidence": float(confidence)}
    else:
        return {"class": "No Image", "confidence": 1}
    
st.write(predict_output['class'])

