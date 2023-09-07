from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import streamlit as st

# import keras
# import tensorflow.keras
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("model.h5", compile=False)
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

def ping():
    return "Hello, I'm alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def predict():
    upload = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
    
    if upload is not None:
         im = Image.open(upload)
         img = np.asarray(im)
         img_batch = np.expand_dims(img, 0)
        #image = st.image(uploaded_file)
    
    # if image is not None:
        #image = read_file_as_image(image)
        #img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)
        #predictions = MODEL.predict(image)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {"class": predicted_class, "confidence": float(confidence)}
    else:
        return {"class": "No Image", "confidence": 1}
    

value = ping()
st.write('hello world')
st.write(value)
predict_output = predict()
print(predict_output['class'])

