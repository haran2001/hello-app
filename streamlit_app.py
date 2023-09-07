from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# import keras
# import tensorflow.keras
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("model.h5", compile=False)
# MODEL = keras.models.load_model("potatoes.h5")
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I'm alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


#@app.post("/predict")
@app.get("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {"class": predicted_class, "confidence": float(confidence)}

# @app.post("/upload")
# @app.get("/predict")
# def upload(file: UploadFile = File(...)):
#     try:
#         contents = file.file.read()
#         with open(file.filename, 'wb') as f:
#             f.write(contents)
#     except Exception:
#         return {"message": "There was an error uploading the file"}
#     finally:
#         file.file.close()

#     return {"message": f"Successfully uploaded {file.filename}"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
