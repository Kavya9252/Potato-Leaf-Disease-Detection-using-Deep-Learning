from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import tensorflow as tf

app = FastAPI()

endpoint = "http://127.0.0.1:8501/v1/models/potatoes_model:predict"

CLASS_NAMES = ["Early Blight","Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return "Hello world"

def read_file_as_image(data)->np.ndarray:
    print(f"Received {len(data)} bytes")
    print(f"First 20 bytes: {data[:20]}")
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
  image = read_file_as_image(await file.read())
  img_batch = np.expand_dims(image,0)
#   prediction = MODEL.predict(img_batch)
  json_data = {
      "instances":img_batch.tolist()
  }
  response = requests.post(endpoint,json=json_data)
  prediction = np.array(response.json()["predictions"][0])
  predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
  confidence = np.max(prediction[0])
  return{
      'class':predicted_class,
      'confidence':float(confidence)
  }


if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost' ,port=8000)
