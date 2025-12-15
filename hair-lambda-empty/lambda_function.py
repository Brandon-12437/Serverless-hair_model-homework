import json
import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load the ONNX model (already included in the base Docker image)
session = ort.InferenceSession("hair_classifier_empty.onnx")

def lambda_handler(event, context=None):
    url = event.get("url")
    if url is None:
        return {"statusCode": 400, "body": "No URL provided"}

    # Load and preprocess the image
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((200,200))  # Resize
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, 0)

    # Run the model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pred = session.run([output_name], {input_name: img_array})[0]

    return {"statusCode": 200, "body": {"probability": float(pred[0][0])}}

