import onnxruntime as ort

session = ort.InferenceSession("hair_classifier_empty.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

from PIL import Image
from io import BytesIO
from urllib import request

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    img = Image.open(BytesIO(buffer)).convert("RGB")
    return img
import numpy as np

def preprocess_image(img):
    img = img.resize((200, 200))
    img_array = np.array(img).astype(np.float32) / 255.0  # scale 0-1

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # HWC → CHW and add batch dimension
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
def lambda_handler(event, context):
    url = event['url']  # URL passed in the event
    img = download_image(url)
    img_array = preprocess_image(img)
    
    pred = session.run([output_name], {input_name: img_array})[0]
    probability = float(pred[0][0])  # convert to Python float
    
    return {
        "statusCode": 200,
        "body": {"probability": probability}
    }
from lambda_function import lambda_handler

event = {
    "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
}

result = lambda_handler(event, None)
print(result)
