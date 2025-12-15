# Use the provided Lambda Python base image with the model
FROM agrigorev/model-2025-hairstyle:v1

# Install required libraries
RUN pip install --no-cache-dir onnxruntime pillow numpy requests

# Copy your lambda function code
COPY lambda_function.py .

# Set the Lambda handler
CMD ["lambda_function.lambda_handler"]
