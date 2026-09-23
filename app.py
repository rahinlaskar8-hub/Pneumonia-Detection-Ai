import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model_deployed = load_model('pneumonia_detection_model.keras')

def predict_xray(image):
    if image is None:
        return "Please upload an image."
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized_img = cv2.resize(gray_img, (150, 150))
    normalized_img = resized_img / 255.0
    reshaped_img = normalized_img.reshape(-1, 150, 150, 1)
    
    prediction_prob = model_deployed.predict(reshaped_img)[0][0]
    p_pneumonia = float(1 - prediction_prob)
    p_normal = float(prediction_prob)
    return {"Pneumonia": p_pneumonia, "Normal": p_normal}

interface = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type="numpy", label="Upload Chest X-Ray"),
    outputs=gr.Label(num_top_classes=2, label="Model Prediction"),
    title="Chest X-Ray Pneumonia Detector",
    description="Upload a chest X-Ray image to detect signs of Pneumonia."
)

import os
interface.launch(server_name="0.0.0.0",
                 server_port=int(os.environ.get("PORT",7860)))
