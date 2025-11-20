"""
Flask web app for chest X-ray pneumonia prediction.

Loads the trained ResNet50 model and serves predictions via drag-and-drop upload.
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("../models/best_resnet_chestx.keras")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess an uploaded image for ResNet50 prediction.

    Args:
        image_bytes: Raw bytes from the uploaded file.

    Returns:
        np.ndarray: Preprocessed image ready for model prediction,
        shape (1, 224, 224, 3), values in [0, 1], RGB order.

    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def index() -> str:
    """Render the main upload page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict() -> tuple[dict, int]:
    """Handle image upload and return pneumonia prediction as JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    image_bytes = file.read()
    img_array = preprocess_image(image_bytes)
    pred = model.predict(img_array, verbose=0)[0][0]
    
    label = "Pneumonia" if pred > 0.5 else "Normal"
    confidence = float(pred if pred > 0.5 else 1 - pred)

    return jsonify({
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    print("Flask app starting – go to http://127.0.0.1:5000")
    app.run(debug=True)