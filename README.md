# Medical Image Classification Project: Chest X-ray Abnormality Detection

This project guides you through building, training, and deploying a convolutional neural network (CNN) to classify chest X-ray images as normal or abnormal (e.g., pneumonia detection) using Python, TensorFlow, and OpenCV on your MacBook Pro. It’s designed to teach deep learning, Python frameworks, computer vision, and local development, leveraging your medical imaging background.

## Prerequisites
- **Hardware**: MacBook Pro with at least 8GB RAM (16GB+ recommended) and 20GB free storage.
- **Software**: Python 3.8+, TensorFlow 2.x, OpenCV, NumPy, Pandas, Flask (optional for web app).
- **Dataset**: Chest X-ray Pneumonia Dataset from Kaggle (~5,800 images, normal vs. pneumonia).

## Step 1: Setup Local Environment and Repository (4-6 hours)
**Objective**: Configure a local development environment and clone the project repository from GitHub on your MacBook Pro.
- **Tasks**:
  1. Create a project folder: `mkdir -p ~/Projects/chestx && cd ~/Projects/chestx`.
  2. Generate SSH keys (optional, for GitHub): `ssh-keygen -t ed25519 -C "your_email@example.com"`, press Enter to accept default file location and optionally set a passphrase.
  3. Add the public key to GitHub:
     - Log in to GitHub, navigate to Settings > SSH and GPG keys > New SSH key or Add SSH key.
     - Paste the public key from `~/.ssh/id_ed25519.pub` and save it.
  4. Clone the repository: `git clone git@github.com:HCVV3arpCn/chestx.git`.
  5. Install Homebrew (if not installed): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`.
  6. Install dependencies: `python3 -m venv env && source env/bin/activate`.
  7. Install libraries: `pip install tensorflow opencv-python numpy pandas flask`.
- **Learning Outcome**: Familiarity with local macOS setup, SSH key generation, GitHub integration, and Python environment management.

## Step 2: Prepare the Dataset (4-6 hours)
**Objective**: Set up the Chest X-ray Pneumonia Dataset locally on your MacBook Pro.
- **Tasks**:
  1. Download the dataset from Kaggle (use the API or manual download).
  2. Move the dataset to `~/Projects/chestx/dataset` (create the folder if needed).
- **Learning Outcome**: Proficiency in dataset management on a local machine.

## Step 3: Data Preprocessing (10-12 hours)
**Objective**: Prepare X-ray images for model training using computer vision techniques.
- **Tasks**:
  1. Write a Python script to load and preprocess images using OpenCV:
     - Resize images to 224x224 pixels.
     - Normalize pixel values (0-1 scale).
     - Convert to grayscale or RGB as needed.
  2. Split dataset into train (70%), validation (20%), and test (10%) sets.
  3. Use `tensorflow.keras.preprocessing.image.ImageDataGenerator` for data augmentation (e.g., rotation, zoom) to improve model robustness.
  4. Save preprocessed data to disk for efficient access.
- **Learning Outcome**: Master image preprocessing with OpenCV and TensorFlow, key for computer vision tasks.

```python

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load and preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize
    return img

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

# Example usage
dataset_path = "dataset"
preprocessed_data = []
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):
        img = preprocess_image(os.path.join(dataset_path, filename))
        preprocessed_data.append(img)
preprocessed_data = np.array(preprocessed_data)

```

## Step 4: Build and Train CNN Model (15-20 hours)
**Objective**: Develop a CNN to classify X-ray images, learning deep learning and TensorFlow on your MacBook Pro.
- **Tasks**:
  1. Design a simple CNN architecture (e.g., 3 convolutional layers, 2 dense layers) using TensorFlow:
     - Conv2D layers with ReLU activation.
     - MaxPooling layers for dimensionality reduction.
     - Dense layers for classification (binary: normal vs. abnormal).
  2. Compile the model with `binary_crossentropy` loss and `adam` optimizer.
  3. Train on the preprocessed dataset (10-20 epochs, batch size 32) locally. Adjust batch size or epochs if memory is limited.
  4. Evaluate model accuracy on validation and test sets (aim for >80% accuracy).
  5. Save the trained model as an `.h5` file.
- **Learning Outcome**: Understand deep learning architectures, model training, and evaluation on local hardware.

```python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Build CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train (example with dummy data)
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator()
train_data = np.random.rand(100, 224, 224, 1)  # Replace with real data
train_labels = np.random.randint(0, 2, (100, 1))
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2)

model.save('chestx_model.h5')

```

## Step 5: Implement Transfer Learning (10-12 hours)
**Objective**: Fine-tune a pre-trained model to enhance accuracy and learn transfer learning.
- **Tasks**:
  1. Load a pre-trained model (e.g., ResNet50) from `tensorflow.keras.applications`.
  2. Freeze base layers and add custom layers for binary classification.
  3. Fine-tune on the X-ray dataset (5-10 epochs).
  4. Compare performance with your custom CNN.
  5. Save the fine-tuned model.
- **Learning Outcome**: Master transfer learning, a key technique for efficient AI development.

```python

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

# Load pre-trained ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train (example with dummy data)
import numpy as np
train_data = np.random.rand(100, 224, 224, 3)  # Replace with real data
train_labels = np.random.randint(0, 2, (100, 1))
model.fit(train_data, train_labels, epochs=5, batch_size=32, validation_split=0.2)

model.save('chestx_transfer_model.h5')

```

## Step 6: Deploy Model Locally (10-15 hours)
**Objective**: Create a Flask web app to serve the model locally on your MacBook Pro.
- **Tasks**:
  1. Write a Flask app to load the trained model and accept image uploads for inference.
  2. Run the app locally on `localhost:5000`.
  3. Test by uploading an X-ray image and verifying the classification output.
  4. (Optional) Share the app via ngrok for remote access (requires ngrok installation and token).
- **Learning Outcome**: Gain local deployment skills with Flask and optional remote sharing.

```python

from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('chestx_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    pred = model.predict(img)
    return jsonify({'result': 'Abnormal' if pred[0] > 0.5 else 'Normal'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

```

## Step 7: Document and Showcase (5-7 hours)
**Objective**: Create a portfolio piece to boost freelance marketability.
- **Tasks**:
  1. Write a README for your GitHub repository, detailing the project, dataset, model architecture, and deployment steps.
  2. Include performance metrics (e.g., accuracy, confusion matrix).
  3. Add screenshots of the local web app and model results.
  4. Update your Upwork/Freelancer.com profiles with a link to the GitHub repo.
- **Learning Outcome**: Learn to present technical work professionally for clients.

## Resources
- **Dataset**: [Kaggle Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Tutorials**:
  - “TensorFlow for Deep Learning” (freeCodeCamp, YouTube, free)
  - “Flask Web Development” (Real Python, free articles)
- **Tools**: Kaggle for datasets, GitHub for version control, Homebrew for macOS dependencies.

## Expected Outcomes
- **Skills Mastered**: Deep learning (CNNs), Python (TensorFlow, OpenCV), computer vision, local development, transfer learning.
- **Portfolio Piece**: A GitHub repo and local web app showcasing a medical AI solution.
- **Market Impact**: Enhances bids for healthcare AI jobs ($30-$100/hour) on Upwork/Freelancer.com, exceeding gig work’s $10-$20/hour net.

---

### Notes
- **Performance**: If training is slow, consider using a smaller dataset subset or Google Colab for heavy computation, then transfer the model back to your MacBook.
- **Storage**: Ensure 20GB+ free space for the dataset and libraries.
- **Dependencies**: If OpenCV installation fails, install dependencies with `brew install libjpeg libpng` before `pip install opencv-python`.

This Markdown eliminates EC2-specific tasks, adapts the workflow to macOS, and includes updated code artifacts for local execution. Let me know if you encounter setup issues or need further adjustments!