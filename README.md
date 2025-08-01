# Medical Image Classification Project: Chest X-ray Abnormality Detection

This project guides you through building, training, and deploying a convolutional neural network (CNN) to classify chest X-ray images as normal or abnormal (e.g., pneumonia detection) using Python, TensorFlow, and OpenCV, deployed on an AWS EC2 Ubuntu instance. It’s designed to teach deep learning, Python frameworks, computer vision, and cloud computing, leveraging your medical imaging background.

## Prerequisites
- **Hardware**: Local computer for development; AWS EC2 Ubuntu instance (e.g., t2.micro for prototyping, t3.medium for training).
- **Software**: Python 3.8+, TensorFlow 2.x, OpenCV, NumPy, Pandas, Flask, AWS CLI.
- **Dataset**: Chest X-ray Pneumonia Dataset from Kaggle (~5,800 images, normal vs. pneumonia).

Below is the updated Markdown content with the steps reordered and renumbered as requested. The new Step 1 combines the original Step 2 (setting up the code repository and cloning the GitHub repository) with the task of setting up the AWS EC2 system (task 1 from the original Step 1). The new Step 2 includes the remaining tasks from the original Step 1 (tasks 2–6). The artifact contains the full relevant portion with the reordered and renumbered steps.

## Step 1: Setup AWS EC2 and Code Repository (6-8 hours)
**Objective**: Configure an AWS EC2 Ubuntu instance and set up a local repository folder with SSH access to clone the project repository from GitHub.
- **Tasks**:
  1. Launch an EC2 Ubuntu 20.04 instance (t2.micro, free tier eligible).
  2. Create a folder for the code repository: `mkdir Repositories && cd Repositories`.
  3. Generate SSH keys: `ssh-keygen -t ed25519 -C "your_email@example.com"`, press Enter to accept default file location and optionally set a passphrase.
  4. Display the public key: `cat ~/.ssh/id_ed25519.pub`, then copy the output.
  5. Add the public key to GitHub:
     - Log in to GitHub, navigate to Settings > SSH and GPG keys > New SSH key or Add SSH key.
     - Paste the copied public key and save it.
  6. Clone the repository: `git clone git@github.com:HCVV3arpCn/chestx.git`.
- **Learning Outcome**: Familiarity with AWS EC2 instance setup, SSH key generation, GitHub SSH configuration, and cloning repositories for project development.

## Step 2: Configure Development Environment (4-6 hours)
**Objective**: Set up the Python environment and prepare the dataset on the EC2 instance.
- **Tasks**:
  1. Install dependencies: `sudo apt update && sudo apt install python3-pip python3-venv`.
  2. Create a Python virtual environment: `python3 -m venv env && source env/bin/activate`.
  3. Install libraries: `pip install tensorflow opencv-python numpy pandas flask boto3`.
  4. Download the Chest X-ray Pneumonia Dataset from Kaggle (use Kaggle API or manual download).
  5. Upload the dataset to EC2 using `scp` or AWS S3.
- **Learning Outcome**: Proficiency in Python environment management and dataset handling on a cloud instance.

## Step 3: Data Preprocessing (10-12 hours)
**Objective**: Prepare X-ray images for model training using computer vision techniques.
- **Tasks**:
  1. Write a Python script to load and preprocess images using OpenCV:
     - Resize images to 224x224 pixels.
     - Normalize pixel values (0-1 scale).
     - Convert to grayscale or RGB as needed.
  2. Split dataset into train (70%), validation (20%), and test (10%) sets.
  3. Use `tensorflow.keras.preprocessing.image.ImageDataGenerator` for data augmentation (e.g., rotation, zoom) to improve model robustness.
  4. Save preprocessed data to disk or S3 for efficient access.
- **Learning Outcome**: Master image preprocessing with OpenCV and TensorFlow, key for computer vision tasks.

```python
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
```

## Step 4: Build and Train CNN Model (15-20 hours)
**Objective**: Develop a CNN to classify X-ray images, learning deep learning and TensorFlow.
- **Tasks**:
  1. Design a simple CNN architecture (e.g., 3 convolutional layers, 2 dense layers) using TensorFlow:
     - Conv2D layers with ReLU activation.
     - MaxPooling layers for dimensionality reduction.
     - Dense layers for classification (binary: normal vs. abnormal).
  2. Compile the model with `binary_crossentropy` loss and `adam` optimizer.
  3. Train on the preprocessed dataset (10-20 epochs, batch size 32) using EC2’s t3.medium instance for GPU support.
  4. Evaluate model accuracy on validation and test sets (aim for >80% accuracy).
  5. Save the trained model as an `.h5` file.
- **Learning Outcome**: Understand deep learning architectures, model training, and evaluation.

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
from tensorflow.keras.layers import GlobalAveragePooling2D

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
```

## Step 6: Deploy Model on AWS EC2 (15-20 hours)
**Objective**: Create a Flask web app to serve the model, learning cloud deployment.
- **Tasks**:
  1. Write a Flask app to load the trained model and accept image uploads for inference.
  2. Configure EC2 security groups to allow HTTP traffic (port 80).
  3. Install Nginx or Gunicorn on Ubuntu to serve the Flask app.
  4. Deploy the app on EC2, ensuring it’s accessible via a public URL.
  5. Test by uploading an X-ray image and verifying the classification output.
- **Learning Outcome**: Gain cloud deployment skills with AWS EC2 and Flask.

```python
from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    pred = model.predict(img)
    return jsonify({'result': 'Abnormal' if pred[0] > 0.5 else 'Normal'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
```

## Step 7: Document and Showcase (5-7 hours)
**Objective**: Create a portfolio piece to boost freelance marketability.
- **Tasks**:
  1. Write a README for your GitHub repository, detailing the project, dataset, model architecture, and deployment steps.
  2. Include performance metrics (e.g., accuracy, confusion matrix).
  3. Add screenshots of the web app and model results.
  4. Update your Upwork/Freelancer.com profiles with a link to the GitHub repo.
- **Learning Outcome**: Learn to present technical work professionally for clients.

## Resources
- **Dataset**: [Kaggle Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Tutorials**:
  - “TensorFlow for Deep Learning” (freeCodeCamp, YouTube, free)
  - “AWS EC2 Flask Deployment” (Medium articles, free)
- **Tools**: Kaggle for datasets, GitHub for version control, AWS Free Tier for EC2.

## Expected Outcomes
- **Skills Mastered**: Deep learning (CNNs), Python (TensorFlow, OpenCV), computer vision, cloud computing (AWS EC2), transfer learning.
- **Portfolio Piece**: A GitHub repo and deployed web app showcasing a medical AI solution.
- **Market Impact**: Enhances bids for healthcare AI jobs ($30-$100/hour) on Upwork/Freelancer.com, exceeding gig work’s $10-$20/hour net.
