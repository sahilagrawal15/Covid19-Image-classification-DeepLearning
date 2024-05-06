## Deep Learning - Covid 19 Classification

Welcome to the Deep Learning - Covid 19 Classification README! This project focuses on classifying COVID-19 and normal chest X-ray images using deep learning techniques.

### Table of Contents
1. [Introduction](#introduction)
2. [Data](#data)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Visualization](#visualization)
7. [Contributing](#contributing)
8. [License](#license)

### Introduction
This project utilizes TensorFlow and Keras to build a convolutional neural network (CNN) for classifying chest X-ray images. The model aims to distinguish between COVID-19 and normal cases, providing valuable support in diagnosing the disease.

### Data
The dataset consists of chest X-ray images categorized into 'COVID19' and 'NORMAL' classes. Data augmentation techniques, such as rotation, shifting, and flipping, are applied to increase diversity and reduce overfitting.

### Model Architecture
The CNN model architecture includes convolutional layers for feature extraction, max-pooling layers for downsampling, and dense layers for classification. Dropout is employed to prevent overfitting during training.

### Training
The model is trained using the Adam optimizer with a learning rate of 0.0001 and binary cross-entropy loss. Training is performed on augmented data generated using the ImageDataGenerator.

### Evaluation
The model's performance is evaluated on a separate test set, and metrics such as loss and accuracy are calculated. Predictions are generated for the test set and compared against the true labels.

### Visualization
Visualization techniques, including plotting predictions for a subset of images, are employed to assess the model's performance visually.

### Contributing
Contributions to this project are welcome! Please follow the contribution guidelines outlined in CONTRIBUTING.md.

### License
This project is licensed under the [MIT License](LICENSE).

---

```python
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary output
    ])
