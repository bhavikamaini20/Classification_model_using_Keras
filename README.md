# Classification_model_using_Keras
# MNIST Digit Classification

This repository contains a Keras-based neural network model for classifying handwritten digits from the MNIST dataset. The project demonstrates the steps to preprocess the data, build and train a neural network, and evaluate its performance.

## Project Structure

- `mnist_classification.ipynb`: Jupyter notebook containing the full code for the project.
- `classification_model.h5`: Pre-trained Keras model saved in HDF5 format.
- `README.md`: Project description and instructions.

## Dataset

The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems. The dataset contains 60,000 training images and 10,000 testing images of digits 0-9.

## Model Architecture

The model is a simple feedforward neural network with the following architecture:
- Input layer: Flatten 28x28 images into a 784-dimensional vector.
- Hidden layer 1: Dense layer with 784 neurons and ReLU activation.
- Hidden layer 2: Dense layer with 100 neurons and ReLU activation.
- Output layer: Dense layer with 10 neurons (one for each class) and softmax activation.

## Training

The model is trained using the Adam optimizer and categorical cross-entropy loss function. Training is performed over 10 epochs with validation on the test dataset.

## Performance

The model achieves an accuracy of approximately 98.1% on the test dataset.

## Usage

### Prerequisites

- Python 3.x
- Keras
- TensorFlow
- NumPy
- Matplotlib

### Running the Code

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/mnist-digit-classification.git
    cd mnist-digit-classification
    ```

2. Install the required libraries:
    ```sh
    pip install keras tensorflow numpy matplotlib
    ```

3. Run the Jupyter notebook to see the code and train the model:
    ```sh
    jupyter notebook mnist_classification.ipynb
    ```

4. To load and use the pre-trained model:
    ```python
    from keras.models import load_model

    pretrained_model = load_model('classification_model.h5')
    ```

### Example

Here's an example of loading the model and making predictions:
```python
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
num_pixels = X_train.shape[1] * X_train.shape[2]
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
X_test = X_test / 255
y_test = to_categorical(y_test)

# Load the pre-trained model
pretrained_model = load_model('classification_model.h5')

# Make predictions
predictions = pretrained_model.predict(X_test)

# Display a sample result
index = 0
print(f'Predicted: {np.argmax(predictions[index])}, Actual: {np.argmax(y_test[index])}')
plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
plt.show()

