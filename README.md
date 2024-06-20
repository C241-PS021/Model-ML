# Fruit and Vegetable Classifier

This repository contains the implementation of a machine learning model for classifying fresh and rotten fruits and vegetables. The model is built using TensorFlow, and it is converted to TensorFlow Lite for deployment on mobile devices.

## Repository Structure
- `fruit_vegetable_classifier.tflite`: The trained TensorFlow Lite model.
- `model_cnn.ipynb`: Jupyter notebook containing the code for training the model.

## Prerequisites

- Python 3.6 or higher
- Git
- Google Collab
- Tensorflow 2.x

## Model Description

The model is a Convolutional Neural Network (CNN) designed to classify images of fresh and rotten fruits and vegetables. The final model is saved in TensorFlow Lite format for efficient deployment on mobile devices.

### Model Architecture

- **Convolutional Layers**: Used to extract features from the input images.
- **MaxPooling Layers**: Reduce the dimensionality of the feature maps.
- **Flatten Layer**: Converts the 2D feature maps to 1D feature vectors.
- **Dense Layers**: Fully connected layers for classification.
- **Dropout Layer**: Helps to prevent overfitting.
- **Output Layer**: Softmax activation function for multi-class classification.

### Training

The model is trained on a dataset of images that are split into training and testing sets (you can check at [Fruit-Vegetables-Dataset](https://github.com/C241-PS021/Fruits-Vegetables-Dataset)). Data augmentation is applied to the training images to improve the model's generalization.

### TensorFlow Lite Model

The trained model is converted to TensorFlow Lite format (`fruit_vegetable_classifier.tflite`) for deployment on mobile devices. The TensorFlow Lite model is optimized for performance and can be used with TensorFlow Lite Interpreter.

## **Installation** 

### Step-by-Step Instructions

1. **Clone the Model and Dataset Repository**

```shell
git clone https://github.com/C241-PS021/Model-ML.git
git clone https://github.com/C241-PS021/Fruits-Vegetables-Dataset.git
```

2. **Create a Virtual Environment (Optional but Recommended)**

```shell
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Run the Notebook**
   
   After you clone, you can open and run the modelcnn.ipynb

Make sure that Python and all the required libraries are installed
