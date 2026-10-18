# Task-4-UGV---5-object-Recognition-ML-Model
An ML model that can identify 5 different types of objects of your choice from images. 

##Member-1 Soumye Chauhan 
# Dataset Preparation and Image Preprocessing

## Overview
This task focuses on preparing an image dataset for a five-class object recognition ML project by organizing raw images, splitting the dataset, and preprocessing images.

## Object Classes
- Apple
- Glass Container
- Oranges
- Popcorn
- Water Bottle

## Project Files
- `raw_images/` – Stores original images organized by class.
- `dataset/` – Contains the organized dataset.
- `setup_dataset.py` – Sets up the dataset directory structure and organizes images.
- `split_dataset.py` – Splits images into training, validation, and testing sets.
- `preprocess_image.py` – Resizes images, converts them into RGB format and NumPy arrays, and normalizes pixel values for consistent ML input.

## Dataset Split Proportions
- **Training:** 70%
- **Validation:** 15%
- **Testing:** 15%

## Purpose of Image Preprocessing
Image preprocessing standardizes image size, format, and pixel values so that images can be efficiently processed by a machine learning model. It also helps ensure consistent input data and prepares class labels for model training.

## Execution
Run the scripts in order:

```bash
python setup_dataset.py
python split_dataset.py
python preprocess_image.py
```

## Outcome
A structured, split, and preprocessed image dataset ready for machine learning model development.

##Preshtha - Training Code 
I chose a CNN (Convolutional Neural Network) because it is suitable for image classification and can learn features like shapes, colours, and patterns. The model has an input layer, two convolutional layers with ReLU activation, max pooling layers, a flatten layer, and dense layers, with Softmax in the final output layer for classifying five objects. I compiled the model using the Adam optimizer, Sparse Categorical Crossentropy loss function, and accuracy as the evaluation metric. Initially, I used SGD, but the accuracy was low, so I changed it to Adam. The validation accuracy improved in some runs, although the model still showed signs of overfitting.

##Himanshu - Testing and Evaluation
I tested the model and evaluate it, i calculate accuracy, precision, recall, F1 score, generated a confusion matrix to analyze incorrect prediction,tested on individual images, visualize class wise performance, at first accuracy was low so we increased the dataset and re-trained the model , after retraining model has 100% accuracy in training, 85% in testing, and 75% in validation. so it still has some overfitting





