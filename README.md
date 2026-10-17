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


