# Task-4-UGV---5-object-Recognition-ML-Model
An ML model that can identify 5 different types of objects of your choice from images. 



# Soumye Chauhan - Dataset Collection

This task focuses on collecting and preparing a dataset for a five-class object recognition ML project, covering Apple, Glass Container, Oranges, Popcorn, and Water Bottle. Initially, the dataset was small, so it was expanded to 500 images to provide more data for model training and testing. The images were organized by class, split into training (70%), validation (15%), and testing (15%) sets, and preprocessed by resizing, converting to RGB format, and normalizing pixel values. The final dataset was structured and prepared for machine learning model development.

## Preshtha - Training Code 
I chose a CNN (Convolutional Neural Network) because it is suitable for image classification and can learn features like shapes, colours, and patterns. The model has an input layer, two convolutional layers with ReLU activation, max pooling layers, a flatten layer, and dense layers, with Softmax in the final output layer for classifying five objects. I compiled the model using the Adam optimizer, Sparse Categorical Crossentropy loss function, and accuracy as the evaluation metric. Initially, I used SGD, but the accuracy was low, so I changed it to Adam. The validation accuracy improved in some runs, although the model still showed signs of overfitting.


## Himanshu - Testing and Evaluation
I tested the model and evaluate it, i calculate accuracy, precision, recall, F1 score, generated a confusion matrix to analyze incorrect prediction,tested on individual images, visualize class wise performance, at first accuracy was low so we increased the dataset and re-trained the model , after retraining model has 100% accuracy in training, 85% in testing, and 75% in validation. so it still has some overfitting




