# Participation Assignment

This project demonstrates the development of a neural network using TensorFlow and Keras for image classification. It is part of an assignment by **John Akech** and **Kuir Juach Kuir**.

## Overview

The goal of this project is to build, train, and evaluate a neural network model capable of classifying images from the dataset. Additionally, the project includes features such as data preprocessing, visualization, and performance evaluation using metrics and a confusion matrix.

## Key Features

- **Dataset Handling**: Loads and preprocesses image and label data.
- **Data Visualization**: Displays sample images and their corresponding labels.
- **Model Creation**: Implements a feedforward neural network with:
  - Input layer
  - Two hidden layers
  - Output layer for 10-class classification
- **Training**: Trains the model over 10 epochs, evaluating its performance on validation data.
- **Visualization**: Plots training history (accuracy and loss).
- **Model Saving and Loading**: Saves the trained model for future use.
- **Predictions**: Makes predictions on unseen data and visualizes results.
- **Confusion Matrix**: Evaluates classification performance with a confusion matrix.

## Tools and Libraries Used

- **TensorFlow & Keras**: For model creation and training.
- **Matplotlib**: For data visualization.
- **NumPy**: For numerical operations.
- **Pandas**: For structured data handling.
- **Scikit-learn**: For generating the confusion matrix.

## How to Run the Project

1. **Open in Google Colab**: Click the "Open in Colab" badge (if added) to directly interact with the notebook.
2. **Install Dependencies**:
   Ensure you have the following Python libraries installed:

   pip install tensorflow matplotlib numpy pandas scikit-learn

1. **Download Dataset**: Place the dataset files (t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte) in the appropriate directory or Colab environment.
   
2. **Execute the Notebook**: Run the notebook cells step-by-step to load the dataset, preprocess data, train the model, and evaluate performance.
   
## Results

**. Training Accuracy**: Achieved a high accuracy of ~99.9% during training.

**. Validation Accuracy**: Maintained over 95% accuracy on the validation dataset.

**. Confusion Matrix**: Highlights the model's performance across different classes.

## Visuals

**. Sample Images**: Showcased input images and their true labels.

**. Training History**: Accuracy and loss graphs demonstrate model improvement over epochs.

**. Prediction Visualization**: Compares predicted labels with true labels for test samples.

## Model Evaluation

The model was evaluated on test data, achieving impressive results in accuracy and efficiency. Future improvements could focus on optimizing the model for better generalization.

## Contributors

**1. John Akech**

**2 .Kuir Juach Kuir**

Feel free to explore and extend this project. Feedback is welcome!
