# DEEPLEARNING PROJECT 
COMPANY : CODTECH IT SOLUTIONS NAME : PAIDIPELLI BHEEMRAJ INTERN ID : CT06DF372 DOMAIN : DATA SCIENCE DURATION : 6 WEEKS MENTOR : NEELA SANTHOSH KUMAR
# Problem statement
Implement a deep learning model for image classification or natural language processing using TensorFlow or PyTorch.
# Table of Contents 
1) Project overview 
2) Dataset
3) Model Architecture
4) Project Structure
5) Output
6) Requirements
7) Note
8) Contact

# Deep Learning CNN - CIFAR-10 Classification (PyTorch)
This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset.

- **Dataset**: CIFAR-10 (60,000 32x32 color images in 10 classes)
- **Framework**: PyTorch
- **Model**: Convolutional Neural Network (CNN)
- **Goal**: Achieve high accuracy in classifying CIFAR-10 images

##  Dataset

- **Name**: CIFAR-10
- **Description**: A labeled subset of tiny images (32x32) in 10 categories such as airplane, car, cat, dog, ship, etc.
- **Source**: Automatically downloaded via `torchvision.datasets.CIFAR10`.

##  Model Architecture

The CNN architecture used in this project consists of:

- 3 Convolutional layers with ReLU activation
- MaxPooling layers after the first two conv layers
- Flatten layer
- Fully connected (Linear) layer
- Output layer with 10 class scores
## Project Structure

Deeplearning_cnn/

├── cnn_pytorch.py         # Main PyTorch training and evaluation script

├── README.md              # Project documentation

├── requirements.txt       # Required Python packages

└── .gitignore 

## Output
Training and validation accuracy/loss printed per epoch

Final test accuracy on CIFAR-10

Accuracy plot using matplotlib

## Requirements
Python 3.7+

PyTorch

torchvision

matplotlib
## Note
The CIFAR-10 dataset will be downloaded automatically when running the script.

The data/ folder containing datasets is excluded from Git using .gitignore to avoid pushing large files.

## Contact
Feel free to reach out or raise issues if you encounter any problems.

GitHub: Bheemrajpaidipelli


