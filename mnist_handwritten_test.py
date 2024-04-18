# Task 1 F -> MNIST Handwritten Digit Test Script
#
# Author: Manushi
# Date: 22 March 2024
# File: mnist_handwritten_test.py
#
# This script tests a pretrained convolutional neural network on images of handwritten digits. It is designed to
# process individual images of digits written by the user, preprocess these images to match the MNIST dataset
# format, and then use the trained model to predict the digit in each image. The script showcases the model's
# ability to generalize from the MNIST dataset to real-world handwritten digits. It demonstrates image preprocessing,
# tensor transformation, and model evaluation on new data not seen during the training phase.

# import statements
import torch
from torchvision import transforms
from PIL import Image
from mnist_digit_recognition_training import MyNetwork
from mnist_digit_evaluation import load_model

# Define a transform to convert the image to tensor and normalize it
# as per MNIST data distribution
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def preprocess_image(image_path):
    """
    Opens, converts to grayscale, and applies transformations to an image from a given path to prepare it for model prediction.

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The preprocessed image as a tensor, ready for model input.
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = transform(img)  # Apply the transformations
    return img


def predict_digit(model, image_tensor):
    """
    Predicts the digit represented by a given image tensor using the trained model.

    Args:
        model (MyNetwork): The trained neural network model for digit recognition.
        image_tensor (torch.Tensor): The preprocessed image tensor.

    Returns:
        int: The predicted digit.
    """
    image_tensor = image_tensor.unsqueeze(0)  # Add an extra batch dimension
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()


def main():
    """
    Executes the process for predicting handwritten digits from images using a trained model.
    """
    model_save_path = 'mnist_model.pth'
    model = MyNetwork()
    model = load_model(model_save_path, model)

    # Test the network on new inputs
    for i in range(10):
        image_path = f'handwritten_digits/digit_{i}.jpg'
        image_tensor = preprocess_image(image_path)
        prediction = predict_digit(model, image_tensor)
        print(f'Handwritten digit image: {i}, Model Prediction: {prediction}')


if __name__ == "__main__":
    main()
