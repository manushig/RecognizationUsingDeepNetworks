# Extension A -> Live Digit Recognition Application
#
# Author: Manushi
# Date: 22 March 2024
# File: live_digit_recognition.py
#
# This script leverages a pre-trained convolutional neural network (CNN) model, specifically trained on the
# MNIST dataset for handwritten digit recognition, to identify and classify digits in a live video stream. Utilizing
# real-time video capture from a webcam, the script dynamically identifies digits presented within a specified
# region of interest (ROI) in the video frame. It preprocesses the captured ROI to conform to the input requirements
# of the CNN, predicts the digit using the model, and displays the predicted digit on the live video feed. This
# implementation demonstrates the practical application of deep learning models in real-world scenarios, providing
# a foundation for further exploration and development of live image and video classification systems.


import cv2
import torch
from torchvision import transforms
import numpy as np
from mnist_digit_recognition_training import MyNetwork


def load_model(model_path='mnist_model.pth'):
    """
    Load the trained model from a specified path.

    Args:
        model_path (str): The path to the trained model file. Defaults to 'mnist_model.pth'.

    Returns:
        nn.Module: The loaded model ready for inference.
    """
    model = MyNetwork()  # Initialize the network
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_image(image):
    """
    Preprocesses the captured image from the video feed for prediction.

    Args:
        image (np.array): The image captured from the video feed.

    Returns:
        torch.Tensor: The preprocessed image tensor ready for model prediction.
        float: The coverage ratio of the digit in the image, to decide on making a prediction.
    """
    _, thresh_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(thresh_image).unsqueeze(0), np.count_nonzero(thresh_image) / (28 * 28)


def predict(model, image, coverage, coverage_threshold=0.05):
    """
    Predicts the digit in the preprocessed image using the trained model, if coverage exceeds a threshold.

    Args:
        model (nn.Module): The loaded digit recognition model.
        image (torch.Tensor): The preprocessed image tensor.
        coverage (float): The ratio of the digit's coverage in the image.
        coverage_threshold (float): The minimum required coverage ratio to make a prediction. Defaults to 0.05.

    Returns:
        str or int: The predicted digit as an integer if coverage is sufficient, otherwise an empty string.
    """
    if coverage < coverage_threshold:
        return " "
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


def main():
    """
    The main function to execute live digit recognition.

    This function initializes the model, sets up webcam access, continuously captures video frames,
    applies preprocessing and prediction on the defined region of interest (ROI) within each frame, and displays the
    predicted digit on the live video feed.
    """
    model = load_model()  # Load the trained model
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Define the ROI (for example, a centered square)
        height, width, _ = frame.shape
        side_length = 200
        top_left = (width // 2 - side_length // 2, height // 2 - side_length // 2)
        bottom_right = (top_left[0] + side_length, top_left[1] + side_length)

        # Extract and process the ROI
        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(gray_roi, (28, 28), interpolation=cv2.INTER_AREA)
        preprocessed_image, coverage = preprocess_image(img_resized)
        digit = predict(model, preprocessed_image, coverage)

        # Display the prediction or "" if not confident
        cv2.putText(frame, f'Predicted Digit: {digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Draw the ROI on the frame
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        cv2.imshow('Digit Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
