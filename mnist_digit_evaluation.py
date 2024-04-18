# Task 1 E -> MNIST Digit Evaluation Script
#
# Author: Manushi
# Date: 22 March 2024
# File: mnist_digit_evaluation.py
#
# This script is designed for evaluating a pretrained convolutional neural network on the MNIST test dataset. It
# demonstrates the process of loading a saved model, preparing the MNIST test dataset, and running the model to
# predict the digits. The script displays the model's predictions alongside the actual labels for the first ten
# examples in the test set, providing a visual inspection of the model's performance. Additionally, it showcases
# the process of plotting these digits with their predicted labels, offering an intuitive way to assess the model's
# accuracy visually.

# import statements
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from mnist_digit_recognition_training import MyNetwork


def load_model(filepath, model):
    """
    Loads the trained model from a file.

    Args:
        filepath (str): The path to the file where the trained model is saved.
        model (MyNetwork): An instance of the MyNetwork class.

    Returns:
        MyNetwork: The model loaded with the trained weights and set to evaluation mode.
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    return model


def evaluate_model(model, test_data):
    """
    Evaluates the model on the MNIST test dataset and displays predictions for the first 10 examples.

    Args:
        model (MyNetwork): The trained network model to evaluate.
        test_data (Dataset): The MNIST test dataset.
    """
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)
    test_examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(test_examples)

    with torch.no_grad():
        output = model(example_data)

    print("Model predictions and actual labels:")
    for i in range(10):
        print(f'\nExample {i + 1}:')
        print(f'Network output: {output[i].data.numpy()}')
        print(f'Predicted label: {output[i].argmax(dim=0, keepdim=True).item()}, Actual label: {example_targets[i]}')

    # Plot the first 9 digits and their corresponding predictions
    fig = plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Prediction: {output[i].argmax(dim=0, keepdim=True).item()}")
        plt.xticks([])
        plt.yticks([])
    plt.show()


def main():
    """
    Main function to execute the model evaluation.
    """
    # Define the path where the model is saved
    model_save_path = 'mnist_model.pth'

    # Load MNIST test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Initialize the network
    my_network = MyNetwork()

    # Load the trained model
    my_network = load_model(model_save_path, my_network)

    # Evaluate the model
    evaluate_model(my_network, test_data)


if __name__ == "__main__":
    main()
