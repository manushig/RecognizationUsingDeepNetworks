# Task 2 A - B -> Network Analysis Script
#
# Author: Manushi
# Date: 22 March 2024
# File: network_analysis.py
#
# This script focuses on analyzing the convolutional neural network's behavior, specifically examining the first
# convolutional layer's filters and their effects on input images. It demonstrates how to load a trained network,
# extract and visualize the weights of its first convolutional layer, and apply these filters to an input image
# to observe the resulting feature maps. This analysis helps in understanding what features the network learns to
# detect in its early stages and provides insights into the internal workings of CNNs.

# Import statements
import sys
import torch
from torchvision import datasets, transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
from mnist_digit_recognition_training import MyNetwork
from mnist_digit_evaluation import load_model


def load_trained_network(filepath):
    """
    Loads a trained network model from a specified file path.

    Args:
        filepath (str): Path to the file containing the trained model weights.

    Returns:
        MyNetwork: The trained network model loaded with weights and set to evaluation mode.
    """
    model = MyNetwork()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model


def visualize_filters(model):
    """
    Visualizes the filters of the first convolutional layer in the network.

    Args:
        model (MyNetwork): The trained network model.

    Displays a plot of the filters using matplotlib.
    """
    filters = model.conv1.weight.data
    n_filters = filters.shape[0]

    # Normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    # Create a 3x4 grid for filter visualization
    fig, axs = plt.subplots(3, 4, figsize=(8, 6))
    axs = axs.flatten()  # Flatten the array for easy iteration

    for i in range(n_filters):  # Loop over all filters
        ax = axs[i]
        ax.imshow(filters[i, 0].cpu(), cmap='viridis')  # Using viridis colormap
        ax.set_title(f'Filter {i}')
        ax.axis('off')

    # Leave the last two slots of the grid empty
    for i in range(n_filters, len(axs)):
        axs[i].axis('off')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)  # Adjust layout to prevent overlap
    plt.show()


def apply_filters_and_visualize(model, train_data):
    """
    Applies the filters of the first convolutional layer to the first training image and visualizes the activation maps.

    Args:
        model (MyNetwork): The trained network model.
        train_data (Dataset): The dataset containing training images.

    Displays a plot of the activation maps resulting from each filter.
    """
    # Access the first training example
    image, _ = train_data[0]
    image = image.unsqueeze(0)  # Add batch dimension

    # Normalize filter values to 0-1 so we can visualize them
    filters = model.conv1.weight.data
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    with torch.no_grad():
        activations = F.conv2d(image, model.conv1.weight)

    # Visualize the filters and activation maps
    fig, axs = plt.subplots(5, 4, figsize=(10, 12))  # 5 rows and 4 columns grid

    for i in range(10):  # Loop over all 10 filters and their activation maps
        # Visualize the i-th filter weights in the first and third column
        filter_ax = axs[i % 5, (i // 5) * 2]
        filter_ax.imshow(filters[i, 0].cpu(), cmap='gray')
        filter_ax.axis('off')

        # Visualize the i-th activation map in the second and fourth column
        activation_ax = axs[i % 5, (i // 5) * 2 + 1]
        activation_map = activations[0, i].cpu().numpy()
        activation_ax.imshow(activation_map, cmap='gray')
        activation_ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to execute network analysis.
    """
    model_path = 'mnist_model.pth'
    model = load_trained_network(model_path)
    print(model)

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)

    # Visualize filters and activations
    visualize_filters(model)
    apply_filters_and_visualize(model, train_data)


if __name__ == '__main__':
    main()
