# Extension B -> Pre-trained Network Analysis Script
#
# Author: Manushi
# Date: 22 March 2024
# File: pretrained_network_analysis.py
#
# This script focuses on analyzing a pre-trained convolutional neural network's behavior, specifically examining
# the first convolutional layer's filters and their effects on input images. It demonstrates how to load a
# pre-trained network, extract and visualize the weights of its first convolutional layer, and apply these filters
# to an input image to observe the resulting feature maps. This analysis provides insights into what features
# the network learns to detect in its early stages.

# Import statements
import torch
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2


def load_pretrained_model():
    """
    Loads a pretrained network model from torchvision.models.

    Returns:
        torch.nn.Module: The loaded pre-trained model set to evaluation mode.
    """
    model = models.resnet18(pretrained=True)
    model.eval()
    return model


def visualize_filters(layer):
    """
    Visualizes the filters of a convolutional layer in the network.

    Args:
        layer (torch.nn.Module): The convolutional layer from the pre-trained network.

    Displays a plot of the filters using matplotlib.
    """
    filters = layer.weight.data
    n_filters = filters.shape[0]

    # Normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    # Create a grid for filter visualization
    fig, axs = plt.subplots(int(np.ceil(n_filters / 8)), 8, figsize=(20, 10))

    for i in range(n_filters):
        ax = axs[i // 8, i % 8]
        filter_img = filters[i].cpu().numpy()
        ax.imshow(filter_img.transpose((1, 2, 0)))
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def apply_filters_and_visualize(image_path, layer):
    """
    Applies the filters of the specified convolutional layer to an image and visualizes the activation maps.

    Args:
        image_path (str): Path to the image file.
        layer (torch.nn.Module): The convolutional layer from the pre-trained network.

    Displays a plot of the activation maps resulting from each filter.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Resize to match input size of model
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)

    with torch.no_grad():
        activations = layer(img_tensor)

    # Visualization
    n_filters = activations.shape[1]
    fig, axs = plt.subplots(int(np.ceil(n_filters / 8)), 8, figsize=(20, 10))

    for i in range(n_filters):
        ax = axs[i // 8, i % 8]
        activation_img = activations[0, i].cpu().numpy()
        ax.imshow(activation_img, cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to execute pre-trained network analysis.
    """
    model = load_pretrained_model()
    print(model)

    # First convolutional layer of ResNet18
    first_conv_layer = model.conv1

    # Visualize the first layer filters
    visualize_filters(first_conv_layer)

    # Example image path
    image_path = 'pineapple.png'

    # Apply filters and visualize on an example image
    apply_filters_and_visualize(image_path, first_conv_layer)


if __name__ == '__main__':
    main()
