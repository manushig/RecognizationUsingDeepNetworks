# Task 1 A - D -> Digit Recognition Training Script
#
# Author: Manushi
# Date: 22 March 2024
# File: mnist_digit_recognition_training.py
#
# This script demonstrates the process of building and training a convolutional neural network (CNN) for recognizing
# handwritten digits using the MNIST dataset. It includes steps for loading and visualizing the dataset, defining a
# CNN architecture, training the model on the MNIST training dataset, evaluating its performance on a test dataset,
# and finally, saving the trained model for future use. The network architecture employs convolutional layers,
# dropout, and fully connected layers to learn to classify digits from 0 to 9 accurately.


# import statements
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader


class MyNetwork(nn.Module):
    """
        Defines the neural network for MNIST digit recognition.

        Inherits from nn.Module and includes convolutional, dropout, and linear layers for classification.
    """
    def __init__(self):
        """
        Initializes the network layers.
        """
        super(MyNetwork, self).__init__()
        # First convolution layer with 10 filters of size 5x5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Second convolution layer with 20 filters of size 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout layer with a dropout rate of 50%
        self.conv2_drop = nn.Dropout2d(p=0.5)
        # First fully connected layer
        self.fc1 = nn.Linear(320, 50)
        # Second fully connected layer
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (Tensor): Input tensor containing the batch of images.

        Returns:
            Tensor: The network's output logits.
        """
        # Apply first convolution layer followed by max pooling and ReLU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Apply second convolution layer followed by dropout, max pooling, and ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten the output for the fully connected layer
        x = x.view(-1, 320)
        # Apply first fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        # Apply second fully connected layer
        x = self.fc2(x)
        # Apply log_softmax to the output layer
        return F.log_softmax(x, dim=1)


def display_mnist_samples(test_data):
    """
    Displays the first six samples from the MNIST test dataset.

    Args:
        test_data (Dataset): The MNIST test dataset.
    """
    images, labels = zip(*[(test_data[i][0], test_data[i][1]) for i in range(6)])
    fig, axes = plt.subplots(1, 6, figsize=(10, 2))
    for img, label, ax in zip(images, labels, axes):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(label)
        ax.axis('off')
    plt.show()


def train_model(model, train_data, test_data, epochs, batch_size):
    """
    Trains the model on the MNIST dataset.

    Args:
        model (MyNetwork): The neural network model to train.
        train_data (Dataset): The training dataset.
        test_data (Dataset): The test dataset.
        epochs (int): The number of epochs to train for.
        batch_size (int): The size of the batch.

    Shows plots of training and testing losses and accuracy.
    """
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    optimizer = Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            total_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        train_accuracy.append(100. * correct / len(train_loader.dataset))

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += loss_function(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracy.append(100. * correct / len(test_loader.dataset))

        print(f'Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'Train Accuracy: {train_accuracy[-1]:.2f}%, Test Accuracy: {test_accuracy[-1]:.2f}%')

    # Plot training and testing errors
    plt.figure(figsize=(10, 8))
    plt.plot(train_losses, color='blue', label='Train loss')
    plt.plot(test_losses, color='red', label='Test loss')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.legend()
    plt.title('Training and Testing Loss over Epochs')
    plt.show()

    # Plot training and testing accuracy
    plt.figure(figsize=(10, 8))
    plt.plot(train_accuracy, color='green', label='Train Accuracy')
    plt.plot(test_accuracy, color='orange', label='Test Accuracy')
    plt.xlabel('number of training examples seen')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Testing Accuracy over Epochs')
    plt.show()


def save_model(model, filepath):
    """
    Saves the model to a specified filepath.

    Args:
        model (MyNetwork): The model to save.
        filepath (str): The file path to save the model to.
    """
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')


def main():
    """
    Main function to execute the training script.
    """
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Display the first six samples
    display_mnist_samples(test_data)

    # Initialize the network and optimizer
    my_network = MyNetwork()

    # Load the training data
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)

    # Define a file path to save the model
    model_save_path = 'mnist_model.pth'

    # Train the model
    train_model(my_network, train_data, test_data, epochs=5, batch_size=64)

    # Save the model
    save_model(my_network, model_save_path)


if __name__ == "__main__":
    main()
