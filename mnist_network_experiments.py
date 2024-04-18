# MNIST Fashion Network Experiment Script
#
# Author: Manushi
# Date: 23 March 2024
# File: mnist_network_experiment.py
#
# This script conducts a series of experiments on the MNIST Fashion dataset using a convolutional neural network (
# CNN) architecture. The purpose is to explore the effects of various hyperparameters and architectural decisions on
# the network's performance and training efficiency. By systematically varying parameters such as the number of
# filters, dropout rate, number of hidden nodes, learning rate, batch size, and training epochs, this script aims to
# identify configurations that optimize accuracy and minimize training time. The experimentation process is automated
# to evaluate a predefined range of network configurations, facilitating a comprehensive analysis of how each aspect
# influences the model's effectiveness in classifying fashion items. The results of these experiments are intended to
# guide the development of more efficient and accurate neural network models for image classification tasks.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
import time


class ExperimentNetwork(nn.Module):
    """
    Defines a customizable convolutional neural network for image classification on the MNIST Fashion dataset.

    This network architecture allows for adjustable parameters such as the number of filters, dropout rate,
    and the number of nodes in the dense layers, catering to experimental flexibility in optimizing the model's performance.
    """
    def __init__(self, n_filters=10, dropout_rate=0.5, n_hidden_nodes=50, num_classes=10):
        """
        Initializes the network layers with customizable parameters.

        Args:
            n_filters (int): Number of filters in the first convolutional layer.
                             This number is doubled in the second convolutional layer.
            dropout_rate (float): The dropout rate applied to prevent overfitting.
            n_hidden_nodes (int): Number of nodes in the hidden dense layer.
            num_classes (int): Number of output classes.
        """
        super(ExperimentNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=5)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        self.fc1 = nn.Linear(n_filters * 2 * 4 * 4, n_hidden_nodes)
        self.fc2 = nn.Linear(n_hidden_nodes, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The network's output log probabilities.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        """
        Calculates the number of features in a batch of tensors after convolution and pooling layers.

        Args:
            x (torch.Tensor): The input tensor after convolution and pooling layers.

        Returns:
            int: The total number of features in the tensor.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def generate_configurations():
    """
    Generates a list of dictionaries, each representing a unique configuration of the network's parameters
    for experimentation.

    Returns:
        List[Dict[str, Union[int, float]]]: A list of configurations with different parameter values.
    """
    configs = []
    for n_filters in [10, 20, 30]:
        for dropout_rate in [0.1, 0.3, 0.5]:
            for n_hidden_nodes in [50, 100, 150]:
                for learning_rate in [0.001, 0.0001]:
                    for epochs in [5, 10, 15]:
                        for batch_size in [64, 128]:
                            configs.append({
                                'n_filters': n_filters,
                                'dropout_rate': dropout_rate,
                                'n_hidden_nodes': n_hidden_nodes,
                                'learning_rate': learning_rate,
                                'epochs': epochs,
                                'batch_size': batch_size
                            })
    return configs


def train_and_evaluate(model, train_loader, test_loader, device, epochs=5, learning_rate=0.01):
    """
    Trains and evaluates the model on the training and test datasets.

    Args:
        model (ExperimentNetwork): The neural network model to be trained and evaluated.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU/GPU) on which to perform training and evaluation.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tuple: Contains model accuracy, training time, and losses for each epoch.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    epoch_losses = []

    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_losses.append(epoch_loss / len(train_loader))

    end_time = time.time()
    training_time = end_time - start_time

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / len(test_loader.dataset)

    return accuracy, training_time, epoch_losses


def main():
    """
    Main function to execute the experiment.

    It loads the dataset, generates configurations, trains and evaluates the model for each configuration,
    and saves the results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.FashionMNIST('./data', download=True, transform=transform)
    configurations = generate_configurations()
    results = []

    for config in configurations:
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [55000, 5000])
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)  # Note: Fixed batch size for testing

        model_params = {k: v for k, v in config.items() if
                        k in ['n_filters', 'dropout_rate', 'n_hidden_nodes', 'num_classes']}
        training_params = {k: v for k, v in config.items() if
                           k in ['epochs', 'learning_rate']}  # Only include relevant parameters

        model = ExperimentNetwork(**model_params)
        accuracy, training_time, epoch_losses = train_and_evaluate(model, train_loader, test_loader, device, **training_params)

        config_result = config.copy()
        config_result.update({
            'accuracy': accuracy,
            'training_time': training_time,
            'epoch_losses': epoch_losses
        })
        results.append(config_result)
        print(f"Config: {config}, Accuracy: {accuracy}%, Training Time: {training_time}s, Epoch Losses: {epoch_losses}")

    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
