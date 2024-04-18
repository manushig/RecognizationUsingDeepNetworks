# Task 3 -> Greek Letter Recognition Script
#
# Author: Manushi
# Date: 22 March 2024
# File: greek_letter_recognition.py
#
# This script extends the application of a pre-trained convolutional neural network (CNN), originally trained on the
# MNIST dataset for handwritten digit recognition, to classify images of Greek letters. It demonstrates the process of
# transfer learning by adjusting the network to recognize three Greek letters: alpha, beta, and gamma. The script
# encompasses loading and transforming additional Greek letter images, freezing the pre-trained model's layers,
# replacing the final layer to suit the new classification task, and evaluating the model's performance on both,
# the original MNIST test set and the newly introduced Greek letter images. This approach showcases the versatility
# and efficiency of transfer learning in adapting existing models to new classification tasks with minimal retraining.


import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from mnist_digit_recognition_training import MyNetwork  # Make sure this import works as expected
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import matplotlib.pyplot as plt
import os

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class GreekTransform:
    """
    Transforms Greek letter images to match the MNIST dataset format.

    This transformation includes converting RGB images to grayscale, resizing to 128x128 pixels,
    applying affine transformations for scaling to 28x28 pixels, center cropping, and finally inverting
    the image colors to match the MNIST digits' background and foreground colors.
    """

    def __call__(self, x):
        x = TF.rgb_to_grayscale(x)
        x = TF.resize(x, [128, 128])
        x = TF.affine(x, angle=0, translate=(0, 0), scale=36 / 128, shear=0)
        x = TF.center_crop(x, output_size=(28, 28))
        x = TF.invert(x)
        return x


def modify_network_for_greek_letters(model):
    """
    Modifies a pre-trained MNIST neural network model to classify Greek letters.

    Args:
        model (nn.Module): A pre-trained model initially designed for MNIST digit classification.

    This function freezes all the parameters of the model to prevent re-training of the existing layers.
    It then replaces the final layer of the network with a new Linear layer tailored for classifying
    five different classes (Greek letters: alpha, beta, gamma, delta, and epsilon).
    """
    for param in model.parameters():
        param.requires_grad = False  # Freeze all parameters
    num_features = model.fc2.in_features
    model.fc2 = nn.Linear(num_features, 5)  # Adjust for 5 classes
    return model


def split_dataset(dataset, split_ratio=0.8):
    """
    Splits a dataset into training and validation sets based on a specified ratio.

    Args:
        dataset (Dataset): The dataset to be split.
        split_ratio (float): The ratio of the dataset to be used as the training set.

    Returns:
        Subset: The training dataset subset.
        Subset: The validation dataset subset.
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_ratio * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def train_model(model, train_loader, val_loader, epochs=20):
    """
    Trains and validates a model on the Greek letter dataset with early stopping.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        epochs (int): Maximum number of epochs to train the model.

    The function trains the model using the training dataset and evaluates its performance on the validation dataset.
    It implements early stopping based on validation loss to prevent overfitting. Training and validation losses are
    recorded and plotted at the end of training to visualize the learning process.
    """
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    best_val_accuracy = 0
    early_stopping_patience = 5
    early_stopping_counter = 0

    train_losses = []  # Store training losses
    val_losses = []  # Store validation losses

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)  # Update to accumulate batch loss
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss / train_total)  # Append average loss

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)  # Update to accumulate batch loss
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * val_correct / val_total
        val_loss /= val_total
        val_losses.append(val_loss / val_total)  # Append average loss

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early Stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save best model
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / train_total}, Train Accuracy: {train_accuracy}%, Val Accuracy: {val_accuracy}%')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def main():
    """
        The main execution block of the script for Greek letter recognition using transfer learning.

        Steps:
        1. Specify the path to the training dataset containing images of Greek letters.
        2. Define the transformations to apply to each image, converting them to a format compatible with the MNIST model.
        3. Load the Greek letters dataset and split it into training and validation subsets.
        4. Initialize data loaders for both the training and validation sets with a specified batch size.
        5. Load the pre-trained MNIST model and modify it to classify Greek letters by adjusting the final layer.
        6. Train the modified model on the Greek letters dataset, implementing early stopping based on validation loss.
        7. Evaluate the model on additional Greek letter images not seen during training, displaying predictions and images.

        This block demonstrates the complete process from data loading and preparation through model adaptation, training, 
        and evaluation, culminating in the visual display of classification results on additional data.
    """
    training_set_path = './greek_letters'
    transform = transforms.Compose(
        [GreekTransform(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    greek_dataset = datasets.ImageFolder(root=training_set_path, transform=transform)
    train_dataset, val_dataset = split_dataset(greek_dataset)

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    model = MyNetwork()
    model_path = 'mnist_model.pth'
    model.load_state_dict(torch.load(model_path))
    model = modify_network_for_greek_letters(model)
    print(model)

    train_model(model, train_loader, val_loader, epochs=20)

    # Load and predict additional Greek letter images
    additional_data_path = './additional_greek_letters'
    additional_dataset = datasets.ImageFolder(root=additional_data_path, transform=transform)
    additional_loader = DataLoader(additional_dataset, batch_size=1, shuffle=False)

    model.eval()  # Ensure the model is in evaluation mode
    print("Results on Additional Data:")
    for i, (images, labels) in enumerate(additional_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        idx_to_class = {v: k for k, v in additional_dataset.class_to_idx.items()}
        predicted_label_name = idx_to_class[predicted.item()]

        # Retrieve the path of the current image
        file_path = additional_dataset.samples[i][0]
        file_name = file_path.split('/')[-1]  # Adjust the separator if needed
        file_name1 = os.path.basename(file_path)

        print(f"\nPredicted Greek alphabet for File {file_name} is: {predicted_label_name}")

        plt.imshow(TF.to_pil_image(images[0]))
        plt.title(f"Prediction for File {file_name1} \nis: {predicted_label_name}")
        plt.show()


if __name__ == '__main__':
    main()
