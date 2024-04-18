# Task 4 C -> MNIST Network Optimization Analysis Script
#
# Author: Manushi
# Date: 22 March 2024
# File: analyze_experiments_results.py
#
# This script is designed to analyze and report the results of various experiments conducted on the MNIST dataset
# with a convolutional neural network (CNN). The primary goal of these experiments is to evaluate the effects of
# changing different aspects of the network architecture, including the number of filters, dropout rates, number of
# hidden nodes, learning rate, batch size, and epochs on the network's performance. By plotting the accuracy of the
# model against these parameters, we aim to identify optimal configurations that enhance model accuracy and training
# efficiency. This analysis contributes to a deeper understanding of how specific network adjustments can impact the
# overall effectiveness of a model for digit recognition tasks. Through systematic experimentation and analysis,
# this script supports the pursuit of achieving the highest possible accuracy by fine-tuning the network's parameters.

import json
import pandas as pd
import matplotlib.pyplot as plt


def load_results():
    """
    Loads the experiment results from a JSON file.

    Returns:
        A dictionary containing the loaded experiment results.
    """
    with open('experiment_results.json', 'r') as file:
        results = json.load(file)
    return results


def convert_to_dataframe(results):
    """
    Converts the list of dictionaries into a pandas DataFrame for easier analysis.

    Args:
        results (dict): The experiment results loaded from the JSON file.

    Returns:
        A pandas DataFrame containing the experiment results.
    """
    return pd.DataFrame(results)


def plot_accuracy_vs_parameter(df, parameter, comparison_parameter='n_filters'):
    """
    Plots the accuracy of the model against a specified parameter for different values of a comparison parameter.

    Args:
        df (DataFrame): The dataframe containing the results.
        parameter (str): The parameter to plot against accuracy.
        comparison_parameter (str): The parameter used for comparison on the X-axis. Default is 'n_filters'.
    """
    plt.figure(figsize=(10, 6))
    for value in df[parameter].unique():
        subset = df[df[parameter] == value]
        plt.plot(subset[comparison_parameter], subset['accuracy'], marker='o', linestyle='-', label=f'{parameter.capitalize()}={value}')

    plt.xlabel(comparison_parameter.capitalize())
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy vs. {parameter.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    Main function to execute the analysis script.

    This function orchestrates the loading of experiment results, conversion to a pandas DataFrame, and plotting
    accuracy against various parameters to analyze the impact of different network configurations on model performance.
    """
    results = load_results()
    df = convert_to_dataframe(results)

    # List of parameters to plot against accuracy
    parameters_to_plot = ['dropout_rate', 'n_filters', 'n_hidden_nodes', 'learning_rate', 'epochs', 'batch_size']

    # Automatically generate plots for each parameter
    for parameter in parameters_to_plot:
        # Skip 'n_filters' for comparison_parameter since it's already used as a primary parameter
        comparison_parameter = 'epochs' if parameter == 'n_filters' else 'n_filters'
        plot_accuracy_vs_parameter(df, parameter, comparison_parameter)


if __name__ == "__main__":
    main()
