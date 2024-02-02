import numpy as np

def one_hot(labels: np.ndarray) -> np.ndarray:
    """
    Converts integer labels to one-hot encoding.

    :param labels: Array of integer labels.
    :return: One-hot encoded array.
    """
    return np.eye(10)[labels].T

def get_predictions(output: np.ndarray) -> np.ndarray:
    """
    Gets predictions by finding the index of the maximum value in each column.

    :param output: Array of model output.
    :return: Array of predictions.
    """
    return np.argmax(output, axis=0)

def get_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the accuracy of predictions compared to true labels.

    :param predictions: Array of predicted labels.
    :param labels: Array of true labels.
    :return: Accuracy value.
    """
    return np.sum(predictions == labels) / labels.size

def calculate_error(output: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the cross-entropy error between the model output and true labels.

    :param output: Array of model output probabilities.
    :param labels: Array of true labels.
    :return: Cross-entropy error value.
    """
    epsilon = 1e-15
    error = -1 / labels.size * np.sum(one_hot(labels) * np.log(output + epsilon))
    return error
