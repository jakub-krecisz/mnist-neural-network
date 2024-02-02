import numpy as np

def one_hot(labels: np.ndarray) -> np.ndarray:
    return np.eye(10)[labels].T

def get_predictions(output):
    return np.argmax(output, 0)

def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size

def calculate_error(output, labels):
    epsilon = 1e-15

    error = -1 / labels.size * np.sum(one_hot(labels) * np.log(output + epsilon))
    return error
