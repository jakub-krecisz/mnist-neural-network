import numpy as np

def linear_activation(x):
    """The linear activation function."""
    return x

def sigmoid(x):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    """The hyperbolic tangent (tanh) function."""
    return np.tanh(x)

def relu(x):
    """The ReLU function."""
    return np.maximum(0, x)

def softmax(x_vec: np.array) -> np.array:
    """The Softmax function."""
    e_x = np.exp(x_vec - np.max(x_vec))
    return e_x / e_x.sum(axis=0)
