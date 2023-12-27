import numpy as np


class ActivationFunction:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative


def sigmoid(x):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """The derivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    """The hyperbolic tangent (tanh) function."""
    return np.tanh(x)


def tanh_derivative(x):
    """The derivative of the hyperbolic tangent (tanh) function."""
    return 1 - np.tanh(x) ** 2


def relu(x):
    """The ReLU function."""
    return np.maximum(0, x)


def relu_derivative(x):
    """The derivative of the ReLU function."""
    return np.where(x > 0, 1, 0)


def linear(x):
    """The linear activation function."""
    return x


def linear_derivative(x):
    """The derivative of the linear activation function."""
    return 1.0


def softmax(x_vec):
    """The Softmax function."""
    e_x = np.exp(x_vec - np.max(x_vec))
    return e_x / e_x.sum(axis=0)

def softmax_derivative(x_vec):
    """The derivative of the Softmax function."""
    s = softmax(x_vec)
    return np.diagflat(s) - np.outer(s, s)


softmax_fun = ActivationFunction(function=softmax, derivative=softmax_derivative)
linear_fun = ActivationFunction(function=linear, derivative=linear_derivative)
sigmoid_function = ActivationFunction(function=sigmoid, derivative=sigmoid_derivative)
tanh_fun = ActivationFunction(function=tanh, derivative=tanh_derivative)
relu_function = ActivationFunction(function=relu, derivative=relu_derivative)
