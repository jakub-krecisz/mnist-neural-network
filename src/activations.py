import numpy as np


class ActivationFunction:
    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.derivative = derivative

    @classmethod
    def get_function_by_name(cls, function_name):
        functions = {
            'sigmoid': sigmoid_fun,
            'tanh': tanh_fun,
            'relu': relu_fun,
            'linear': linear_fun,
            'softmax': softmax_fun,
        }

        if function_name in functions:
            return functions[function_name]
        else:
            raise ValueError(f"Activation function '{function_name}' not supported.")


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
    return np.maximum(x, 0)


def relu_derivative(x):
    """The derivative of the ReLU function."""
    return np.where(x > 0, 1, 0)


def linear(x):
    """The linear activation function."""
    return x


def linear_derivative(x):
    """The derivative of the linear activation function."""
    return 1.0


def softmax(x):
    """The Softmax activation function."""
    return np.exp(x) / sum(np.exp(x))

def softmax_derivative(x):
    """The derivative of the Softmax function."""
    s = softmax(x)
    return np.diagflat(s) - np.outer(s, s)


softmax_fun = ActivationFunction(name='softmax', function=softmax, derivative=softmax_derivative)
linear_fun = ActivationFunction(name='linear', function=linear, derivative=linear_derivative)
sigmoid_fun = ActivationFunction(name='sigmoid', function=sigmoid, derivative=sigmoid_derivative)
tanh_fun = ActivationFunction(name='tanh', function=tanh, derivative=tanh_derivative)
relu_fun = ActivationFunction(name='relu', function=relu, derivative=relu_derivative)
