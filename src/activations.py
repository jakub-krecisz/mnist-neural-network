import numpy as np


class ActivationFunction:
    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.derivative = derivative

    @classmethod
    def get_function_by_name(cls, function_name):
        functions = {
            'sigmoid': sigmoid,
            'tanh': tanh,
            'relu': relu,
            'linear': linear,
            'softmax': softmax,
        }

        if function_name in functions:
            return functions[function_name]
        else:
            raise ValueError(f"Activation function '{function_name}' not supported.")


def _sigmoid(x):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_derivative(x):
    """The derivative of the sigmoid function."""
    return _sigmoid(x) * (1 - _sigmoid(x))


def _tanh(x):
    """The hyperbolic tangent (tanh) function."""
    return np.tanh(x)


def _tanh_derivative(x):
    """The derivative of the hyperbolic tangent (tanh) function."""
    return 1 - np.tanh(x) ** 2


def _relu(x):
    """The ReLU function."""
    return np.maximum(x, 0)


def _relu_derivative(x):
    """The derivative of the ReLU function."""
    return np.where(x > 0, 1, 0)


def _linear(x):
    """The linear activation function."""
    return x


def _linear_derivative(x):
    """The derivative of the linear activation function."""
    return 1.0


def _softmax(x):
    """The Softmax activation function."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / sum(exp_x)

def _softmax_derivative(x):
    """The derivative of the Softmax function."""
    s = _softmax(x)
    return np.diagflat(s) - np.outer(s, s)


softmax = ActivationFunction(name='softmax', function=_softmax, derivative=_softmax_derivative)
linear = ActivationFunction(name='linear', function=_linear, derivative=_linear_derivative)
sigmoid = ActivationFunction(name='sigmoid', function=_sigmoid, derivative=_sigmoid_derivative)
tanh = ActivationFunction(name='tanh', function=_tanh, derivative=_tanh_derivative)
relu = ActivationFunction(name='relu', function=_relu, derivative=_relu_derivative)
