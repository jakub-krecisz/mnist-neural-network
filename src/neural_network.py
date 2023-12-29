import numpy as np

from typing import List
from data_loader import MNISTDataLoader
from activations import ActivationFunction


class Layer:
    def __init__(self, num_neurons: int, function: ActivationFunction):
        self.num_neurons = num_neurons
        self.activation_function = function
        self.weights = None
        self.biases = None
        self.z = None
        self.delta_weights = None
        self.delta_biases = None


class MNISTNeuralNetwork(object):
    def __init__(self, layers: List[Layer], loader: MNISTDataLoader):
        self.num_of_layers = len(layers)
        self.sizes = tuple(layer.num_neurons for layer in layers)
        self.data_loader = loader
        self.layers = layers
        self._initialize_weights_and_biases()

    def _initialize_weights_and_biases(self):
        for i in range(1, self.num_of_layers):
            input_size = self.sizes[i - 1]
            output_size = self.sizes[i]
            self.layers[i].weights = np.random.rand(output_size, input_size) - 0.5
            self.layers[i].biases = np.random.rand(output_size, 1) - 0.5

    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        layer_outputs = [inputs.reshape((inputs.shape[0], -1)).T]

        for i in range(1, self.num_of_layers):
            weights = self.layers[i].weights
            biases = self.layers[i].biases
            z = np.dot(weights, layer_outputs[-1]) + biases
            a = self.layers[i].activation_function.function(z)

            layer_outputs.append(a)
            self.layers[i].z = z
        self.layers[0].z = layer_outputs[0]
        return layer_outputs[-1]

    def train(self, epochs: int, batch_size: int, learning_rate: float):
        total_samples = len(self.data_loader.train_dataset.inputs)

        for epoch in range(epochs):
            indices = np.random.permutation(total_samples)
            inputs_shuffled = self.data_loader.train_dataset.inputs[indices]
            labels_shuffled = self.data_loader.train_dataset.labels[indices]

            for i in range(0, total_samples, batch_size):
                batch_inputs = inputs_shuffled[i:i + batch_size]
                batch_labels = labels_shuffled[i:i + batch_size]

                output = self.forward_propagation(batch_inputs)
                self.backward_propagation(output, batch_labels, len(batch_inputs))
                self.update_weights_and_biases(learning_rate)

            print("Epoch: ", epoch)
            predictions = self.get_predictions(output)
            accuracy = self.get_accuracy(predictions, batch_labels)
            print("Batch accuracy:", accuracy)

    def backward_propagation(self, output, batch_labels, total_samples):
        delta_output = output - self.one_hot(batch_labels)

        for i in range(self.num_of_layers - 1, 0, -1):
            a_prev = self.layers[i - 1].activation_function.function(self.layers[i - 1].z)

            delta_weights = 1 / total_samples * np.dot(delta_output, a_prev.T)
            delta_biases = 1 / total_samples * np.sum(delta_output)     # axis=1, keepdims=True

            delta_output = (np.dot(self.layers[i].weights.T, delta_output) *
                            self.layers[i - 1].activation_function.derivative(self.layers[i - 1].z))

            self.layers[i].delta_weights = delta_weights
            self.layers[i].delta_biases = delta_biases

    def update_weights_and_biases(self, learning_rate: float):
        for i in range(1, self.num_of_layers):
            self.layers[i].weights -= learning_rate * self.layers[i].delta_weights
            self.layers[i].biases -= learning_rate * self.layers[i].delta_biases

            self.layers[i].delta_weights = None
            self.layers[i].delta_biases = None

    def evaluate(self):
        output = self.forward_propagation(self.data_loader.test_dataset.inputs)
        prediction = self.get_predictions(output)
        return self.get_accuracy(prediction, self.data_loader.test_dataset.labels)

    @staticmethod
    def one_hot(labels: np.ndarray) -> np.ndarray:
        one_hot_labels = np.eye(10)[labels].T
        return one_hot_labels

    @staticmethod
    def get_predictions(output):
        return np.argmax(output, 0)

    @staticmethod
    def get_accuracy(predictions, batch_labels):
        return np.sum(predictions == batch_labels) / batch_labels.size
