import numpy as np

from typing import List
from data_loader import MNISTDataLoader


class Layer:
    def __init__(self, num_neurons: int, activation_function):
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.weights = None
        self.biases = None


class NeuralNetwork(object):
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
            # Initialize weights and biases from a normal distribution
            self.layers[i].weights = np.random.randn(output_size, input_size)
            self.layers[i].biases = np.zeros((output_size, 1))

    def forward_propagation(self, inputs):
        """Performs forward propagation through the network."""
        for i in range(1, self.num_of_layers):
            z = np.dot(self.layers[i].weights, inputs) + self.layers[i].biases
            inputs = self.layers[i].activation_function(z)
        return inputs

    def train(self, epochs, learning_rate):
        """Trains the neural network on the training dataset."""
        for epoch in range(epochs):
            for inputs, labels in zip(self.data_loader.train_dataset.inputs, self.data_loader.train_dataset.labels):
                inputs = inputs.flatten().reshape((-1, 1))  # Flatten input data
                labels = labels.reshape((-1, 1))  # Flatten labels

                # Forward propagation
                activations = [inputs]
                for i in range(1, self.num_of_layers):
                    z = np.dot(self.layers[i].weights, activations[-1]) + self.layers[i].biases
                    a = self.layers[i].activation_function(z)
                    activations.append(a)

                # Backpropagation
                delta = (activations[-1] - labels) * activations[-1] * (1 - activations[-1])
                for i in range(self.num_of_layers - 1, 0, -1):
                    self.layers[i].biases -= learning_rate * delta
                    self.layers[i].weights -= learning_rate * np.dot(delta, activations[i - 1].T)
                    delta = np.dot(self.layers[i].weights.T, delta) * activations[i - 1] * (1 - activations[i - 1])

    def evaluate(self):
        """Evaluates the neural network on the test dataset."""
        correct_predictions = 0
        total_samples = len(self.data_loader.test_dataset.inputs)

        for inputs, labels in zip(self.data_loader.test_dataset.inputs, self.data_loader.test_dataset.labels):
            inputs = inputs.flatten().reshape((-1, 1))
            labels = labels.reshape((-1, 1))

            prediction = self.forward_propagation(inputs)
            predicted_label = np.argmax(prediction)
            true_label = np.argmax(labels)

            if predicted_label == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / total_samples
        return accuracy
