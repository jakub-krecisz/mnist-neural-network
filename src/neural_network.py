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
            self.layers[i].weights = np.random.randn(output_size, input_size)
            self.layers[i].biases = np.zeros((output_size, 1))

    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        layer_outputs = [inputs.reshape((inputs.shape[0], -1)).T]

        for i in range(1, self.num_of_layers):
            weights = self.layers[i].weights
            biases = self.layers[i].biases
            z = np.dot(weights, layer_outputs[-1]) + biases
            self.layers[i].z = z  # Store the value of z
            a = self.layers[i].activation_function.function(z)
            layer_outputs.append(a)
        return layer_outputs[-1]

    def train(self, epochs: int, batch_size: int, learning_rate: float):
        total_samples = len(self.data_loader.train_dataset.inputs)

        for epoch in range(epochs):
            epoch_loss = 0  # Inicjalizacja sumy straty w danym epochu

            # Permute the data (shuffle) for each epoch
            indices = np.random.permutation(total_samples)
            inputs_shuffled = self.data_loader.train_dataset.inputs[indices]
            labels_shuffled = self.data_loader.train_dataset.labels[indices]

            for i in range(0, total_samples, batch_size):
                batch_inputs = inputs_shuffled[i:i + batch_size]
                batch_labels = labels_shuffled[i:i + batch_size]
                one_hot_batch_labels = self.one_hot(batch_labels)

                output = self.forward_propagation(batch_inputs)

                # Compute the loss (you need to implement this method)
                loss = self.mean_sq_error(output, one_hot_batch_labels)
                epoch_loss += loss

                if i % batch_size * 500 == 0:
                    average_loss = epoch_loss / (i / batch_size + 1)
                    print(f"Epoch {epoch + 1}/{epochs}, Iteration {i} / {total_samples // 16}, Average Loss: {average_loss}")

                # Backpropagation
                self.backward_propagation(output, one_hot_batch_labels)
                # Update weights and biases using gradient descent
                self.update_weights_and_biases(learning_rate)

                # Print average loss for the current epoch
            average_loss = epoch_loss / (total_samples / batch_size)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}")

    def backward_propagation(self, output, one_hot_batch_labels):
        pass

    def update_weights_and_biases(self, learning_rate: float):
        # Update weights and biases using the stored gradients
        for i in range(1, self.num_of_layers):
            self.layers[i].weights -= learning_rate * self.layers[i].delta_weights
            self.layers[i].biases -= learning_rate * self.layers[i].delta_biases

            # Clear stored gradients for the next iteration
            self.layers[i].delta_weights = None
            self.layers[i].delta_biases = None

    @staticmethod
    def mean_sq_error(output: np.ndarray, labels: np.ndarray) -> float:
        return np.mean((output - labels) ** 2)

    @staticmethod
    def one_hot(labels: np.ndarray) -> np.ndarray:
        """
        Convert integer labels to one-hot encoded format.

        :param labels: Integer labels.
        :return: One-hot encoded labels.
        """
        num_classes = labels.max() + 1
        one_hot_labels = np.eye(num_classes)[labels].T
        return one_hot_labels
