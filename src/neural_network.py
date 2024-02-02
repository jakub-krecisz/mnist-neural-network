import csv
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
    def __init__(self, layers: List[Layer], loader: MNISTDataLoader, init_weights_n_biases=True):
        self.num_of_layers = len(layers)
        self.sizes = tuple(layer.num_neurons for layer in layers)
        self.data_loader = loader
        self.layers = layers
        self._initialize_weights_and_biases() if init_weights_n_biases else None

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

    def train(self, epochs: int, batch_size: int, learning_rate: float, decay_rate: float = 1.0, log_interval=100):
        total_samples = len(self.data_loader.train_dataset.inputs)
        accumulated_err = 0.

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

                if (i / batch_size) % log_interval == 0:
                    accuracy = self.evaluate(1000)
                    accumulated_err += self.calculate_error(output, batch_labels)
                    average_error = accumulated_err / ((i // batch_size) + 1)
                    print("Epoch: {} | Iteration: {:<5} | Accuracy: {:<5} | Error: {:<10.5f}"
                          .format(epoch, i, accuracy, average_error))

            learning_rate *= decay_rate

    def backward_propagation(self, output, batch_labels, total_samples):
        delta_output = output - self.one_hot(batch_labels)

        for i in range(self.num_of_layers - 1, 0, -1):
            a_prev = self.layers[i - 1].activation_function.function(self.layers[i - 1].z)

            delta_weights = 1 / total_samples * np.dot(delta_output, a_prev.T)
            delta_biases = 1 / total_samples * np.sum(delta_output)  # axis=1, keepdims=True

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

    def evaluate(self, num_of_samples=None, random=True):
        if num_of_samples is None:
            num_of_samples = len(self.data_loader.test_dataset)

        test_inputs = self.data_loader.test_dataset.inputs[:num_of_samples]
        test_labels = self.data_loader.test_dataset.labels[:num_of_samples]

        if random:
            indices = np.random.permutation(len(test_inputs))
            test_inputs = test_inputs[indices]
            test_labels = test_labels[indices]

        output = self.forward_propagation(test_inputs)
        prediction = self.get_predictions(output)
        return self.get_accuracy(prediction, test_labels)

    def calculate_error(self, output, labels):
        epsilon = 1e-15

        error = -1 / labels.size * np.sum(self.one_hot(labels) * np.log(output + epsilon))
        return error

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

    @classmethod
    def save_model(cls, model, path):
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)

            for layer in model.layers:
                writer.writerow(["Layer", layer.num_neurons])
                writer.writerow(["ActivationFunction", layer.activation_function.name])
                if layer.weights is not None:
                    writer.writerow(["Weights"] + layer.weights.flatten().tolist())
                if layer.biases is not None:
                    writer.writerow(["Biases"] + layer.biases.flatten().tolist())

    # @classmethod
    # def load_model(cls, path, loader: MNISTDataLoader):
    #     layers = []
    #     current_layer = None
    #
    #     with open(path, 'r') as file:
    #         reader = csv.reader(file)
    #         for row in reader:
    #             if row[0] == "Layer":
    #                 # Create a new layer
    #                 num_neurons = int(row[1])
    #                 function_name = next(reader)[1]
    #                 activation_function = ActivationFunction.get_function_by_name(function_name)
    #                 current_layer = Layer(num_neurons, activation_function)
    #
    #             elif row[0] == "Weights":
    #                 # Load weights for the current layer
    #                 current_layer.weights = np.array([float(x) for x in row[1:]]).reshape(current_layer.weights.shape)
    #
    #             elif row[0] == "Biases":
    #                 # Load biases for the current layer
    #                 current_layer.biases = np.array([float(x) for x in row[1:]]).reshape(current_layer.biases.shape)
    #
    #                 # Add the current layer to the list of layers
    #                 layers.append(current_layer)
    #
    #     # The first layer should not have biases or weights
    #     layers[0].biases = None
    #     layers[0].weights = None
    #
    #     # Create the model with the loaded layers and set the data loader
    #     loaded_model = cls(layers, loader)
    #     return loaded_model
