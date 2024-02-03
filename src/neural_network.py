import csv

from typing import List
from matplotlib import pyplot as plt
from data_loader import MNISTDataLoader
from activations import ActivationFunction
from utils import *


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
                    accumulated_err += calculate_error(output, batch_labels)
                    average_error = accumulated_err / ((i // batch_size) + 1) * (epoch + 1)
                    print("Epoch: {} | Iteration: {:<5} | Accuracy: {:<5} | Error: {:<10.5f}"
                          .format(epoch, i, accuracy, average_error))

            learning_rate *= decay_rate

    def backward_propagation(self, output, batch_labels, total_samples):
        delta_output = output - one_hot(batch_labels)

        for i in range(self.num_of_layers - 1, 0, -1):
            a_prev = self.layers[i - 1].activation_function.function(self.layers[i - 1].z)

            delta_weights = 1 / total_samples * np.dot(delta_output, a_prev.T)
            delta_biases = 1 / total_samples * np.sum(delta_output)

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
        prediction = get_predictions(output)
        return get_accuracy(prediction, test_labels)

    def predict(self, input_data: np.ndarray, actual_label: int, plot=False):
        output = self.forward_propagation(input_data[np.newaxis, ...])
        prediction = get_predictions(output)[0]

        if plot:
            plt.gray()
            plt.imshow(input_data, interpolation='nearest')

            plt.title(f"Prediction: {prediction}, Actual Label: {actual_label}")
            plt.show()
        return prediction

    @classmethod
    def save_model(cls, model, path):
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)

            for layer in model.layers:
                writer.writerow(["Layer", layer.num_neurons])
                writer.writerow(["ActivationFunction", layer.activation_function.name])
                writer.writerow(
                    ["Weights"] + (layer.weights.flatten().tolist() if layer.weights is not None else ['-']))
                writer.writerow(["Biases"] + (layer.biases.flatten().tolist() if layer.biases is not None else ['-']))

    @classmethod
    def load_model(cls, path, loader: MNISTDataLoader):
        layers = []

        with open(path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == "Layer":
                    num_neurons = int(row[1])
                    function = ActivationFunction.get_function_by_name(next(reader)[1])
                    weights = next(reader)[1:]
                    biases = next(reader)[1:]

                    weights_formatted = np.array(list(map(float, weights))).reshape((num_neurons, layers[-1].num_neurons)) if weights != ['-'] else None
                    biases_formatted = np.array(list(map(float, biases))).reshape((num_neurons, 1)) if biases != ['-'] else None

                    current_layer = Layer(num_neurons, function)
                    current_layer.biases = biases_formatted
                    current_layer.weights = weights_formatted

                    layers.append(current_layer)
        return cls(layers, loader, init_weights_n_biases=False)
