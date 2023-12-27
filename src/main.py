from neural_network import Layer, NeuralNetwork
from src.data_loader import MNISTDataLoader
from activations import linear_activation, tanh, softmax

def main():
    # Initialize MNISTDataLoader
    custom_loader = MNISTDataLoader(train_path="../data/train", test_path="../data/test")
    custom_loader.load_mnist_data(use_local=True)
    custom_loader.normalize_datasets()

    # Define the architecture of the neural network
    input_layer = Layer(num_neurons=784, activation_function=linear_activation)
    hidden_layer = Layer(num_neurons=128, activation_function=tanh)
    output_layer = Layer(num_neurons=10, activation_function=softmax)

    layers = [input_layer, hidden_layer, output_layer]

    # Initialize NeuralNetwork
    neural_network = NeuralNetwork(layers=layers, loader=custom_loader)

    # Train the neural network
    epochs = 10
    learning_rate = 0.1
    neural_network.train(epochs=epochs, learning_rate=learning_rate)

    # Evaluate the performance on the test dataset
    accuracy = neural_network.evaluate()
    print(f"Accuracy on the test dataset: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
