from neural_network import Layer, NeuralNetwork
from src.data_loader import MNISTDataLoader
from activations import linear_fun, tanh_fun, softmax_fun

def main():
    custom_loader = MNISTDataLoader(train_path="../data/train", test_path="../data/test")
    custom_loader.load_mnist_data(use_local=True)
    custom_loader.normalize_datasets()

    layers = [
        Layer(num_neurons=784, function=linear_fun),
        Layer(num_neurons=128, function=tanh_fun),
        Layer(num_neurons=10, function=softmax_fun)
    ]

    neural_network = NeuralNetwork(layers=layers, loader=custom_loader)

    epochs = 2
    learning_rate = 0.001
    neural_network.train(epochs=epochs, batch_size=16, learning_rate=learning_rate)

    # # Evaluate the performance on the test dataset
    # accuracy = neural_network.evaluate()
    # print(f"Accuracy on the test dataset: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
