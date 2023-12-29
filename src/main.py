from neural_network import Layer, MNISTNeuralNetwork
from src.data_loader import MNISTDataLoader
from activations import linear_fun, softmax_fun, relu_fun, tanh_fun


def main():
    custom_loader = MNISTDataLoader(train_path="../data/train", test_path="../data/test")
    custom_loader.load_mnist_data(use_local=True)
    custom_loader.normalize_datasets(mean=0.0, std=255.)

    layers = [
        Layer(num_neurons=784, function=linear_fun),
        Layer(num_neurons=400, function=relu_fun),
        Layer(num_neurons=150, function=tanh_fun),
        Layer(num_neurons=10, function=softmax_fun)
    ]

    neural_network = MNISTNeuralNetwork(layers=layers, loader=custom_loader)

    epochs = 10
    learning_rate = 0.01
    neural_network.train(epochs=epochs, batch_size=16, learning_rate=learning_rate)
    test_accuracy = neural_network.evaluate()

    print(f"Test accuracy, tested on 10k samples: {test_accuracy}")


if __name__ == '__main__':
    main()
