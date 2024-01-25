from neural_network import Layer, MNISTNeuralNetwork
from src.data_loader import MNISTDataLoader
from activations import linear, softmax, relu, tanh


def main():
    custom_loader = MNISTDataLoader(train_path="../data/train", test_path="../data/test")
    custom_loader.load_mnist_data(use_local=True)
    custom_loader.normalize_datasets(mean=0.0, std=255.)

    layers = [
        Layer(num_neurons=784, function=linear),
        Layer(num_neurons=400, function=relu),
        Layer(num_neurons=150, function=tanh),
        Layer(num_neurons=10, function=softmax)
    ]

    neural_network = MNISTNeuralNetwork(layers=layers, loader=custom_loader)

    epochs = 10
    learning_rate = 0.01
    neural_network.train(epochs=epochs, batch_size=16, learning_rate=learning_rate)
    print(f"Test accuracy: {(test_accuracy := neural_network.evaluate())}")
    MNISTNeuralNetwork.save_model(neural_network, f'../models/trained_model_{test_accuracy * 100:.0f}_acc.csv')

    # loaded_model = MNISTNeuralNetwork.load_model('../models/trained_model_89_acc.csv', custom_loader)
    # test_accuracy_loaded_model = loaded_model.evaluate()
    # print(f"Test accuracy, tested on 10k samples on loaded model: {test_accuracy_loaded_model}")


if __name__ == '__main__':
    main()
