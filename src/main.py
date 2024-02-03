from src.neural_network import Layer, MNISTNeuralNetwork
from src.data_loader import MNISTDataLoader
from src.gui import MNISTGui
from activations import linear, softmax, relu, tanh, sigmoid


def main():
    custom_loader = MNISTDataLoader(train_fpath="../data/train", test_fpath="../data/test")
    custom_loader.load_mnist_data()
    custom_loader.normalize_datasets(mean=0.0, std=255.)

    model = MNISTNeuralNetwork(
        layers=[
            Layer(num_neurons=784, function=linear),
            Layer(num_neurons=200, function=relu),
            Layer(num_neurons=150, function=tanh),
            Layer(num_neurons=10, function=softmax)
        ],
        loader=custom_loader
    )

    epochs = 5
    learning_rate = 0.01
    model.train(
        epochs=epochs,
        batch_size=16,
        decay_rate=0.7,
        learning_rate=learning_rate
    )

    print(f"Test accuracy of model: {model.evaluate() * 100:.2f}")
    model.predict(*custom_loader.get_sample(), plot=True)

    # gui = MNISTGui(model)
    # gui.start()


if __name__ == '__main__':
    main()
