# Simple MNIST Neural Network


## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)


## Overview

The project encompasses the implementation from the ground up of a simple feed-forward neural network, which is constructed solely using the NumPy library and trained on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). As a result, the model is capable of recognizing hand-written digits. Additionally, the inclusion of a straightforward interactive GUI allows users to draw digits themselves and assess the model's accuracy.

## Project Structure

- `main.py`: The main script for initialization, training the network (or loading it), and interaction with the GUI.
- `utils.py`: Helper functions for one-hot encoding, obtaining predictions, and calculating accuracy.
- `neural_network.py`: Implementation of layers (Layer class) and the entire network (MNISTNeuralNetwork class), with methods for forward and backward propagation, training, evaluation, and model saving/loading.
- `gui.py`: Simple interactive GUI for drawing digits and receiving predictions.
- `data_loader.py`: Module for loading and processing data from the MNIST dataset. This module provides functionality for both downloading the dataset and loading it locally. It supports normalization and offers options for saving the dataset or reloading it anew each time the program is run.
- `activations.py`: Contains activation functions (sigmoid, tanh, relu, linear, softmax) and their derivatives.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jakub-krecisz/mnist-neural-network.git
    ```

2. Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```
   
3. Sometimes an additional tool is needed without which the gui does not work
   - On Linux: Install Ghostscript using your system's package manager. For example, on Ubuntu, you can run ```sudo apt-get install ghostscript```.
   - On macOS: Use Homebrew with `brew install ghostscript`.
   - On Windows: Download and install the Ghostscript executable from the official website: [Ghostscript Downloads](https://www.ghostscript.com/download/gsdnld.html).

## Usage

To get started, simply:
```bash
python main.py
```


By default, the loading of the MNIST dataset from the internet is implemented without saving, along with a sample implementation of a neural network, training (with all parameters set), and printing the model's accuracy, followed by testing on a sample. To use the GUI, simply uncomment:

```python
gui = MNISTGui(model)
gui.start()
```

<br>
To save the loaded MNIST dataset during model loading, use:

```python
custom_loader.load_mnist_data(save=True)
```
<br>
To avoid downloading the dataset each time when it's already locally available, use:

```python
custom_loader.load_mnist_data(use_local=True)
```
<br>
To load a model from saved files, use:

```python
model = MNISTNeuralNetwork.load_model('file_path/filename.csv', custom_loader)
```
<br>
To save a trained model, use:

```python
MNISTNeuralNetwork.save_model(model, 'file_path/filename.csv')
```

## Demo
<p align="center">
  <img src="https://github.com/jakub-krecisz/mnist-neural-network/assets/93099511/32d8d53c-b043-4862-aeb3-2e02870b95d9" alt="demo_mnist" width="400">
</p>
