import os

import numpy as np
import pandas as pd

from typing import Optional, Tuple
from keras.datasets import mnist


class MNISTDataLoader:
    """ Loader for handling MNIST dataset, including loading from keras datasets or local CSV files."""
    def __init__(self, train_path: str, test_path: str):
        self._train_filename = None
        self._test_filename = None
        self._train_path = train_path
        self._test_path = test_path
        self.train_dataset = None
        self.test_dataset = None
        self.set_default_filenames()

    def set_default_filenames(self):
        self._train_filename = "train_dataset.csv"
        self._test_filename = "test_dataset.csv"

    @property
    def train_filename(self) -> str:
        return self._train_filename

    @train_filename.setter
    def train_filename(self, value):
        self._train_filename = value

    @property
    def test_filename(self) -> str:
        return self._test_filename

    @test_filename.setter
    def test_filename(self, value):
        self._test_filename = value

    @property
    def train_path(self) -> str:
        return self._train_path

    @train_path.setter
    def train_path(self, value):
        self._train_path = value

    @property
    def test_path(self) -> str:
        return self._test_path

    @test_path.setter
    def test_path(self, value):
        self._test_path = value

    def load_mnist_data(self, use_local: Optional[bool] = False, save: Optional[bool] = False) -> None:
        """
        Load MNIST data, either from local CSV files or keras datasets.

        :param use_local: If True, load data from local CSV files. If False, use keras datasets.
        :param save: If True, save the datasets to CSV files.
        """
        if use_local:
            self.load_from_csv()
        else:
            train, test = mnist.load_data()
            self.train_dataset = Dataset(*train)
            self.test_dataset = Dataset(*test)
            self.save_datasets() if save else None

    def load_from_csv(self) -> None:
        """
        Load MNIST data from local CSV files.

        :raises FileNotFoundError: If one or both of the CSV files are not found.
        """
        try:
            train_filepath = os.path.join(self._train_path, self._train_filename)
            test_filepath = os.path.join(self._test_path, self._test_filename)

            self.train_dataset = Dataset.from_csv(train_filepath)
            self.test_dataset = Dataset.from_csv(test_filepath)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"CSV file not found: {e.filename}")

    def save_datasets(self) -> None:
        """
        Save train and test datasets to CSV files.

        :raises ValueError: If train and test datasets are not initialized.
        """
        if self.train_dataset is not None and self.test_dataset is not None:
            train_df = self.train_dataset.to_dataframe()
            test_df = self.test_dataset.to_dataframe()

            train_filepath = os.path.join(self._train_path, self._train_filename)
            test_filepath = os.path.join(self._test_path, self._test_filename)

            train_df.to_csv(train_filepath, index=False)
            test_df.to_csv(test_filepath, index=False)
        else:
            raise ValueError("Train and test dataset is not initialized. Use load_mnist_data() firstly.")

    def normalize_datasets(self, mean: float = 0.1307, std: float = 0.3081) -> None:
        """
        Normalize both train and test datasets.

        :param mean: Mean value for normalization.
        :param std: Standard deviation value for normalization.
        """
        if self.train_dataset is not None and self.test_dataset is not None:
            self.train_dataset.normalize(mean=mean, std=std)
            self.test_dataset.normalize(mean=mean, std=std)
        else:
            raise ValueError("Train and test datasets are not initialized. Use load_mnist_data() firstly.")


class Dataset:
    """ Representation of a dataset with inputs and labels. """
    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        self.inputs = inputs
        self.labels = labels

    def __repr__(self) -> str:
        return f"Dataset(inputs shape={self.inputs.shape}, labels shape={self.labels.shape})"

    @property
    def shape(self) -> Tuple[Tuple, Tuple]:
        return self.inputs.shape, self.labels.shape

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.

        :return: DataFrame containing labels and flattened pixel values.
        """
        flat_inputs = self.inputs.reshape(self.inputs.shape[0], -1)

        df = pd.DataFrame(
            data=np.c_[self.labels, flat_inputs],
            columns=['label'] + [f'pixel_{i}' for i in range(flat_inputs.shape[1])]
        )
        return df

    def normalize(self, mean: float = 0.1307, std: float = 0.3081) -> None:
        """
        Normalize the input data.

        :param mean: Mean value.
        :param std: Standard deviation value.
        """
        self.inputs = (self.inputs - mean) / std

    @classmethod
    def from_csv(cls, filepath) -> 'Dataset':
        """
        Create a Dataset instance from a CSV file.

        :param filepath: Path to the CSV file.
        :return: Instance of the Dataset class.
        """
        df = pd.read_csv(filepath)
        labels = df['label'].values
        inputs = df.drop('label', axis=1).values.reshape((df.shape[0], 28, 28))
        return cls(inputs, labels)
