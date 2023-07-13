""" A k-nearest neighbors classifier for text data. """

# pylint: disable=invalid-name
# Disabled because it is requied to conform to the scikit-learn interface.

from __future__ import annotations

import statistics
from typing import Iterable, Optional, Sequence

from textknnassifier import compressor


class TextKNNClassifier:
    """A k-nearest neighbors classifier for text data.

    Attributes:
        compressor: A Compressor object used to compress the data.
        n_neighbors: The number of nearest neighbors to consider when predicting
            the label of a test entry.
        training_data: The training data used to fit the classifier.
        training_labels: The labels for the training data.

    References:
        Jiang, Z., Yang, M., Tsirlin, M., Tang, R., Dai, Y., & Lin, J. (2023,
        July). “Low-Resource” Text Classification: A Parameter-Free
        Classification Method with Compressors. In Findings of the Association
        for Computational Linguistics: ACL 2023 (pp. 6810-6828).

    """

    def __init__(self, algorithm: str = "gzip", n_neighbors: int = 10):
        """Initializes a TextKNNClassifier object.

        Args:
            algorithm: The compression algorithm to use. Defaults to "gzip".
            n_neighbors: The number of nearest neighbors to consider when predicting
                the label of a test entry. Defaults to 10.

        """
        self.compressor = compressor.Compressor(algorithm=algorithm)
        self.n_neighbors = n_neighbors
        self.training_data: Optional[Sequence[str]] = None
        self.training_labels: Optional[Sequence[str]] = None

    def fit(self, X: Sequence[str], y: Sequence[str]) -> None:
        """Sets up the TextKNNClassifier with the training data.

        Args:
            X: An iterable of strings representing the training data.
            y: An iterable of strings representing the labels for the training
                data.

        Notes:
            This function exists solely to conform to the standard scikit-learn
            interface.

        """
        if len(X) != len(y):
            raise ValueError(
                "Training data and training labels must have the same length."
            )

        self.training_data = X
        self.training_labels = y

    def predict(self, X: Iterable[str]) -> list[str]:
        """Fits the TextKNNClassifier to the training data and predicts the
        labels for the testing data.

        Args:
            training: An iterable of DataEntry objects representing the training data.
            testing: An iterable of DataEntry objects representing the testing data.

        Returns:
            A list of predicted labels for the testing data.
        """
        return [self._predict_class(test_entry) for test_entry in X]

    def _predict_class(self, test_entry) -> str:
        """Predicts the label for a single test entry based on the labels of
        the k-nearest neighbors in the training data.

        Args:
            test_entry: A DataEntry object representing the test entry.
            training: An iterable of DataEntry objects representing the training data.

        Returns:
            The predicted label for the test entry.
        """
        if not self.training_data or not self.training_labels:
            raise ValueError("The classifier must be fit to training data first.")

        distance_from_training = [
            self._compute_distance(test_entry, train_entry)
            for train_entry in self.training_data
        ]
        sorted_indices = sorted(
            range(len(distance_from_training)),
            key=lambda i: distance_from_training[i],
        )
        top_n_class = [
            self.training_labels[i] for i in sorted_indices[: self.n_neighbors]
        ]
        return statistics.mode(top_n_class)

    def _compute_distance(self, text_1: str, text_2: str) -> float:
        """Computes the normalized compressed distance between two texts.

        Args:
            text_1: The first text.
            text_2: The second text.

        Returns:
            The normalized compressed distance between the two texts.
        """
        combined = f"{text_1} {text_2}"
        compressed_size_combined = len(self.compressor.fit(combined))
        compressed_size_1 = len(self.compressor.fit(text_1))
        compressed_size_2 = len(self.compressor.fit(text_2))
        normalized_distance = (
            compressed_size_combined - min(compressed_size_1, compressed_size_2)
        ) / max(compressed_size_1, compressed_size_2)
        return normalized_distance
