""" A k-nearest neighbors classifier for text data. """
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pydantic

from textknnassifier import compressor


class DataEntry(pydantic.BaseModel):
    """
    Represents a single data entry for the TextKNNClassifier.

    Attributes:
        text: The text of the entry.
        label: The label of the entry.
    """

    text: str = pydantic.Field(..., description="The text of the entry.", min_length=1)
    label: Optional[str] = pydantic.Field(
        None, description="The label of the entry.", min_length=1
    )


class TextKNNClassifier:
    """
    A k-nearest neighbors classifier for text data.

    Attributes:
        algorithm: The compression algorithm to use for computing the distance
            between texts.
        n_labels: The number of nearest neighbors to consider when predicting
            the label of a test entry.

    References:
        Jiang, Z., Yang, M., Tsirlin, M., Tang, R., Dai, Y., & Lin, J. (2023,
        July). “Low-Resource” Text Classification: A Parameter-Free
        Classification Method with Compressors. In Findings of the Association
        for Computational Linguistics: ACL 2023 (pp. 6810-6828).

    """

    def __init__(self, algorithm: str = "gzip", n_labels: int = 10):
        self.compressor = compressor.Compressor(algorithm=algorithm)
        self.n_labels = n_labels

    def fit(
        self, training: Iterable[DataEntry], testing: Iterable[DataEntry]
    ) -> list[str]:
        """
        Fits the TextKNNClassifier to the training data and predicts the labels for the testing data.

        Args:
            training: An iterable of DataEntry objects representing the training data.
            testing: An iterable of DataEntry objects representing the testing data.

        Returns:
            A list of predicted labels for the testing data.
        """
        if any((not entry.label for entry in training)):
            raise ValueError("All training entries must have a label.")

        return [self._predict_class(entry, training) for entry in testing]

    def _predict_class(self, test_entry, training) -> str:
        """
        Predicts the label for a single test entry based on the labels of
        the k-nearest neighbors in the training data.

        Args:
            test_entry: A DataEntry object representing the test entry.
            training: An iterable of DataEntry objects representing the training data.

        Returns:
            The predicted label for the test entry.
        """
        distance_from_training = [
            self._compute_distance(test_entry.text, train_entry.text)
            for train_entry in training
        ]
        sorted_indices = np.argsort(distance_from_training)
        top_k_class = [training[i].label for i in sorted_indices[: self.n_labels]]
        predicted_class = max(set(top_k_class), key=top_k_class.count)
        return predicted_class

    def _compute_distance(self, text_1: str, text_2: str) -> float:
        """
        Computes the normalized compressed distance between two texts.

        Args:
            text_1: The first text.
            text_2: The second text.

        Returns:
            The normalized compressed distance between the two texts.
        """
        combined = text_1 + " " + text_2
        compressed_size_combined = len(self.compressor.fit(combined))
        compressed_size_1 = len(self.compressor.fit(text_1))
        compressed_size_2 = len(self.compressor.fit(text_2))
        normalized_distance = (
            compressed_size_combined - min(compressed_size_1, compressed_size_2)
        ) / max(compressed_size_1, compressed_size_2)
        return normalized_distance
