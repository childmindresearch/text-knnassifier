""" Module for implementing file compression methods. """
from __future__ import annotations

import bz2
import gzip
import lzma

ALGORITHMS = {"bz2": bz2, "gzip": gzip, "lzma": lzma}


class Compressor:
    """A class for compressing data using various algorithms.

    Attributes:
        algorithm: The compression algorithm to use. Defaults to 'gzip'.
    """

    def __init__(self, algorithm: str = "gzip"):
        """Initializes a Compressor object with the specified compression algorithm.

        Args:
            algorithm (str): The compression algorithm to use. Defaults to 'gzip'.
        """
        try:
            self.algorithm = ALGORITHMS[algorithm]
        except KeyError as exc_info:
            raise ValueError(
                f"Invalid compression algorithm '{algorithm}'. Valid options are {list(ALGORITHMS.keys())}."
            ) from exc_info

    def fit(self, data: str) -> bytes:
        """Compresses the input data using the specified algorithm.

        Args:
            data: The data to compress.

        Returns:
            bytes: The compressed data.
        """
        compressed = self.algorithm.compress(data.encode("utf-8"))
        # Ensure identical output type across compression methods
        return bytes(compressed)
