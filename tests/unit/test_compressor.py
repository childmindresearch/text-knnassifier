""" Unit tests for the Compressor class. """
from pytest import mark

from textknnassifier.compressor import Compressor


@mark.parametrize("algorithm", ["gzip", "bz2", "lzma"])
def test_compressor_gzip(algorithm: str) -> None:
    """Test that the Compressor class can compress data using the gzip algorithm.

    Args:
        algorithm (str): The compression algorithm to use.
    """
    compressor = Compressor(algorithm=algorithm)
    data = "hello world" * 1000

    compressed = compressor.fit(data)

    assert isinstance(compressed, bytes)
    assert len(compressed) < len(data.encode("utf-8"))


def test_compressor_invalid_algorithm():
    """Test that an invalid compression algorithm raises a ValueError with a
    specific error message."""
    try:
        Compressor(algorithm="invalid_algorithm")
    except ValueError as exc_info:
        assert (
            str(exc_info)
            == "Invalid compression algorithm 'invalid_algorithm'. Valid options are ['bz2', 'gzip', 'lzma']."
        )
