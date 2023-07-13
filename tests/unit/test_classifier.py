""" Unit tests for the classifier module. """
# pylint: disable=protected-access,redefined-outer-name
import gzip

import pytest

from textknnassifier import classifier


@pytest.fixture
def training_data() -> list[classifier.DataEntry]:
    """Returns a list of DataEntry objects for the training dataset.

    Returns:
        list[classifier.DataEntry]: A list of DataEntry objects.
    """
    return [
        classifier.DataEntry(text="This is a test", label="test1"),
        classifier.DataEntry(text="Another test", label="test1"),
        classifier.DataEntry(text="General Tarkin", label="test2"),
        classifier.DataEntry(text="General Grievous", label="test2"),
    ]


@pytest.fixture
def testing_data() -> list[classifier.DataEntry]:
    """Returns a list of DataEntry objects for the testing dataset.

    Returns:
        list[classifier.DataEntry]: A list of DataEntry objects.
    """
    return [
        classifier.DataEntry(text="This is a test"),
        classifier.DataEntry(text="Testing here too!"),
        classifier.DataEntry(text="General Kenobi"),
        classifier.DataEntry(text="General Skywalker"),
    ]


def test_data_entry_init_no_label() -> None:
    """Test the initialization of a DataEntry object without a label."""
    entry = classifier.DataEntry(text="test")

    assert entry.text == "test"
    assert entry.label is None


def test_data_entry_init() -> None:
    """Test the initialization of a DataEntry object with a label."""
    entry = classifier.DataEntry(text="test", label="test")

    assert entry.text == "test"
    assert entry.label == "test"


def test_data_entry_empty_values() -> None:
    """Test the initialization of a DataEntry object with invalid text."""
    with pytest.raises(ValueError):
        classifier.DataEntry(text="")

    with pytest.raises(ValueError):
        classifier.DataEntry(text="s", label="")


def test_text_knn_classifier_init() -> None:
    """Test the initialization of a TextKNNClassifier object."""
    gzip_knn = classifier.TextKNNClassifier(algorithm="gzip", n_labels=10)

    assert gzip_knn.compressor.algorithm == gzip
    assert gzip_knn.n_labels == 10


def test_text_knn_classifier_compute_distance_identical() -> None:
    """Test the computation of the normalized compressed distance between two texts."""
    gzip_knn = classifier.TextKNNClassifier(algorithm="gzip", n_labels=10)
    distance = gzip_knn._compute_distance("test", "test")

    # Currently I'm not sure how to properly test this, but 0.125 is the value
    # found in the original implementation. So at least this can be used to
    # ensure that the implementation remains identical.
    assert distance == 0.125


def test_text_knn_classifier_predict_class(
    training_data: list[classifier.DataEntry],
) -> None:
    """Test the prediction of the label for a single test entry.

    Args:
        training_data: A fixture for the DataEntry objects for the training dataset.
    """
    gzip_knn = classifier.TextKNNClassifier(algorithm="gzip", n_labels=2)
    predicted_class = gzip_knn._predict_class(training_data[0], training_data)

    assert predicted_class == training_data[0].label


def test_text_knn_classifier_fit(
    training_data: list[classifier.DataEntry],
    testing_data: list[classifier.DataEntry],
) -> None:
    """Test the fitting of the TextKNNClassifier to the training data and the prediction of the labels for the testing data.

    Args:
        training_data: A fixture for the DataEntry objects for the training dataset.
        testing_data: A fixture for the DataEntry objects for the testing dataset.
    """
    gzip_knn = classifier.TextKNNClassifier(algorithm="gzip", n_labels=2)
    predicted_classes = gzip_knn.fit(training_data, testing_data)

    assert predicted_classes == ["test1", "test1", "test2", "test2"]
