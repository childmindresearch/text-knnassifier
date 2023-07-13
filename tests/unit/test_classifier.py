""" Unit tests for the classifier module. """
# pylint: disable=protected-access,redefined-outer-name
import gzip

import pytest

from textknnassifier import classifier


@pytest.fixture
def training_data() -> list[str]:
    """Returns a list of DataEntry objects for the training dataset.

    Returns:
        list[classifier.DataEntry]: A list of DataEntry objects.
    """
    return [
        "This is a test",
        "Another test",
        "General Tarkin",
        "General Grievous",
    ]


@pytest.fixture
def training_labels() -> list[str]:
    """Returns a list of labels for the training dataset.

    Returns:
        list[str]: A list of labels.
    """
    return ["test", "test", "star_wars", "star_wars"]


@pytest.fixture
def testing_data() -> list[str]:
    """Returns a list of DataEntry objects for the testing dataset.

    Returns:
        list[classifier.DataEntry]: A list of DataEntry objects.
    """
    return [
        "This is a test",
        "Testing here too!",
        "General Kenobi",
        "General Skywalker",
    ]


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
    training_data: list[str], training_labels: list[str]
) -> None:
    """Test the prediction of the label for a single test entry.

    Args:
        training_data: A fixture for the DataEntry objects for the training dataset.
    """
    gzip_knn = classifier.TextKNNClassifier(algorithm="gzip", n_labels=2)
    gzip_knn.fit(training_data, training_labels)
    predicted_class = gzip_knn._predict_class(training_data[0])

    assert predicted_class == training_labels[0]


def test_text_knn_classifier_fit(
    training_data: list[str], training_labels: list[str], testing_data: list[str]
) -> None:
    """Test the fitting of the TextKNNClassifier to the training data and the prediction of the labels for the testing data.

    Args:
        training_data: A fixture for the DataEntry objects for the training dataset.
        testing_data: A fixture for the DataEntry objects for the testing dataset.
    """
    gzip_knn = classifier.TextKNNClassifier(algorithm="gzip", n_labels=2)
    gzip_knn.fit(training_data, training_labels)
    predicted_classes = gzip_knn.predict(testing_data)

    assert predicted_classes == training_labels
