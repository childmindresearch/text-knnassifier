# TextKNNClassifier

`TextKNNClassifier` is a k-nearest neighbors classifier for text data. It uses a compression algorithm to compute the distance between texts and predicts the label of a test entry based on the labels of the k-nearest neighbors in the training data.

## Installation

You can install `TextKNNassifier` using pip:

```bash
pip install textknnassifier
```

## Usage

Here's an example of how to use `TextKNNClassifier`:

```python
from textknnassifier import classifier

training_text = [
    "This is a test",
    "Another test",
    "General Tarkin",
    "General Grievous",
]
training_labels = ["test", "test", "star_wars", "star_wars"]


testing_data = [
    "This is a test",
    "Testing here too!",
    "General Kenobi",
    "General Skywalker",
]


KNN = classifier.TextKNNClassifier(max_neighbors=2)
KNN.fit(training_data, training_labels)
predicted_labels = KNN.predict(testing_data)
print(predicted_labels)
# Output: ['test1', 'test1', 'star_wars', 'star_wars']
```

In this example, we create a `TextKNNClassifier` instance and use it to predict the labels of the test entries. The initialization is given `max_neighbors=2`, this denotes the number of training datapoints to consider for predicting the testing label. The `fit` method takes two arguments: the training data, and the training labels. It simply stores these values for later use. The `predict` method takes the testing data as an argument and returns the predicted labels.

## References

- Jiang, Z., Yang, M., Tsirlin, M., Tang, R., Dai, Y., & Lin, J. (2023, July). “Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors. In Findings of the Association for Computational Linguistics: ACL 2023 (pp. 6810-6828).
