# TextKNNClassifier

`TextKNNClassifier` is a k-nearest neighbors classifier for text data. It uses a compression algorithm to compute the distance between texts and predicts the label of a test entry based on the labels of the k-nearest neighbors in the training data.

## Installation

You can install `TextKNNClassifier` using pip:

```bash
pip install textknnclassifier
```

## Usage

Here's an example of how to use `TextKNNClassifier`:

```python
from textknnassifier import classifier

training_data = [
    classifier.DataEntry(text="This is a test", label="test1"),
    classifier.DataEntry(text="Another test", label="test1"),
    classifier.DataEntry(text="General Tarkin", label="test2"),
    classifier.DataEntry(text="General Grievous", label="test2"),
]

testing_data = [
    classifier.DataEntry(text="This is a test"),
    classifier.DataEntry(text="General Patton"),
]

KNN = classifier.TextKNNClassifier()
predicted_labels = KNN.fit(training_data, testing_data, n_labels=2)

print(predicted_labels)
# Output: ['test1', 'test2']
```

In this example, we create a `TextKNNClassifier` instance and use it to predict the labels of the test entries. The `fit` method takes three arguments: the training data, the testing data, and the number of labels. The training data is a list of `DataEntry` objects, which contain the text and label of each training entry. The testing data is a list of `DataEntry` objects, which contain the text of each testing entry. The number of labels is the number of labels to consider when predicting the label of a test entry. The `fit` method returns a list of predicted labels.

## References

- Jiang, Z., Yang, M., Tsirlin, M., Tang, R., Dai, Y., & Lin, J. (2023, July). “Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors. In Findings of the Association for Computational Linguistics: ACL 2023 (pp. 6810-6828).
