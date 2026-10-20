import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference


@pytest.fixture
def sample_data():
    # generate small synthetic dataset for testing
    np.random.seed(42)
    X = np.random.rand(50, 10)
    y = np.random.choice([0, 1], size=50)
    return X, y


# TODO: implement the first test. Change the function name and input as needed
def test_train_model(sample_data):
    # check that model trains and returns a random forest
    X, y = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_")


# TODO: implement the second test. Change the function name and input as needed
def test_inference(sample_data):
    # verify predictions have correct shape and binary values
    X, y = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    # check metrics return valid float scores between 0 and 1
    y = np.array([1, 1, 0, 0, 1])
    preds = np.array([1, 0, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)

# TP = 2, FP = 0, FN = 1 -> Precision = 1.0, Recall = 2/3, F1 = 0.8
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(0.8)
