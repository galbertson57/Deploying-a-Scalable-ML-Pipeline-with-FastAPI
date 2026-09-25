import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    Test train_model returns fitted Random Forest Classifier
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    Test compute_model_metrics returns expected precision, recall, and
    F1 values on set of labels/predictions
    """
    y = np.array([1, 0, 1, 1, 0])
    preds = np.array([1, 0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(0.8)


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    Test inference returns predictions of correct type/shape, matching number
    of input rows.
    """
    data = pd.DataFrame({
        "age": [25, 40, 33, 52],
        "workclass": ["Private", "Self-emp", "Private", "Gov"],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"]
    })
    cat_features = ["workclass"]
    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
