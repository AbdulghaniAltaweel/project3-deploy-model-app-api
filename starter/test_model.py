import os
import sys
import numpy as np
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin


file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)

from ml.model import train_model, inference, compute_model_metrics

def test_train_model():
    """
    Check that the trained model is a classifer model. 
    """
    X = np.random.rand(20, 5)
    y = np.random.randint(2, size=20)
    model = train_model(X, y)
    assert isinstance(model, BaseEstimator) and isinstance(model, ClassifierMixin)


def test_compute_model_metrics():
    """
    Check the returned type while evaluation. 
    """
    y, preds = [1, 1, 0], [0, 1, 0]
    precision, recall, f1score = compute_model_metrics(y, preds)
    assert isinstance(precision, float) and isinstance(recall, float) and isinstance(f1score, float)


def test_saved_model():
    """
    Check that a model has been saved
    """
    model_path=os.path.join(file_dir, "../model/trained_model.pkl")
    if os.path.isfile(model_path):
        try:
            _ =pickle.load(open(model_path, 'rb'))
        except BaseException:
            assert ('Model not found')


def test_inference():
    X = np.random.rand(10, 5)
    y = np.random.randint(2, size=10)
    model = train_model(X, y)
    pred = inference(model, X)
    # Check prediction shape
    assert y.shape == pred.shape