"""
Unit tests for preprocess.py
"""

from src.preprocess import create_preprocessor
from sklearn.compose import ColumnTransformer


def test_create_preprocessor():

    preprocessor = create_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)