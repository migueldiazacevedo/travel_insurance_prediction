import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score


def model_assessment_series(predictions, observations, model_name="model"):
    """
    A pandas Series for assessment of classification model predictions.

    :param predictions: predictions from model
    :type predictions: array-like
    :param observations: observations from data
    :type observations: array-like
    :param model_name: a name for the model to use as a title for the column
    :type model_name: str

    :return: model assessment scores
    :rtype: pandas.Series
    """
    return pd.Series(
        [
            accuracy_score(predictions, observations),
            precision_score(predictions, observations),
            recall_score(predictions, observations),
            f1_score(predictions, observations),
        ],
        index=["Accuracy", "Precision", "Recall", "F1-Score"],
        name=model_name,
    )


def model_assessment_series_cv(model, X, y, cv=5, model_name="model"):
    """
    A pandas Series for assessment of classification model predictions using k-fold cross validation.

    :param model: the model being used
    :type model: scikit-learn model or similar
    :param X: features
    :type X: array-like
    :param y: outcomes
    :type y: array-like
    :param cv: number of folds
    :type cv: int
    :param model_name: a name for the model to use as a title for the column
    :type model_name: str

    :return: model assessment scores
    :rtype: pandas.Series
    """
    accuracy = np.mean(cross_val_score(model, X, y, cv=cv, scoring="accuracy"))
    precision = np.mean(cross_val_score(model, X, y, cv=cv, scoring="precision"))
    recall = np.mean(cross_val_score(model, X, y, cv=cv, scoring="recall"))
    f1 = np.mean(cross_val_score(model, X, y, cv=cv, scoring="f1"))
    return pd.Series(
        [accuracy, precision, recall, f1],
        index=["Accuracy", "Precision", "Recall", "F1-Score"],
        name=model_name,
    )
