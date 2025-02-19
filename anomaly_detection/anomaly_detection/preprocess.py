"""
This module contains functions to preprocess the time series data before fitting the models.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ImputeVals(BaseEstimator, TransformerMixin):
    """
    Impute missing values in the first column of X based on the day of the week and time of day

    Parameters
    ----------
    m : int, default=48
        The number of intervals in a day 
        (e.g., 48 for 30-minute intervals, 96 for 15-minute intervals)

    Attributes
    ----------
    avg_day_of_week : np.ndarray, shape (7, m)
        The average numeric value for each day of the week and time of day
    """

    def __init__(self, m=48):
        self.m = m
        self.avg_day_of_week = None

    def fit(self, X, y=None):
        """
        Fit the imputer by computing the average numeric value for each day and time of day

        Parameters
        ----------
        X : np.ndarray
            A 2D numpy array of shape (num_samples, 2). The first column is the target variable 
            and the second column is the day of the week. The day of the week ranges from [0,6] 
            representing days Monday to Sunday.

        y: None
            Ignored. This parameter exists only for compatibility with the scikit-learn API.

        Returns
        -------
        self : ImputeVals
            The fitted imputer
        """
        X = np.copy(X)

        X_numeric = X[:, 0].reshape(-1, self.m)
        X_day_of_week = X[:, 1].reshape(-1, self.m)
        X_day_of_week = X_day_of_week[:, 0]

        avg_day_of_week = np.empty((7, self.m))

        for i in range(7):
            mask = X_day_of_week == i
            X_week_day = X_numeric[mask]
            # masked array to handle missing values
            X_week_day_ma = np.ma.array(X_week_day, mask=np.isnan(X_week_day))
            avg_day_of_week[i, :] = np.ma.average(X_week_day_ma, axis=0).data

        self.avg_day_of_week = avg_day_of_week

        return self

    def transform(self, X, y=None):
        """
        Transform the input by filling the missing values in the first column of X

        Parameters
        ----------
        X : np.ndarray
            A 2D numpy array of shape (num_samples, 2). The first column is the target variable 
            and the second column is the day of the week. The day of the week ranges from [0,6] 
            representing days Monday to Sunday.

        y: None
            Ignored. This parameter exists only for compatibility with the scikit-learn API.

        Returns
        -------
        X : np.ndarray
            A 2D numpy array of shape (num_samples, 2). The first column is the target variable 
            and the second column is the day of the week. The day of the week ranges from [0,6] 
            representing days Monday to Sunday. 
        """

        X = np.copy(X)
        X_numeric = X[:, 0].reshape(-1, self.m)
        X_day_of_week = X[:, 1].reshape(-1, self.m)
        X_day_of_week = X_day_of_week[:, 0].astype(int)

        empty_mask = np.isnan(X_numeric)
        average_days = np.vstack([self.avg_day_of_week[i]
                                 for i in X_day_of_week])
        X_numeric[empty_mask] = 0
        X_numeric = X_numeric + empty_mask*average_days

        X[:, 0] = X_numeric.flatten()

        return X
