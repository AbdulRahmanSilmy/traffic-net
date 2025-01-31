"""
This module contains custom time series models that can be used in the time series pipeline.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error


class AverageModel(BaseEstimator, RegressorMixin):
    """
    A scikit-learn wrapper for predicting the future values using a rolling forecast approach. 
    The seasonality is assumed to be defined by the unique days of the week. 
    This is meant to be used as a baseline model for time series forecasting.

    The model fits a rolling average of the past values for each unique day of the week and time of day.

    Parameters
    ----------
    m : int, default=48
        The number of periods in each season. Defines the number of intervals in a day.

    window_size : int, default=2
        The number of seasons/days to consider in the rolling average.

    error_tol : float, default=5
        The error tolerance to update the rolling average buffer. 
        If the mean squared error between the predicted and actual values
        is less than the error tolerance, the buffer is updated with the actual values.


    Attributes
    ----------
    buffers: dict
        A dictionary containing the rolling average buffer for each unique day of the week.
        The dictionary is structured as follows:
        {
            0: np.array of shape (m*window_size,) containing the rolling average buffer for Monday,
            1: np.array of shape (m*window_size,) containing the rolling average buffer for Tuesday,
            ...
            6: np.array of shape (m*window_size,) containing the rolling average buffer for Sunday,
        }
    """

    def __init__(self, m=48, window_size=2, error_tol=5):
        self.m = m
        self.window_size = window_size
        self.error_tol = error_tol
        self.buffers = None

    def _extract_window_days(self, X_numeric, X_days, day):
        """ Extract the last window_size days for a given day of the week"""
        mask = X_days == day
        X_day_of_week = X_numeric[mask]

        X_tail = X_day_of_week[-self.window_size*self.m:]
        return X_tail

    def _average_buffer(self, numeric):
        """
        Compute the average of the rolling average buffer for a given day of the week
        across different times of the day.
        """
        average = np.average(numeric.reshape(-1, self.m), axis=0)
        return average

    def fit(self, X, y=None):
        """
        Fit the rolling average model by computing the rolling average buffer for each unique day of the week.

        Parameters
        ----------
        X : np.ndarray
            A 2D numpy array of shape (num_samples, 2). The first column is the target variable 
            and the second column is the day of the week. The day of the week ranges from [0,6] 
            representing days Monday to Sunday.

        y : None
            Ignored.

        Returns
        -------
        self : AverageModel
            The fitted AverageModel
        """
        X_numeric = X[:, 0]
        X_days = X[:, 1]

        unique_days = np.unique(X_days)

        self.buffers = {day: self._extract_window_days(
            X_numeric, X_days, day) for day in unique_days}

        return self

    def predict(self, X):
        """
        Predict the future values using the rolling average buffer for each unique day of the week.

        Parameters
        ----------
        X : np.ndarray
            A 2D numpy array of shape (num_samples, 2). The first column is the target variable 
            and the second column is the day of the week. The day of the week ranges from [0,6] 
            representing days Monday to Sunday.

        Parameters
        ----------
        forecast : np.ndarray
            A 1D numpy array of shape (num_samples,) containing the predicted values.
        """
        numeric = X[:, 0].reshape(-1, self.m)
        day_of_week = X[:, 1].reshape(-1, self.m)

        forecast = np.empty_like(X[:, 0])

        for i, (numeric_day, day) in enumerate(zip(numeric, day_of_week)):
            day = int(day[0])
            day_buffer = self.buffers[day]
            day_forecast = self._average_buffer(day_buffer)
            forecast[i*self.m:(i+1)*self.m] = day_forecast

            # Update the buffer if the error is less than the tolerance
            error = mean_squared_error(numeric_day, day_forecast)
            if error < self.error_tol:
                self.buffers[day] = np.concatenate(
                    [day_buffer[-self.m:], numeric_day])

        return forecast
