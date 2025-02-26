"""
This module contains custom time series models that can be used in the time series pipeline.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import pmdarima as pm


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
        
        self._is_fitted = True

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
        check_is_fitted(self)
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
    
    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


class PMDARIMAWrapper(BaseEstimator, RegressorMixin):
    """
    A wrapper for PMDARIMA's auto_arima model to fit and predict time series data with daily seasonality.
    This wrapper fits a unique ARIMA model for each day of the week.

    Parameters
    ----------
    seasonal : bool, optional (default=True)
        Whether to fit a seasonal ARIMA model.
    m : int, optional (default=48)
        The number of periods in each season.
    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for model fitting.
    
    Attributes
    ----------
    seasonal : bool
        Whether to fit a seasonal SARIMA model.
    m : int
        The number of periods in each season.
    n_jobs : int
        The number of jobs to run in parallel for model fitting.
    model_dict : dict
        A dictionary where keys are unique days and values are fitted SARIMA models.

    Methods
    -------
    fit(X, y=None)
        Fit the SARIMA models to the training data.
    predict(X)
        Predict the future values using the fitted SARIMA models.
    """
    def __init__(self, seasonal=True, m=48, n_jobs=1):
        self.seasonal = seasonal
        self.m = m
        self.n_jobs = n_jobs
        self.model_dict = {}
    
    def _train_day(self, day):
        mask = self.train_days==day
        y_day = self.train_numeric[mask]
        model = pm.auto_arima(y_day, seasonal=self.seasonal, m=self.m)
        return model


    def fit(self, X, y=None):
        """
        Fit the SARIMA models to the training data.

        Parameters
        ----------
        X : np.ndarray
            A 2D numpy array where the first column is the target variable and the second column is the day of the week.
            The day of the week should be an integer between 0 and 6 where 0 is Monday and 6 is Sunday.

        y : None
            Ignored.

        Returns
        -------
        self : PMDARIMAWrapper
            The fitted PMDARIMAWrapper model.
        """
        self.train_numeric = X[:,0]
        self.train_days = X[:,1]

        unique_days = np.unique(self.train_days)

        results = Parallel(n_jobs=self.n_jobs)(delayed(self._train_day)(day) for day in unique_days)

        self.model_dict = {day: model for day, model in zip(unique_days, results)}

        return self
    
    def predict(self, X):
        """
        Predict the future values using the fitted SARIMA models.

        Parameters
        ----------
        X : np.ndarray
            A 2D numpy array where the first column is the target variable which is ignored
            and the second column is the day of the week.
            The day of the week should be an integer between 0 and 6 where 0 is Monday and 6 is Sunday. 
        """
        numeric = X[:,0]
        y_pred = np.zeros(len(numeric))
        days = X[:,1]
        unique_days, counts = np.unique(days, return_counts=True)

        for day, count in zip(unique_days, counts):
            mask = days==day
            count = int(count)
            forecast = self.model_dict[day].predict(count)
            y_pred[mask] = forecast
        
        return y_pred