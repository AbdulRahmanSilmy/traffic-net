"""
This module contains classes for plotting the forecast against the real values. 
The forecast is plotted for each day of the week in a subplot. The number of 
days to plot in a row is determined dynamically based on the number of days 
in the forecast.

Classes
-------
- BaseForecastPlotter -> Abstract base class for forecast plotting
- ForecastPerWeekdayPlotter -> Forecast plotter that plots the forecast against 
the real values for a specific day of the week
- ForecastAllDayPlotter -> Forecast plotter that plots the forecast against the 
real values for all days of the week
"""
from typing import List, Tuple
from datetime import datetime
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod


_DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
_DEFAULT_S_DAY = 48
_DEFAULT_N_DAYS_IN_ROW = 10


class BaseForecastPlotter(ABC):
    """
    Base class for forecast plotting where the forecast is plotted against the real values.
    Subplot is created to aid in visualizing the forecast for multiple days. The number of 
    columns is always one. The number of rows is determined dynamically based on the number of
    days to plot in a row.

    Parameters
    ----------
    s_day : int, default=48
        Number of samples in a day

    n_days_in_row : int, default=10
        Number of days to plot in a row of the subplot
    """

    DAY_NAMES = _DAY_NAMES

    def __init__(self, s_day: int = _DEFAULT_S_DAY, n_days_in_row: int = _DEFAULT_N_DAYS_IN_ROW):
        self.s_day = s_day
        self.n_days_in_row = n_days_in_row

    @staticmethod
    @abstractmethod
    def _xtick_label(sample: int, day: int) -> str:
        """
        Returns the label for the x-axis tick. The label is based on the 
        sample number and the day of the week.

        Parameters
        ----------
        sample : int
            Sample number

        day : int
            Day of the week. Range is [0, 6] where 0 is Monday and 6 is Sunday.

        Returns
        -------
        str
            Label for the x-axis tick
        """
        raise NotImplementedError

    def _base_plot(self, true_values, forecast, day_of_week, dates=None) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Base plot function that creates the subplots and plots the forecast against the real values.

        Parameters
        ----------
        true_values : np.ndarray
            Array of true values. Shape is (n_samples,).

        forecast : np.ndarray
            Array of forecasted values. Shape is (n_samples,).

        day_of_week : np.ndarray
            Array of day of the week for each sample. Shape is (n_samples,).

        dates : List[datetime]
            List of dates for each sample. Shape is (n_samples,).

        Returns
        -------
        Tuple[plt.Figure, List[plt.Axes]]
            Tuple containing the figure and list of axe
        """
        n_days = true_values.shape[0] // self.s_day
        n_rows = int(np.ceil(n_days / self.n_days_in_row))

        fig, axes = plt.subplots(
            n_rows, 1, figsize=(15, 5 * n_rows), sharey=True)
        plt.subplots_adjust(top=0.94)
        max_val = max(np.nanmax(true_values), max(forecast))

        if n_rows == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.set_ylim([0, max_val + 5])
            start = i * self.n_days_in_row * self.s_day
            end = min((i + 1) * self.n_days_in_row *
                      self.s_day, true_values.shape[0])
            ax.plot(np.arange(start, end),
                    true_values[start:end], label='real')
            ax.plot(np.arange(start, end),
                    forecast[start:end], label='forecast')
            ax.legend(loc='upper left')

            for j in range(self.n_days_in_row + 1):
                ax.axvline(x=start + j * self.s_day, color='grey',
                           linestyle='--', alpha=0.5)
                if start + j * self.s_day >= end:
                    break
                day_true = true_values[start + j *
                                       self.s_day:start + (j + 1) * self.s_day]
                day_forecast = forecast[start + j *
                                        self.s_day:start + (j + 1) * self.s_day]
                is_forecast_absent = np.isnan(day_forecast).all()
                if is_forecast_absent:
                    mse_text = "mse: N/A"
                else:
                    mask = ~np.isnan(day_true)
                    mse = mean_squared_error(day_true[mask], day_forecast[mask])
                    mse_text = f"mse: {mse:.1f}"
            
                annotation_x = start + (j + 0.5) * self.s_day
                ax.annotate(mse_text,
                            xy=(annotation_x, max_val),
                            xytext=(annotation_x, max_val),
                            fontsize=10,
                            color='black',
                            ha='center')
                if dates is not None:
                    day_date = dates[start + j * self.s_day]
                    ax.annotate(f"{day_date}",
                                xy=(annotation_x, max_val + 2),
                                fontsize=10,
                                color='black',
                                ha='center')

            xticks = np.arange(start, end, self.s_day)
            row_days = day_of_week[start:end:self.s_day]
            xlabels = [self._xtick_label(i, day)
                       for i, day in zip(xticks, row_days)]
            ax.set_xticks(xticks, xlabels)
        return fig, axes

    @abstractmethod
    def plot(self, X: np.ndarray, forecast: np.ndarray, dates: List[datetime] = None) -> plt.Figure:
        """
        Plot the forecast against the real values.

        Parameters
        ----------
        X : np.ndarray
            Array of input features. Shape is (n_samples, 2). The first column is the true values, 
            the second column is the day of the week.

        forecast : np.ndarray
            Array of forecasted values. Shape is (n_samples,).

        dates : List[datetime], optional
            List of dates for each sample. Shape is (n_samples,).

        Returns
        -------
        plt.Figure
            Figure containing the plot
        """
        raise NotImplementedError


class ForecastPerWeekdayPlotter(BaseForecastPlotter):
    """
    Forecast plotter that plots the forecast against the real values for a specific day of the week.

    Parameters
    ----------
    day : int
        Day of the week to plot. Range is [0, 6] where 0 is Monday and 6 is Sunday.

    s_day : int, default=48
        Number of samples in a day

    n_days_in_row : int, default=10
        Number of days to plot in a row of the subplot
    """
    _TITLE_PREFIX = 'Forecast vs Real | '

    def __init__(self, day: int, s_day: int = _DEFAULT_S_DAY, n_days_in_row: int = _DEFAULT_N_DAYS_IN_ROW):
        super().__init__(s_day, n_days_in_row)
        self.day = day

    def _xtick_label(self, sample: int, day: int) -> str:
        """
        Returns the label for the x-axis tick. The label is based on the
        sample number and the day of the week.

        Parameters
        ----------
        sample : int
            Sample number

        day : int
            Day of the week. Range is [0, 6] where 0 is Monday and 6 is Sunday.

        Returns
        -------
        str
            Label for the x-axis tick
        """

        return f'{sample // self.s_day} day'

    def plot(self, X: np.ndarray, forecast: np.ndarray, dates: List[datetime] = None) -> plt.Figure:
        """
        Plot the forecast against the real values for the specific day of the week.

        Parameters
        ----------
        X : np.ndarray
            Array of input features. Shape is (n_samples, 2). The first column is the true values, 
            the second column is the day of the week.

        forecast : np.ndarray
            Array of forecasted values. Shape is (n_samples,).

        dates : List[datetime], optional
            List of dates for each sample. Shape is (n_samples,).

        Returns
        -------
        plt.Figure
            Figure containing the plot
        """
        true_values = X[:, 0]
        day_of_week = X[:, 1].astype(int)
        mask = day_of_week == self.day
        true_values = true_values[mask]
        forecast = forecast[mask]
        if dates is not None:
            dates = np.array(dates)[mask]

        fig, _ = self._base_plot(true_values, forecast, day_of_week, dates)

        fig.suptitle(f'{self._TITLE_PREFIX}{self.DAY_NAMES[self.day]}')

        return fig


class ForecastAllDayPlotter(BaseForecastPlotter):
    """
    Forecast plotter that plots the forecast against the real values for all days of the week.

    Parameters
    ----------
    s_day : int, default=48
        Number of samples in a day

    n_days_in_row : int, default=10
        Number of days to plot in a row of the subplot
    """
    _TITLE = 'Forecast vs Real'

    def _xtick_label(self, sample: int, day: int) -> str:
        """
        Returns the label for the x-axis tick. The label is based on the
        sample number and the day of the week.

        Parameters
        ----------
        sample : int
            Sample number

        day : int
            Day of the week. Range is [0, 6] where 0 is Monday and 6 is Sunday.

        Returns
        -------
        str
            Label for the x-axis tick
        """
        return f'{sample // self.s_day} | {self.DAY_NAMES[day]}'

    def plot(self, X: np.ndarray, forecast: np.ndarray, dates: List[datetime] = None) -> plt.Figure:
        """
        Plot the forecast against the real values for all days of the week.

        Parameters
        ----------
        X : np.ndarray
            Array of input features. Shape is (n_samples, 2). The first column is the true values, 
            the second column is the day of the week.

        forecast : np.ndarray
            Array of forecasted values. Shape is (n_samples,).

        dates : List[datetime], optional
            List of dates for each sample. Shape is (n_samples,).

        Returns
        -------
        plt.Figure
            Figure containing the plot
        """

        true_values = X[:, 0]
        day_of_week = X[:, 1].astype(int)

        fig, _ = self._base_plot(true_values, forecast, day_of_week, dates)

        fig.suptitle(self._TITLE)

        return fig
