"""
This module contains the pipeline for the anomaly detection system. The pipeline
consists of the following processes:
    1. MinSamplesPerDay: Check if there are enough samples per day
    2. TemporalAggregation: Temporally aggregate the data
    3. TimeSeriesForecast: Forecast the number of cars using a rolling average model
    4. AnomalyDetection: Detect anomalies in the forecasted data
    5. PlotCSVHandler: Write the plot csv file

The pipeline is started from the previous date stored in the temporary configuration file.
The pipeline is run until the most recent complete day. The entire pipeline can be run 
using the AnomalyDetectionPipeline class.

Currently the pipeline is designed to work with one day of data at a time. Future versions
of the pipeline will be extended to run on a more frequent basis.
"""
from abc import ABC, abstractmethod
import os
import json
import pickle
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

_TEMPORAL_AGGREGATION_COLUMNS = ['num_cars', 'incoming', 'outgoing']


class PipelineProcess(ABC):
    """
    This is an abstract class for the pipeline processes. The class has two abstract
    methods: reset and run. The reset method is used to reset the process to the initial
    state. The run method is used to perform the process.
    """

    @staticmethod
    def get_kwarg(arg_str, kwargs):
        """
        Get the argument from the kwargs dictionary. If the argument is not
        present, raise a ValueError.

        Parameters
        ----------
        arg_str : str
            The argument name

        kwargs : dict
            A dictionary of keyword arguments

        Returns
        -------
        arg : object
            The argument value
        """
        arg = kwargs.get(arg_str, None)
        if arg is None:
            raise ValueError(f"{arg_str} is required")

        return arg

    @abstractmethod
    def reset(self):
        """
        Perform any reset operations for the process. 
        """

    @abstractmethod
    def run(self, **kwargs):
        """
        Perform the process operation. Add to or modify the kwargs 
        dictionary and return it at the end of the function. The returned 
        dictionary will be passed to the next process in the pipeline.

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments

        Returns
        -------
        kwargs: dict
            A dictionary of keyword arguments
        """


class MinSamplesPerDay(PipelineProcess):
    """
    Check if there are enough samples per day

    Parameters
    ----------
    min_samples_per_day : int
        Minimum number of samples to consider a day

    """

    def __init__(self, min_samples_per_day):
        self.min_samples_per_day = min_samples_per_day

    def reset(self):
        """
        Required method for the abstract class
        """

    def run(self, **kwargs):
        """
        Check if there are enough samples per day

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments
            Required key:
                df_day_data : pd.DataFrame
                    A DataFrame of the daily data

        Returns
        -------
        kwargs : dict
            A dictionary of keyword arguments
            New key:
                enough_data : bool
                    Boolean indicating if there is enough data to forecast
        """
        df_day_data = self.get_kwarg('df_day_data', kwargs)

        if df_day_data.shape[0] < self.min_samples_per_day:
            enough_data = False
        else:
            enough_data = True

        kwargs.update({'enough_data': enough_data})

        return kwargs


class TemporalAggregation(PipelineProcess):
    """
    Temporally aggregate the data. Use the mean of the data for the aggregation.
    Pandas resample method is used to aggregate the data.

    Parameters
    ----------
    sample_period : str, default='30min'
        Period to temporally aggregate the data

    index_name : str, default='time'
        Name of the index column

    columns : list, default=['num_cars', 'incoming', 'outgoing']
        List of column names to aggregate
    """

    def __init__(self, sample_period='30min', index_name='time', columns=None):
        self.sample_period = sample_period
        self.index_name = index_name
        if columns is None:
            self.columns = _TEMPORAL_AGGREGATION_COLUMNS

    def reset(self):
        """
        Required method for the abstract class
        """

    def run(self, **kwargs):
        """
        Temporally aggregate the data. 

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments
            Required key:
                df_day_data : pd.DataFrame
                    A DataFrame of the daily data

        Returns
        -------
        kwargs : dict
            A dictionary of keyword arguments
            New key:
                df_agg : pd.DataFrame
                    A DataFrame of the aggregated
        """
        df_day_data = self.get_kwarg('df_day_data', kwargs)

        df_agg = df_day_data[self.columns].resample(self.sample_period).mean()
        full_index = pd.date_range(
            start=df_agg.index.min().replace(hour=0, minute=0, second=0),
            end=df_agg.index.max().replace(hour=23, minute=30, second=0),
            freq=self.sample_period
        )
        df_agg = df_agg.reindex(full_index)
        df_agg.index.name = self.index_name

        df_agg['day_of_week'] = df_agg.index.dayofweek.values

        kwargs.update({'df_agg': df_agg})

        return kwargs


class TimeSeriesForecast(PipelineProcess):
    """
    Forecast the number of cars using a rolling average model. The model is trained
    on the number of cars and the day of the week.

    Parameters
    ----------
    model_path : str
        Path to save the model
    """

    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self) -> object:
        """ 
        Get the model from the model path 

        Returns
        -------
        model : object
            A model object
        """
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def save_model(self, model):
        """ 
        Save the model to the model path

        Parameters
        ----------
        model : object
            A model object
        """
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)

    def reset(self):
        """ Reset the model to the initial state after training """
        model = self.load_model()
        model.reset()
        self.save_model(model)

    def run(self, **kwargs) -> dict:
        """
        Forecast the number of cars using a rolling average model. The model is trained
        on the number of cars and the day of the week. Forecast is generated only when
        there is enough data.

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments
            Required keys:
                df_agg : pd.DataFrame
                    A DataFrame of the aggregated data
                enough_data : bool
                    Boolean indicating if there is enough data to forecast

        Returns
        -------
        kwargs : dict
            A dictionary of keyword arguments
            Updated keys:
                df_agg : pd.DataFrame
                    A DataFrame of the aggregated data with the forecast
        """
        df_agg = self.get_kwarg('df_agg', kwargs)
        enough_data = self.get_kwarg('enough_data', kwargs)

        if enough_data:
            X = df_agg[['num_cars', 'day_of_week']].values
            model = self.load_model()
            df_agg['forecast'] = model.predict(X)
            self.save_model(model)
        else:
            df_agg['forecast'] = np.nan

        kwargs.update({'df_agg': df_agg})

        return kwargs


class AnomalyDetection(PipelineProcess):
    """
    Detect anomalies in the forecasted data using the mean squared error.

    Parameters
    ----------
    max_mse : float, default=30
        Maximum mean squared error to be considered as an non-anomaly day
    """

    def __init__(self, max_mse=30):
        self.max_mse = max_mse

    def reset(self):
        """
        Required method for the abstract
        """

    def run(self, **kwargs) -> dict:
        """
        Detect anomalies in the forecasted data using the mean squared error.

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments
            Required keys:
                enough_data : bool
                    Boolean indicating if there is enough data to forecast
                df_agg : pd.DataFrame
                    A DataFrame of the aggregated data

        Returns
        -------
        kwargs : dict
            A dictionary of keyword arguments
            Updated keys:
                anomaly_detect : str
                    A string indicating if an anomaly is detected

                df_agg : pd.DataFrame
                    A DataFrame of the aggregated data with the anomaly
                    status
        """
        enough_data = self.get_kwarg('enough_data', kwargs)
        df_agg = self.get_kwarg('df_agg', kwargs)
        if enough_data:

            df_metric = df_agg.dropna(subset=['num_cars'])

            mse = mean_squared_error(
                df_metric['num_cars'], df_metric['forecast'])
            if mse > self.max_mse:
                anomaly_detect = 'Anomaly Detected'
            else:
                anomaly_detect = 'No Anomaly'

        else:
            anomaly_detect = 'Not enough data'

        df_agg['anomaly_detect'] = anomaly_detect
        kwargs.update({'anomaly_detect': anomaly_detect,
                       'df_agg': df_agg})

        return kwargs


class PlotCSVHandler(PipelineProcess):
    """
    Writing the plot csv file
    """

    def __init__(self, plot_csv_path):
        self.plot_csv_path = plot_csv_path

    def reset(self):
        """
        Delete the plot csv file if it exists
        """
        if os.path.exists(self.plot_csv_path):
            os.remove(self.plot_csv_path)

    def run(self, **kwargs) -> dict:
        """
        Write the plot csv file

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments
            Required key:
                df_agg : pd.DataFrame
                    A DataFrame of the aggregated data

        Returns
        -------
        kwargs : dict
            A dictionary of keyword arguments
        """
        df_agg = self.get_kwarg('df_agg', kwargs)

        if os.path.exists(self.plot_csv_path):
            df_old = pd.read_csv(self.plot_csv_path, index_col='time')
            df_old.index = pd.to_datetime(df_old.index)
            df_new = pd.concat([df_old, df_agg])
            df_new.to_csv(self.plot_csv_path, index=True)
        else:
            df_agg.to_csv(self.plot_csv_path, index=True)

        return kwargs


class AnomalyDetectionPipeline():
    """
    A pipeline to generate the plot csv file for the traffic data.

    The day is processed one day at a time. The pipeline is started from the previous date
    stored in the temporary configuration file. The pipeline is run until the most recent complete
    day.

    This pipeline consists of the following processes:
        1. MinSamplesPerDay: Check if there are enough samples per day
        2. TemporalAggregation: Temporally aggregate the data
        3. TimeSeriesForecast: Forecast the number of cars using a rolling average model
        4. AnomalyDetection: Detect anomalies in the forecasted data
        5. PlotCSVHandler: Write the plot csv file

    Parameters
    ----------
    min_samples_per_day : int
        Minimum number of samples to consider a day

    sample_period : str
        Period to temporally aggregate the data

    max_mse : float
        Maximum mean squared error to be considered as an non-anomaly day

    average_model_path : str
        Path to save the average model

    plot_csv_path : str
        Path to save the plot csv file

    temp_config_path : str
        Path to save the temporary configuration file. This file is used to
        store the previous processed date to start the pipeline.

    previous_date : str
        Previous date to start the pipeline

    tabular_csv_path : str
        Path to the tabular csv file
    """

    def __init__(self,
                 min_samples_per_day,
                 sample_period,
                 max_mse,
                 average_model_path,
                 plot_csv_path,
                 temp_config_path,
                 previous_date,
                 tabular_csv_path):

        self.min_samples_per_day = min_samples_per_day
        self.sample_period = sample_period
        self.max_mse = max_mse
        self.average_model_path = average_model_path
        self.plot_csv_path = plot_csv_path
        self.temp_config_path = temp_config_path
        self.previous_date = previous_date
        self.tabular_csv_path = tabular_csv_path
        self.processes = [
            MinSamplesPerDay(self.min_samples_per_day),
            TemporalAggregation(self.sample_period),
            TimeSeriesForecast(self.average_model_path),
            AnomalyDetection(self.max_mse),
            PlotCSVHandler(self.plot_csv_path)
        ]

    def _load_temp_config(self):
        """
        Load the temporary configuration file. If the file does not exist, create
        an empty dictionary.

        Returns
        -------
        temp_config : dict
            A dictionary of the temporary configuration
        """
        if os.path.exists(self.temp_config_path):
            with open(self.temp_config_path, 'r', encoding="utf-8") as f:
                temp_config = json.load(f)
        else:
            root_dir = os.path.dirname(self.temp_config_path)
            os.makedirs(root_dir, exist_ok=True)
            temp_config = {}

        return temp_config

    def _run_process(self, kwargs):
        """
        Run the pipeline processes

        Parameters
        ----------
        kwargs : dict
            The final dictionary of keyword arguments
        """
        for process in self.processes:
            kwargs = process.run(**kwargs)

        return kwargs

    def _reset_process(self):
        """
        Reset the pipeline processes
        """
        for process in self.processes:
            process.reset()

    def _save_temp_config(self, temp_config):
        """
        Save the temporary configuration file
        """
        with open(self.temp_config_path, 'w', encoding="utf-8") as f:
            json.dump(temp_config, f)

    def reset_pipeline(self):
        """
        Reset the pipeline
        """
        self._reset_process()
        temp_config = self._load_temp_config()
        temp_config['previous_date'] = self.previous_date
        self._save_temp_config(temp_config)

    def generate_plot_csv(self):
        """
        Generate the plot csv file for the traffic data
        """

        temp_config = self._load_temp_config()

        previous_date = temp_config.get('previous_date', self.previous_date)
        previous_date = pd.to_datetime(previous_date)

        df = pd.read_csv(self.tabular_csv_path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

        current_time = dt.datetime.now()

        df_sub = df[df.index > previous_date]
        unique_dates = np.unique(df_sub.index.date)
        for date_obj in unique_dates:
            datetime_obj = dt.datetime.combine(
                date_obj, dt.datetime.min.time())
            datetime_obj = datetime_obj.replace(hour=23, minute=59, second=59)

            if current_time > datetime_obj:

                df_day_data = df_sub[df_sub.index.date == date_obj]
                kwargs = {'df_day_data': df_day_data}
                self._run_process(kwargs)
                previous_date = datetime_obj

                temp_config['previous_date'] = previous_date.strftime(
                    '%Y-%m-%d %H:%M:%S')
                self._save_temp_config(temp_config)
