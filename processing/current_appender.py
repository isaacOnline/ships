import argparse
import logging
import os
import dask

import dask.dataframe as dd
import numpy as np
import pandas as pd

from config import config
from config.dataset_config import datasets
from processing_step import ProcessingStep
from utils import clear_path

class WeatherAppender(ProcessingStep):
    """
    This class is used for joining the interpolated AIS messages to the weather dataset
    """
    def __init__(self):
        super().__init__()
        self._define_directories(
            from_name='interpolated_with_destination' + ('_debug' if args.debug else ''),
            to_name='interpolated_with_currents_stride_3' + ('_debug' if args.debug else '')
        )
        self.weather_dir = os.path.join(self.box_and_year_dir, 'ocean_currents_aggregated')
        self._initialize_logging(args.save_log, 'add_currents')

        self.unneeded_columns = [
            'mmsi',
            'heading',
        ]

        logging.info(f'Not using columns {self.unneeded_columns}')

    def load(self):
        """
        Load the test, train, and validation sets, as well as the weather data

        This function just specifies the test/train/valid paths for dask. (Dask uses lazy evaluation so the full sets aren't read in
        here.) The weather data is loaded with pandas.

        :return:
        """
        for dataset_name in ['test', 'valid', 'train']:
            dataset_path = os.path.join(self.from_dir, f'{dataset_name}.parquet')
            self.datasets[dataset_name] = dd.read_parquet(dataset_path)
            for col in self.unneeded_columns:
                if col in self.datasets[dataset_name].columns:
                    self.datasets[dataset_name] = self.datasets[dataset_name].drop(columns=col, axis=1)
            if args.debug:
                self.datasets[dataset_name] = self.datasets[dataset_name].partitions[:1]

        if config.currents_window != 'stable':
            raise ValueError('Moving currents windows have been deprecated')
        else:
            self.datasets['weather'] = pd.read_csv(os.path.join(self.weather_dir,
                                                                'weather_aggregated.csv'))
            self.datasets['weather'] = self.datasets['weather'].set_index(['year'])

            logging.info('File paths have been specified for dask')

    def _change_data_sizes(self, partition, features):
        """
        Go through and change the sizes of columns.

        :param partition: Dataset to change sizes of
        :param features: Second dataset specifying column names and desired datatypes for dataset
        :return:
        """
        for col, dtype in features.iteritems():
            if str(partition[col].dtype) != dtype:
                partition[col] = partition[col].astype(dtype)
        return partition


    def save(self):
        """
        Save the joined dataset to disk

        Because Dask uses lazy evaluation, the processing will actually happen only when this method is called.

        Also changes data types so as to save space.

        :return:
        """

        for dataset_name in ['test','valid','train']:
            out_path = os.path.join(self.to_dir, f'{dataset_name}.parquet')
            clear_path(out_path)

            features = self.datasets[dataset_name].dtypes.astype(str)
            size = 32
            for col in features.index:
                original_type = features[col]
                if col == 'base_datetime':
                    new_type = 'float64'
                elif 'water_u' in col or 'water_v' in col:
                    new_type = 'int16'
                elif features[col] == 'float64':
                    new_type = f'float{size}'
                elif features[col] == 'int64':
                    new_type = f'int{size}'
                elif features[col] == 'Sparse[bool, False]':
                    new_type = 'bool'
                else:
                    new_type = original_type

                features[col] = new_type

            partition = self.datasets[dataset_name]._partitions(0).compute()
            output_meta = self._change_data_sizes(partition, features)
            self.datasets[dataset_name] = self.datasets[dataset_name].map_partitions(self._change_data_sizes, features,
                                                                                    meta=output_meta)
            dd.to_parquet(self.datasets[dataset_name], out_path, schema='infer')
            dataset_len = len(dd.read_parquet(out_path))
            logging.info(
                f'{dataset_name} dataset has {dataset_len:,} records after joining to weather data')


            logging.info(f'{dataset_name} set saved to {out_path}')


    def calculate(self):
        """
        Join each AIS message to the most recent forecasted ocean currents

        :return:
        """
        if config.currents_window != 'stable':
            raise ValueError('Moving currents windows have been deprecated')
        else:
            # Iterate through train/test/valid
            for dataset_name in ['train','test','valid']:
                dataset_len = len(self.datasets[dataset_name])
                logging.info(f'Length of {dataset_name} set is {dataset_len:,} before merging with weather')

                # Get hour/day where message occurred (year and month are already calculated)
                # (hour is rounded down to every 3rd hour, as the ocean current forecasts only occur every 3 hours, on
                # the hour)
                self.datasets[dataset_name]['hour'] = dd.to_datetime(self.datasets[dataset_name]['base_datetime'], unit='s').dt.hour // 3 * 3
                self.datasets[dataset_name]['day'] = dd.to_datetime(self.datasets[dataset_name]['base_datetime'], unit='s').dt.day
                idx = self.datasets[dataset_name].index

                # Join to weather data
                self.datasets[dataset_name] = dd.merge(self.datasets[dataset_name],
                                                  self.datasets['weather'],
                                                  left_on =['year','month','day','hour'],
                                                  right_on=['year','month','day','hour'],
                                                  how='left')
                self.datasets[dataset_name].index = idx
                self.datasets[dataset_name] = self.datasets[dataset_name].drop(columns=['hour','day'])

                # If this is the training set, calculate the mean u/v values for each lat/lon location, so we can use
                # these values for imputation
                if dataset_name == 'train':
                    means = self.datasets[dataset_name].mean(axis=0).round().astype(np.int16).compute()

                # Perform the mean imputation
                nas = self.datasets[dataset_name].isna().sum(axis=0).compute()
                na_cols = []
                for col in nas.index:
                    na_count = nas[col]
                    if na_count != 0:
                        na_cols.append(col)
                        logging.info(f'{dataset_name} has {na_count:,} NA values in column {col} '
                                     f'({na_count / dataset_len * 100:0.3}%). These values have been imputed with '
                                     f'{means[col]}.')
                    else:
                        logging.info(f'{dataset_name} has {na_count:,} NA values in column {col} '
                                     f'({na_count / dataset_len * 100:0.3}%).')

                means_dict = {c: means[c] for c in means.index if c in na_cols}

                # We only need a single "weather_is_imputed" col here, as imputation happens for all the columns at once
                self.datasets[dataset_name]['weather_is_imputed'] = self.datasets[dataset_name][na_cols].isna().any(axis=1)

                self.datasets[dataset_name] = self.datasets[dataset_name].fillna(means_dict)



            if len(self.datasets['train'].columns) != len(self.datasets['test'].columns):
                raise ValueError(
                    'There was an error in preprocessing and the train and test sets have differing numbers of'
                    'columns. This is likely due to the NA filling code in buoy_appender.py, which should '
                    'be edited to account for your use case.')
            if len(self.datasets['train'].columns) != len(self.datasets['valid'].columns):
                raise ValueError(
                    'There was an error in preprocessing and the train and valid sets have differing numbers of'
                    'columns. This is likely due to the NA filling code in buoy_appender.py, which should '
                    'be edited to account for your use case.')




        if len(self.datasets['train'].columns) != len(self.datasets['test'].columns):
            raise ValueError('There was an error in preprocessing and the train and test sets have differing numbers of'
                             'columns.')
        if len(self.datasets['train'].columns) != len(self.datasets['valid'].columns):
            raise ValueError('There was an error in preprocessing and the train and valid sets have differing numbers of'
                             'columns.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_name', choices=datasets.keys())

    # Logging and debugging
    parser.add_argument('-l', '--log_level', type=int,
                        default=2, choices=[0, 1, 2, 3, 4],
                        help='Level of logging to use')
    parser.add_argument('-s', '--save_log', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    config.dataset_config = datasets[args.dataset_name]
    config.set_log_level(args.log_level)

    if args.debug:
        dask.config.set(scheduler='single-threaded')
    else:
        dask.config.set(scheduler='single-threaded')

    appender = WeatherAppender()
    appender.load()
    appender.calculate()
    appender.save()
