import argparse
import logging
import os
import dask

import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from config import config
from config.dataset_config import datasets
from processing_step import ProcessingStep
from utils import clear_path


class Interpolator(ProcessingStep):
    """
    Class for performing the interpolation preprocessing step, by linearly interpolating between messages so that
    they are at a regular interval.
    """
    def __init__(self, method):
        super().__init__()
        self._define_directories(
            from_name='cleaned' + ('_debug' if args.debug else ''),
            to_name='interpolated' + ('_debug' if args.debug else '')
        )
        self._initialize_logging(args.save_log, 'interpolate')

        self.method = method
        if self.method == '1d':
            self._interpolator = np.interp
        else:
            raise ValueError('Only 1d interpolation is currently supported')

        # Specify interpolation methods for different columns
        self.timestamp_column = ['base_datetime']
        self.columns_to_interpolate = [
            'lat',
            'lon',
            'sog',
            'cog',
            'heading'
        ]
        self.columns_to_use_most_recent = [
            # 'status',
            # 'draft',
            # 'cargo',
            # 'transceiver_class',
        ]
        self.stable_columns = [
            'mmsi',
            # 'vessel_type',
            # 'length',
            # 'width',
            'vessel_group',
        ]
        self.columns_to_calculate = {
            'year': 'int16',
            'month': 'byte'
        }

    def load(self):
        """
        Load the test, train, and validation sets

        This function just specifies the paths for dask. (Dask uses lazy evaluation so the full sets aren't read in
        here.)

        :return:
        """
        for dataset_name in ['test', 'train', 'valid']:
            dataset_path = os.path.join(self.from_dir, f'{dataset_name}.parquet')
            self.datasets[dataset_name] = dd.read_parquet(dataset_path)
            logging.info(f'{dataset_name} set starting with {self.datasets[dataset_name].shape[0].compute():,} messages')

        logging.debug('File paths have been specified for dask')

    def save(self):
        """
        Save the interpolated datasets to disk

        Because Dask uses lazy evaluation, the processing will actually happen only when this method is called.

        :return:
        """
        clear_path(self.to_dir)
        os.mkdir(self.to_dir)

        for dataset_name in ['test', 'train', 'valid']:
            out_path = os.path.join(self.to_dir, f'{dataset_name}.parquet')
            self.current_file = out_path
            dd.to_parquet(self.datasets[dataset_name + '_interpolated'], out_path, schema='infer')
            logging.info(f'{dataset_name} contains {self.datasets[dataset_name+ "_interpolated"].shape[0].compute():,} messages after interpolation')
            logging.debug(f'Interpolation complete for {dataset_name} set and dataset saved to {out_path}')
            self.current_file = None

    def interpolate(self):
        """
        Interpolate each of the datasets

        Also converts the timestamps to seconds

        Does not actually do the interpolation, as Dask uses lazy evaluation

        :return:
        """
        for dataset_name in ['test', 'train', 'valid']:
            self.datasets[dataset_name]['base_datetime'] = self.datasets[dataset_name]['base_datetime'].astype(
                int) / 10 ** 9

            out_meta = self.datasets[dataset_name].dtypes.append(pd.Series(self.columns_to_calculate.values(),
                                                                           index=self.columns_to_calculate.keys()))
            out_meta = [(i, z) for i, z in out_meta.items()]

            self.datasets[dataset_name + '_interpolated'] = self.datasets[dataset_name].map_partitions(
                self.interpolate_partition,
                meta=out_meta)

    def interpolate_partition(self, partition: pd.DataFrame):
        """
        Interpolate the messages in a single partition.

        Dask works by splitting up a DataFrame into multiple partitions, then spreading the partitions across multiple
        processes (or threads, if you configure it that way). The map_partitions dask method can be used to
        have each processor do a transformation of its partition. This is the transformation that we are applying. This
        method uses the pandas groupby().apply() method to interpolate each track using the interpolate_track method
        below.

        :param partition: Partition to interpolate
        :return: Interpolated partition
        """
        interpolated = partition.groupby('track').apply(self.interpolate_track)
        interpolated = interpolated.reset_index()
        interpolated = interpolated.drop('level_1', axis=1)
        interpolated = interpolated.set_index('track')
        return interpolated

    def interpolate_track(self, track: pd.DataFrame):
        """
        Interpolate the messages in a single track

        This is a function that is applied to each track, using Pandas' groupby().apply() method. The input is a
        Pandas DataFrame that has all the messages for a single track, and the output is a Pandas DataFrame for
        this track that has been interpolated.

        It keeps the first message in the trajectory, then resamples at every config.interpolation_time_gap seconds.

        :param track: Track to interpolate
        :return: Interpolated track
        """

        # Find the times that we want to sample at
        first_ts = track['base_datetime'].iloc[0]
        last_ts = track['base_datetime'].iloc[-1]
        times_to_sample = np.arange(first_ts, last_ts + 1, config.interpolation_time_gap)

        # Because categorical variables can't be interpolated linearly, we are instead taking the value from the most
        # recent timestamp. The categorical_interpolator object just finds the index of the most recent timestamp (in
        # the true data) for each time in times_to_sample, so that the categorical variables for this index can be found
        # more quickly
        categorical_interpolator = interp1d(track['base_datetime'],
                                            range(len(track['base_datetime'])),
                                            kind='previous', assume_sorted=True)
        most_recent_idx = categorical_interpolator(times_to_sample)

        interpolated = {}
        # iterate through columns and interpolate each one
        for col in track.columns:
            if col == 'base_datetime':
                interpolated[col] = times_to_sample
            elif col in self.columns_to_interpolate:
                # If this should be interpolated, do so
                interpolated[col] = self._interpolator(times_to_sample, track['base_datetime'], track[col])
            elif col in self.columns_to_use_most_recent:
                # If this column is categorical and can change, then use the most recent value
                interpolated[col] = [track[col].iloc[int(i)] for i in most_recent_idx]
            elif col in self.stable_columns:
                # If this column is categorical but should be stable over the whole dataset, just use the first value
                interpolated[col] = track[col].iloc[0]
            else:
                raise ValueError(f'Please specify how to handle column {col}')

        # Add year and month variables
        for col in self.columns_to_calculate.keys():
            if col == 'year':
                interpolated[col] = pd.to_datetime(interpolated['base_datetime'] * 10 ** 9).year
            elif col == 'month':
                interpolated[col] = pd.to_datetime(interpolated['base_datetime'] * 10 ** 9).month
            else:
                raise ValueError(f'Please specify how to interpolate column {col}')

        interpolated = pd.DataFrame(interpolated)
        return interpolated


if __name__ == '__main__':
    # Because interpolation is at five minutes, and most of the original timestamps are *more frequent* than five
    # minutes, this is really more of a downsampling than an interpolation
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_name', choices=datasets.keys())
    # Interpolation Method
    parser.add_argument('--method', type=str, default='1d', choices=['1d'],
                        help='Interpolation method')

    # Logging and debugging
    parser.add_argument('-l', '--log_level', type=int,
                        default=2, choices=[0, 1, 2, 3, 4],
                        help='Level of logging to use')
    parser.add_argument('-s', '--save_log', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    config.set_log_level(args.log_level)

    config.dataset_config = datasets[args.dataset_name]

    if args.debug:
        dask.config.set(scheduler='single-threaded')
    else:
        dask.config.set(scheduler='processes')


    interpolator = Interpolator(args.method)
    interpolator.load()
    interpolator.interpolate()
    interpolator.save()
