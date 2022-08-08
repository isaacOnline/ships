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

class SlidingWindow(ProcessingStep):
    """
    Class for performing the sliding window processing step, where long tracks are split up into multiple
    shorter tracks that can be used for fitting the model.
    """
    def __init__(self):
        super().__init__()
        self._define_directories(
            from_name='interpolated_with_currents_stride_3' + ('_debug' if args.debug else ''),
            to_name='windowed_with_currents_stride_3' + ('_debug' if args.debug else '')
        )
        self._initialize_logging(args.save_log, 'sliding_window_with_weather')


    def load(self):
        """
        Load the test, train, and validation sets

        This function just specifies the paths for dask. (Dask uses lazy evaluation so the full sets aren't read in
        here.)

        :return:
        """
        for dataset_name in ['test', 'valid', 'train']:
            dataset_path = os.path.join(self.from_dir, f'{dataset_name}.parquet')
            self.datasets[dataset_name] = dd.read_parquet(dataset_path)

            if self.datasets[dataset_name].index.name is None:
                def rename_index(partition):
                    partition.index.name = 'track'
                    return partition
                self.datasets[dataset_name] = self.datasets[dataset_name].map_partitions(rename_index)
        logging.info('File paths have been specified for dask')

    def save(self):
        """
        Save the windowed datasets to disk

        Because Dask uses lazy evaluation, the processing will actually happen only when this method is called.

        :return:
        """

        for dataset_name in self.datasets.keys():
            self.datasets[dataset_name] = self.datasets[dataset_name]
            out_path = os.path.join(self.to_dir, f'{dataset_name}.parquet')
            clear_path(out_path)
            logging.info(
                f'Number of messages in {dataset_name} set is {len(self.datasets[dataset_name]):,}')
            dd.to_parquet(self.datasets[dataset_name], out_path, schema='infer')
            logging.info(f'{dataset_name} set saved to {out_path}')

    def calculate(self):
        """
        Iterate through and perform the sliding window calculations.

        This creates two types of windowed datasets, which differ in the size of their sliding window movements. The
        long_term_train datasets are used for model training and validation - they use a shorter sliding window
        movement so that there will be a significant overlap between successive sequences in these datasets. The
        long_term_test datasets are used for performance evaluation, and for this reason their sliding window movement
        is set so that the input portions of successive sequences do not overlap at all.

        :return:
        """
        # Only the train/validation sets need long_term_train versions
        for dataset_name in ['train', 'valid']:
            out_meta = self.datasets[dataset_name].dtypes
            out_meta = [(i, z) for i, z in out_meta.items()]
            self.datasets[dataset_name + '_long_term_train'] = self.datasets[dataset_name].map_partitions(
                self.window_partition,
                self.window_track_long_term_train,
                meta=out_meta)
        # Only the test/validation sets need long_term_test versions
        for dataset_name in ['test', 'valid']:
            out_meta = self.datasets[dataset_name].dtypes
            out_meta = [(i, z) for i, z in out_meta.items()]
            self.datasets[dataset_name + '_long_term_test'] = self.datasets[dataset_name].map_partitions(
                self.window_partition,
                self.window_track_long_term_test,
                meta=out_meta)
        del self.datasets['test'], self.datasets['train'], self.datasets['valid']
        logging.info(f'Calculation methods have been defined for Dask')

    def window_partition(self, partition: pd.DataFrame, track_fn) -> pd.DataFrame:
        """
        Perform sliding window calculations on the messages in a single partition

        Dask works by splitting up a DataFrame into multiple partitions, then spreading the partitions across multiple
        processes (or threads, if you were to configure it that way). The map_partitions dask method can be used to
        have each processor do a transformation of its partition. This is the transformation that we are applying. This
        method uses the pandas groupby().apply() method to window each track using the track_fn function that is
        specified.

        :param partition: Partition to window
        :param track_fn: Function for windowing
        :return: Windowed partition
        """
        partition = partition.groupby('track').apply(track_fn).reset_index()
        partition = partition.drop('level_1', axis=1)
        partition = partition.set_index('track')
        return partition

    def window_track_long_term_test(self, track: pd.DataFrame) -> pd.DataFrame:
        """
        Create the long term dataset for a single track, for testing purposes

        Windows a long trajectory into multiple shorter ones. Even though we're still breaking up the trajectories using
        a sliding window, the input portions of the new trajectories should not overlap. e.g. if there's 6 hours of
        data, the first set will use (hour 1) to predict (hours 2, 3, 4, 5), and the second set will use (hour 2) to
        predict (hours 3, 4, 5, 6). This is different than 'window_track_long_term_train', as the training data is
        allowed to overlap

        :param track: Track to window
        :return: Windowed track
        """
        number_of_subtracks = len(track) - (config.length_of_history + config.length_into_the_future)
        subtrack_idxs = [np.arange(i, i + config.length_of_history + config.length_into_the_future + 1) for i in
                         range(0, number_of_subtracks, config.length_of_history)]
        windowed_track = [track.iloc[idxs] for idxs in subtrack_idxs]
        windowed_track = pd.concat(windowed_track).reset_index(drop=True)
        return windowed_track

    def window_track_long_term_train(self, track: pd.DataFrame) -> pd.DataFrame:
        """
        Create the long term dataset for a single track, for training purposes

        Windows a long trajectory into multiple shorter ones. The difference between this and
        window_track_long_term_test, is the sliding window size. The 'test' version makes sure that input sequences do
        not overlap, where as this version does include overlaps.

        :param track: Track to window
        :return: Windowed track
        """
        number_of_gaps_into_the_future = config.length_into_the_future
        number_of_subtracks = len(track) - (config.length_of_history + number_of_gaps_into_the_future)
        window_movement_in_ts = int(config.dataset_config.sliding_window_movement / config.interpolation_time_gap)
        subtrack_idxs = [np.arange(i, i + config.length_of_history + number_of_gaps_into_the_future + 1) for i in
                         range(0, number_of_subtracks, window_movement_in_ts)]
        windowed_track = [track.iloc[idxs] for idxs in subtrack_idxs]
        windowed_track = pd.concat(windowed_track).reset_index(drop=True)
        return windowed_track



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

    window = SlidingWindow()
    window.load()
    window.calculate()
    window.save()
