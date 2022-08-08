import argparse
import logging
import os
import dask
import gc

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml.preprocessing import OneHotEncoder

from config import config
from config.dataset_config import datasets
from processing_step import ProcessingStep
from utils import clear_path


class Formatter(ProcessingStep):
    """
    Class for performing the final processing step, whereby the data is saved as numpy arrays which can easily be read
    in for training and evaluation.
    """
    def __init__(self):
        super().__init__()
        self._define_directories(
            from_name='windowed_with_currents_stride_3' + ('_debug' if args.debug else ''),
            to_name='formatted_with_currents_stride_3' + ('_debug' if args.debug else '')
        )
        self._initialize_logging(args.save_log, 'format_with_weather_and_time')

        logging.info(f'categorical_columns used are {config.categorical_columns}')
        self.dataset_names = [
            'train_long_term_train',
            'test_long_term_test',
            'valid_long_term_train',
            'valid_long_term_test'
        ]

    def load(self):
        """
        Load the test, train, and validation sets

        This function will also repartition the datasets to make the partition sizes manegable

        :return:
        """
        for dataset_name in self.dataset_names:
            dataset_path = os.path.join(self.from_dir, f'{dataset_name}.parquet')
            self.datasets[dataset_name] = dd.read_parquet(dataset_path)
            tmp_dir = os.path.join(self.from_dir, f'.tmp_{dataset_name}.parquet')
            if not os.path.exists(tmp_dir):
                self._repartition_ds(dataset_name)
            self.datasets[dataset_name] = dd.read_parquet(tmp_dir)

        logging.info('File paths have been specified for dask')

    def save(self):
        """
        Save the test, train, and validation sets

        Because of the potential size of the dataset, this will by default only process one partition at a time. The
        partitions will need to be combined later. To change this behavior, you can set conserve_memory to False

        :return:
        """
        conserve_memory = True

        clear_path(self.to_dir)
        os.mkdir(self.to_dir)

        # Write features to disk
        self.features.to_csv(os.path.join(self.to_dir, 'features.csv'))

        for dataset_name in self.dataset_names:

            # Find path that we want to write data to
            set_dir = os.path.join(self.to_dir, dataset_name)
            os.mkdir(set_dir)

            # If we don't need to conserve memory, then just compute the entire dataset at once
            if not conserve_memory:
                self.datasets[dataset_name] = self.datasets[dataset_name].compute()

            # Iterate through the time gaps
            for new_time_gap in config.time_gaps:
                # Find the indexes of the timestamps we want to select in the input sequence
                x_idxs = np.arange(0,
                                   config.length_of_history,
                                   new_time_gap / config.interpolation_time_gap, dtype=int)

                # Find the indexes of the timestamps we want to select in the output sequence
                if 'long_term' in dataset_name:
                    timesteps_into_the_future = config.length_of_history + config.length_into_the_future + 1
                else:
                    raise ValueError('Unknown Dataset')
                next_gap = config.length_of_history + new_time_gap / config.interpolation_time_gap - 1
                y_idxs = np.arange(next_gap,
                                   timesteps_into_the_future,
                                   new_time_gap / config.interpolation_time_gap, dtype=int)

                num_minutes = int(new_time_gap / 60)

                if conserve_memory:
                    # If we are conserving memory, then iterate through the partitions, calculating one at a time
                    # and saving the calculated value to disk
                    npart = self.datasets[dataset_name].npartitions
                    x_dir = os.path.join(set_dir, f'{num_minutes}_min_time_gap_x')
                    os.mkdir(x_dir)
                    x_len = 0
                    for i in range(npart):
                        x_path = os.path.join(set_dir, f'{num_minutes}_min_time_gap_x', f'{i}.npy')
                        data = self.datasets[dataset_name].partitions[i].compute()[:, x_idxs, :]
                        gc.collect()
                        np.save(x_path, data)
                        x_len += len(data)
                        x_shape = list(data.shape)
                        del data
                        gc.collect()
                    x_shape[0] = x_len
                    logging.info(f'For {dataset_name}, the shape of {num_minutes}_min_time_gap_x is {x_shape}')

                else:
                    # Otherwise, just select the indexes we need and save to disk (since the entire dataset has already
                    # been computed)
                    x_path = os.path.join(set_dir, f'{num_minutes}_min_time_gap_x.npy')
                    X = self.datasets[dataset_name][:, x_idxs, :]
                    np.save(x_path, X)
                    logging.info(f'For {dataset_name}, the shape of {num_minutes}_min_time_gap_x.npy is {X.shape}')
                    del X

                if conserve_memory:
                    # If we need to conserve memory, then just processes and save partitions one at a time
                    y_dir = os.path.join(set_dir, f'{num_minutes}_min_time_gap_y')
                    os.mkdir(y_dir)
                    y_len = 0
                    for i in range(npart):
                        y_path = os.path.join(set_dir, f'{num_minutes}_min_time_gap_y',f'{i}.npy')
                        data = self.datasets[dataset_name].partitions[i].compute()[:, y_idxs, :]
                        gc.collect()
                        y_len += len(data)
                        np.save(y_path, data)
                        y_shape = list(data.shape)
                        del data
                        gc.collect()
                    y_shape[0] = y_len
                    logging.info(f'For {dataset_name}, the shape of {num_minutes}_min_time_gap_y is {y_shape}')
                else:
                    # Otherwise, select the correct indexes from the precalculated data and save to a single file
                    y_path = os.path.join(set_dir, f'{num_minutes}_min_time_gap_y.npy')
                    Y = self.datasets[dataset_name][:, y_idxs, :]
                    np.save(y_path, Y)
                    logging.info(f'For {dataset_name}, the shape of {num_minutes}_min_time_gap_y.npy is {Y.shape}')
                    del Y

            logging.info(f'{dataset_name} set saved to directory {set_dir}')
            del self.datasets[dataset_name]
        self._clear_tmp_files()

    def _reshape_partition(self, partition: pd.DataFrame, into_the_future):
        """
        Reshape partition so that it can be easily used by Keras

        Create a 3D array for Keras to use for training the model

        :param partition:
        :param into_the_future:
        :return:
        """
        partition = partition.to_numpy()
        num_ts = config.length_of_history + into_the_future
        partition = partition[[range(idx, idx + num_ts) for idx in range(0, len(partition), num_ts)]]
        partition = np.stack(partition)
        return partition

    def _one_hot(self, dataset_name):
        """
        One hot encode categorical variables

        :param dataset_name: Whether this is the training, testing, or validation set
        :return: The one hot encoder object used for the transformation
        """
        # Convert dtypes to category with dask
        for col in config.categorical_columns:
            self.datasets[dataset_name][col] = self.datasets[dataset_name][col].astype('category')

        # Identify the category values
        self.datasets[dataset_name] = self.datasets[dataset_name].categorize()

        # Based on the category values, fit a onehotencoder object
        encoder = OneHotEncoder()
        encoder = encoder.fit(self.datasets[dataset_name][config.categorical_columns])

        # Define a constant order for the categories
        if not hasattr(self, 'column_order'):
            self.column_order = []
            for col, levels in zip(config.categorical_columns, encoder.categories_):
                for level in levels:
                    self.column_order += [f'{col}_{level}']

        # Transform the dataset using the one hot encoder, and place columns in correct order
        transformed = encoder.transform(self.datasets[dataset_name][config.categorical_columns])
        transformed = transformed[self.column_order].astype('bool')

        # Log the categories
        for col in config.categorical_columns:
            categories = self.datasets[dataset_name][col].cat.categories.to_list()
            logging.info(f'Using {len(categories)} values for column {col} in {dataset_name} set: {categories}')

        # Add one hot encode versions to dataset
        self.datasets[dataset_name] = self.datasets[dataset_name].drop(config.categorical_columns, axis=1)
        self.datasets[dataset_name] = dd.concat([self.datasets[dataset_name], transformed], axis=1)
        return encoder

    def calculate(self):
        """
        Iterate through the datasets, formatting them correctly for training and testing

        Also performs one hot encoding and transformation to floats.

        :return:
        """
        for dataset_name in self.dataset_names:
            self._one_hot(dataset_name)

            self.features = self.datasets[dataset_name].dtypes.astype(str)

            self.features = self.features.replace('Sparse[bool, False]', 'bool')
            self.features.name = 'dtype'
            self.features.index.name = 'column'

            if 'long_term' in dataset_name:
                timesteps_into_the_future = config.length_into_the_future + 1
            else:
                raise ValueError('Unknown Dataset')


            out_meta = [(i, z) for i, z in self.datasets[dataset_name].dtypes.items()]
            self.datasets[dataset_name] = self.datasets[dataset_name].map_partitions(self._reshape_partition,
                                                                                     timesteps_into_the_future,
                                                                                     meta=out_meta)
        logging.info(f'Calculation methods have been defined for Dask')

    def _repartition_ds(self, dataset_name):
        """
        Change partition sizes to make sure each one can fit in memory at once

        :param dataset_name: Dataset to use
        :return:
        """
        dataset_path = os.path.join(self.from_dir, f'{dataset_name}.parquet')

        tmp_dir = os.path.join(self.from_dir, f'.tmp_{dataset_name}.parquet')

        MAX_PARTITION_SIZE = '4000MB'

        self.datasets[dataset_name] = self.datasets[dataset_name].repartition(partition_size=MAX_PARTITION_SIZE,
                                                                              force=True)
        divisions = [self.datasets[dataset_name].partitions[i].index.min().compute() for i in
                     range(self.datasets[dataset_name].npartitions)]
        divisions += [self.datasets[dataset_name].partitions[
                          self.datasets[dataset_name].npartitions - 1].index.max().compute() + 1]

        # The above doesn't create create clean breaks on track, (tracks will be spread across different
        # partitions), which messes up the mapping that comes next. The below corrects this.
        self.datasets[dataset_name] = dd.read_parquet(dataset_path)
        self.datasets[dataset_name] = self.datasets[dataset_name].repartition(
            divisions=divisions,
            force=True)
        dd.to_parquet(self.datasets[dataset_name], tmp_dir, schema='infer')
        del self.datasets[dataset_name]

    def _clear_tmp_files(self):
        """
        Once dataset has been saved, clear any temporary files that are on disk

        :return:
        """
        for dataset_name in self.dataset_names:
            tmp_dir = os.path.join(self.from_dir, f'.tmp_{dataset_name}.parquet')
            clear_path(tmp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_name', choices=datasets.keys())
    # Logging
    parser.add_argument('-l', '--log_level', type=int,
                        default=2, choices=[0, 1, 2, 3, 4],
                        help='Level of logging to use')
    parser.add_argument('-s', '--save_log', action='store_true')
    parser.add_argument('--debug', action='store_true')


    args = parser.parse_args()
    config.set_log_level(args.log_level)
    config.dataset_config = datasets[args.dataset_name]

    dask.config.set(scheduler='single-threaded')


    formatter = Formatter()
    formatter.load()
    formatter.calculate()
    formatter.save()
