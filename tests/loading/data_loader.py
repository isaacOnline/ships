import os
import gc
import json

import pandas as pd
import numpy as np
from mlflow import log_artifact

from utils.utils import clear_path
from loading import loading
from loading.normalizer import Normalizer
from loading.disk_array import DiskArray

class ProcessingError(Exception):
    """
    Error that occurwed due to preprocessing
    """
    pass

class RunConfig():
    """
    Class for storing information about the final preprocessing steps used for preparing a
    dataset for model fitting.
    """
    def __init__(self):
        self.data = {}

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def load_from_dir(self, path):
        """
        Load a run config from disk

        :param path: Path to load from
        :return:
        """
        other_path = os.path.join(path, 'other.json')
        with open(other_path, 'r') as infile:
            data = json.load(infile)

        for k, v in data.items():
            if k == 'normalization_factors' or k not in self.data.keys() or self.data[k] is None:
                self[k] = v

        columns_path = os.path.join(path, 'columns.csv')
        self.data['columns'] = pd.read_csv(columns_path)

    def save_to_dir(self, path, register_with_mlflow=False):
        """
        Save a run config to disk

        :param path: Path to save to
        :param register_with_mlflow: Whether to register the saved artifact with mlflow
        :return:
        """
        clear_path(path)
        os.mkdir(path)

        # We don't want to store this is in the run config file, as we'll be storing multiple hours out together in the
        # same directory, so it has to be loaded in from the args
        hours_out = self.data.pop('hours_out')

        columns = self.data.pop('columns')
        columns_path = os.path.join(path, 'columns.csv')
        columns.to_csv(columns_path, index=False)

        other_path = os.path.join(path, 'other.json')
        with open(other_path, 'w') as outfile:
            json.dump(self.data, outfile)
        if register_with_mlflow:
            log_artifact(path)

        self.data['columns'] = columns
        self.data['hours_out'] = hours_out


class DataLoader():
    """
    Class for performing the final preprocessing steps to get a dataset ready for training and evaluation
    """
    def __init__(self, config, args, run_config_path=None, hard_reload=False, conserve_memory=False):
        self.config = config
        self.normalizer = Normalizer()
        self.conserve_memory = conserve_memory

        self.run_config = RunConfig()

        keys_to_save = [
            'time', 'length_of_history', 'model_type',
            'hours_out', 'time_of_day', 'sog_cog',
            'distance_traveled', 'weather', 'debug',
            'extended_recurrent_idxs', 'fusion_layer_structure',
            'destination'
        ]
        for k in keys_to_save:
            if hasattr(args, k):
                self.run_config[k] = getattr(args, k)
            else:
                self.run_config[k] = None


        self.run_config['formatted_dir'] = os.path.join(self.config.data_directory, f'{self.config.dataset_config.lat_1}_{self.config.dataset_config.lat_2}_'
                                     f'{self.config.dataset_config.lon_1}_{self.config.dataset_config.lon_2}_'
                                     f'{self.config.start_year}_{self.config.end_year}',
                                     self.config.dataset_name)
        self.run_config['dataset_name'] = config.dataset_name

        self.run_config['normalization_factors'] = None

        if hard_reload:
            self._create_run_config()
        elif run_config_path is not None:
            self._load_run_config_from_cache(run_config_path)
        elif not self._cached_config_exists():
            self._create_run_config()
        else:
            self._load_run_config_from_cache()

    def _load_run_config_from_cache(self, run_config_path=None):
        """
        Load a run_config from cached location

        This should only be called if a run_config exists in the cache, which can be checked using
        _cached_config_exists()

        :param run_config_path: Cached to load run config from
        :return:
        """
        if run_config_path is None:
            run_config_path = os.path.join(self._get_cache_dir('x'), 'run_config')
        self.run_config.load_from_dir(run_config_path)

        self._amend_hours_out_transformation()

    def _amend_y_timestamp_selection(self, transformations, timestamps_to_take):
        """
        Select a different y timestamp than the one stored in the cached run config

        A single run config may be used with multiple different hours out values (i.e. one
        run config may correspond to datasets for predicting 1 hours out/2 hours out/3 hours out).

        The run config stores information about selecting the y timestamp (which is based on the hours out value),
        so this simply updates the run config so that it matches with the correct hours out

        :param transformations: The list of transformations that are being made
        :param timestamps_to_take: Index of the y timestamps
        :return:
        """
        timestamps_to_take = timestamps_to_take if type(timestamps_to_take) == list else [timestamps_to_take]

        transformation_idx = [i for i, t in enumerate(transformations) if
                              t['function'] == 'select_timestamps' and t['dataset'] == ['y']][0]

        transformations[transformation_idx] = {'dataset': ['y'],
                                               'function': 'select_timestamps',
                                               'to_select': timestamps_to_take}
        return transformations


    def _amend_hours_out_transformation(self):
        """
        Wrapper for _amend_y_timestamp_selection

        A single run config may be used with multiple different hours out values (i.e. one
        run config may correspond to datasets for predicting 1 hours out/2 hours out/3 hours out).

        The run config stores information about selecting the y timestamp (which is based on the hours out value),
        so this simply updates the run config so that it matches with the correct hours out

        :return:
        """
        if self.run_config['model_type'] in ['long_term', 'long_term_fusion']:
            percent_of_timestamps_to_take = self.config.interpolation_time_gap / (self.run_config['time'] * 60)
            number_of_timestamps = int((self.config.length_into_the_future + 1) * percent_of_timestamps_to_take)
            total_number_of_hours = int(self.config.interpolation_time_gap * (self.config.length_into_the_future + 1)
                                        / (60 ** 2))
            y_timestamp_to_take = int(self.run_config['hours_out'] / total_number_of_hours * number_of_timestamps) - 1
        elif self.run_config['model_type'] == 'iterative':
            y_timestamp_to_take = 0
        else:
            return

        self.run_config['transformations'] = self._amend_y_timestamp_selection(self.run_config['transformations'],
                                                                               y_timestamp_to_take)

    def _cached_config_exists(self):
        """
        Check whether a cached run config exists

        If a cached run config exists, one will not need to be recreated

        :return:
        """
        cache_dir = self._get_cache_dir('x')
        config_dir = os.path.join(cache_dir, 'run_config')
        return os.path.exists(config_dir)

    def _create_run_config(self):
        """
        Create a run config from scratch

        :return:
        """
        self._create_columns_df()
        self._find_shapes()
        self._identify_transformations()

        self._calculate_normalization_factors()
        self._save_run_config_to_cache()

    def load_set(self, time_period, sliding_window_method, x_or_y, hard_reload = False, for_analysis=False):
        """
        Load a dataset, based on the run config

        :param time_period: Whether this is the training, validation, or test set
        :param sliding_window_method: What sliding window method was used for the dataset. Relevant as the validation
                                      set was processed with both the train and test sliding window method. (The
                                      training set was processed with the training sliding window method, while the
                                      test set was processed with the test sliding window method.)
        :param x_or_y: Whether this is an input or output set. Should be either 'x' or 'y'
        :param hard_reload: If True, will recreate the dataset from scratch, even if a cached version exists
        :param for_analysis: Whether the dataset is being loaded for post-hoc analysis. If True, a set of summary
                             statistics will be calculated for each trajectory (e.g. the mean bearing angle, the mean
                             magnitude for nearby surface current vectors, etc.). If False, then the normal
                             training/evaluation data sets will be loaded, which can be used to train and evaluate
                              keras models.
        :return:
        """
        if time_period not in ['train','valid','test']:
            raise ValueError('time_period must be one of "train", "valid", "test"]')
        if sliding_window_method not in ['train','test']:
            raise ValueError('sliding_window_method must be one of "train", "test"]')

        if for_analysis:
            return self._load_data_for_analysis(time_period, sliding_window_method, x_or_y)
        else:
            if hard_reload or not self._cached_ds_exists(time_period, sliding_window_method, x_or_y):
                return self._hard_reload_data(time_period, sliding_window_method, x_or_y)
            else:
                return self._load_ds_from_cache(time_period, sliding_window_method, x_or_y)


    def _get_cache_dir(self, x_or_y):
        """
        Find the directory to save/load the cached dataset to/from

        :param x_or_y: One of 'x' or 'y'
        :return:
        """
        base_cache_dir = os.path.join(
            os.path.dirname(self.run_config['formatted_dir']),
            f'.{self.run_config["dataset_name"]}_{self.run_config["model_type"]}_{self.run_config["debug"]}_'
            f'{self.run_config["sog_cog"]}_{self.run_config["time_of_day"]}_{self.run_config["time"]}_'
            f'{self.run_config["distance_traveled"]}_{self.run_config["weather"]}_'
            f'{self.run_config["extended_recurrent_idxs"]}_{self.run_config["fusion_layer_structure"]}_'
            f'{self.run_config["length_of_history"]}'
        )
        if self.run_config["destination"] != 'ignore':
            base_cache_dir += '_' + self.run_config["destination"]
        if x_or_y == 'x':
            return base_cache_dir
        elif self.run_config["hours_out"] is None:
            return base_cache_dir
        else:
            return os.path.join(base_cache_dir, f'{self.run_config["hours_out"]}')

    def _save_run_config_to_cache(self):
        """
        Store a recently-calculated run config object to disk, so it can be easily reloaded later

        :return:
        """
        if not os.path.exists(self._get_cache_dir('x')):
            os.mkdir(self._get_cache_dir('x'))
        if not os.path.exists(self._get_cache_dir('y')):
            os.mkdir(self._get_cache_dir('y'))
        self.run_config.save_to_dir(os.path.join(self._get_cache_dir('x'), 'run_config'))

    def _save_ds_to_cache(self, time_period, sliding_window_method, x_or_y):
        """
        Store a recently-calculated dataset to disk, so it can be easily reloaded later

        :param time_period: Whether this is the training, validation, or test set
        :param sliding_window_method: What sliding window method was used for the dataset. Relevant as the validation
                                      set was processed with both the train and test sliding window method. (The
                                      training set was processed with the training sliding window method, while the
                                      test set was processed with the test sliding window method.)
        :param x_or_y: One of 'x' or 'y'
        :return:
        """
        if not os.path.exists(self._get_cache_dir('x')):
            os.mkdir(self._get_cache_dir('x'))
        if not os.path.exists(self._get_cache_dir('y')):
            os.mkdir(self._get_cache_dir('y'))

        dir = self._get_cache_dir(x_or_y)

        if type(self.dataset) == list:
            dir = os.path.join(dir, f'{time_period}_{sliding_window_method}_{x_or_y}')
            clear_path(dir)
            os.mkdir(dir)
            for i in range(len(self.dataset)):
                path = os.path.join(dir, f'{i}.npy')
                np.save(path, self.dataset[i])
        elif type(self.dataset) == DiskArray:
            dir = os.path.join(dir, f'{time_period}_{sliding_window_method}_{x_or_y}')
            self.dataset.save_to_disk(dir)
        else:
            path = os.path.join(dir, f'{time_period}_{sliding_window_method}_{x_or_y}.npy')
            clear_path(path)
            np.save(path, self.dataset)


    def _load_ds_from_cache(self, time_period, sliding_window_method, x_or_y):
        """
        Load a dataset from disk, if it has been created previously

        :param time_period: Whether this is the training, validation, or test set
        :param sliding_window_method: What sliding window method was used for the dataset. Relevant as the validation
                                      set was processed with both the train and test sliding window method. (The
                                      training set was processed with the training sliding window method, while the
                                      test set was processed with the test sliding window method.)
        :param x_or_y: One of 'x' or 'y'
        :return:
        """
        dir = self._get_cache_dir(x_or_y)
        path1 = os.path.join(dir, f'{time_period}_{sliding_window_method}_{x_or_y}')
        path2 = os.path.join(dir, f'{time_period}_{sliding_window_method}_{x_or_y}_disk_array')
        if os.path.isdir(path1):
            names = os.listdir(path1)
            # Sort
            names = np.array(names)[np.argsort([int(n.split('.')[0]) for n in names])].tolist()
            dataset = [np.load(os.path.join(path1, n)) for n in names]
        elif os.path.isdir(path2):
            dataset = DiskArray()
            dataset = dataset.load_from_disk(path2)
        else:
            path3 = os.path.join(dir, f'{time_period}_{sliding_window_method}_{x_or_y}.npy')
            dataset = np.load(path3)

        return dataset

    def _cached_ds_exists(self, time_period, sliding_window_method, x_or_y):
        """
        Check whether a version of the dataset already exists on disk, so that we don't need to recalculate it

        :param time_period: Whether this is the training, validation, or test set
        :param sliding_window_method: What sliding window method was used for the dataset. Relevant as the validation
                                      set was processed with both the train and test sliding window method. (The
                                      training set was processed with the training sliding window method, while the
                                      test set was processed with the test sliding window method.)
        :param x_or_y: One of 'x' or 'y'
        :return:
        """
        dir = self._get_cache_dir(x_or_y)
        # The dataset can either be saved as a directory or a single file, so this just checks for both
        path1 = os.path.join(dir, f'{time_period}_{sliding_window_method}_{x_or_y}')
        path2 = os.path.join(dir, f'{time_period}_{sliding_window_method}_{x_or_y}_disk_array')
        path3 = os.path.join(dir, f'{time_period}_{sliding_window_method}_{x_or_y}.npy')
        return os.path.exists(path1) or os.path.exists(path2) or os.path.exists(path3)


    def _identify_transformations(self):
        """
        Create the list of steps that will need to be performed to prepare the dataset for training/evaluation

        Goes through and creates a step-by-step list of transformaitons that need to be made to the dataset, including
        any supplementary information. An example step might be normalization, where the supplmentary information is
        the scaling factors that will need to be used for each column. Another example step might be the deletion of
        a column, where the supplementary information would be the index of the column to delete

        The list is created to ensure that the same exact set of transformations can be applied to each of the
        test/train/validation sets, and so that a record of the preprocessing steps that were used can be stored with
        each model that is fit.

        :return:
        """
        self.run_config['transformations'] = []

        # Check if we only want to take the most recent timestamps (for example if the dataset on disk
        # contains 3 hours in the X set, but we only want to make predictions using 1 hour of history)
        desired_num_x_timestamps = int(self.run_config['length_of_history'] * 60 / self.run_config['time'] + 1)
        current_num_x_timestamps = self.run_config['original_x_shape'][1]
        if desired_num_x_timestamps < current_num_x_timestamps:
            x_timestamps_to_keep = list(range(current_num_x_timestamps)[-desired_num_x_timestamps:])
            self.run_config['transformations'] += [{'dataset':['x'], 'function':'select_timestamps',
                                                    'to_select':x_timestamps_to_keep}]
        elif desired_num_x_timestamps > current_num_x_timestamps:
            raise ProcessingError('There is not enough data in the formatted dataset to accommodate the number of '
                                  'hours desired for prediction. Please reprocess the data.')

        self.run_config['input_ts_length'] = desired_num_x_timestamps

        # Check if we only want to keep a single timestamp from Y (If this is a model that only predicts 1 timestamp)
        if self.run_config['model_type'] in ['long_term', 'long_term_fusion']:
            percent_of_timestamps_to_take = self.config.interpolation_time_gap / (self.run_config['time'] * 60)
            number_of_timestamps = int((self.config.length_into_the_future + 1) * percent_of_timestamps_to_take)
            total_number_of_hours = int(self.config.interpolation_time_gap * (self.config.length_into_the_future + 1)
                                        / (60 ** 2))
            y_timestamp_to_take = int(self.run_config['hours_out'] / total_number_of_hours * number_of_timestamps) - 1
            self.run_config['transformations'] += [{'dataset': ['y'],
                                                    'function':'select_timestamps',
                                                    'to_select': [y_timestamp_to_take]}]
        elif self.run_config['model_type'] == 'iterative':
            y_timestamp_to_take = 0
            self.run_config['transformations'] += [{'dataset': ['y'],
                                                    'function':'select_timestamps',
                                                    'to_select': [y_timestamp_to_take]}]


        # Check if we need to add hour/day columns
        if self.run_config['time_of_day'] == 'hour_day':
            base_datetime_idx = loading._find_current_col_idx('base_datetime', self.run_config['columns'])
            self.run_config['columns'], self.run_config['transformations'] = self._identify_cols_to_add(
                self.run_config['columns'], self.run_config['transformations'],
                ['hour', 'day_of_week'],'int16',
                extra_info ={'base_datetime_idx': base_datetime_idx})

        # Delete the base_datetime column
        self.run_config['columns'], self.run_config['transformations'] = self._identify_cols_to_delete(
            self.run_config['columns'], self.run_config['transformations'], ['base_datetime'])

        # Check if we need to transform sog/cog columns
        if self.run_config['sog_cog'] in ['min_median_max', 'median']:
            if self.run_config['sog_cog'] == 'min_median_max':
                columns_to_add = ['sog_min','sog_median','sog_max','cog_min','cog_median','cog_max']
            elif self.run_config['sog_cog'] == 'median':
                columns_to_add = ['sog_median','cog_median']
            extra_info = {'sog_index':loading._find_current_col_idx('sog', self.run_config['columns']),
                          'cog_index':loading._find_current_col_idx('cog', self.run_config['columns'])}
            self.run_config['columns'], self.run_config['transformations'] = self._identify_cols_to_add(
                self.run_config['columns'], self.run_config['transformations'],
                columns_to_add, 'float32', extra_info)

            self.run_config['columns'], self.run_config['transformations'] = self._identify_cols_to_delete(
                self.run_config['columns'], self.run_config['transformations'], ['sog', 'cog'])

        elif self.run_config['sog_cog'] == 'ignore':
            self.run_config['columns'], self.run_config['transformations'] = self._identify_cols_to_delete(
                self.run_config['columns'], self.run_config['transformations'], ['sog', 'cog'])

        # Check if we need to add distance traveled
        if self.run_config['distance_traveled'] == 'use':
            extra_info = {'lat_lon_idxs':
                          [loading._find_current_col_idx('lat',self.run_config['columns']),
                           loading._find_current_col_idx('lon',self.run_config['columns'])]}
            self.run_config['columns'], self.run_config['transformations'] = self._identify_cols_to_add(
                self.run_config['columns'], self.run_config['transformations'],
                ['distance_traveled'], 'float32', extra_info
            )

        # Check if we need to remove weather data
        if self.run_config['weather'] == 'ignore':
            weather_columns = self.run_config['columns']['column'][self.run_config['columns']['column_group'] == 'weather']
            self.run_config['columns'], self.run_config['transformations'] = self._identify_cols_to_delete(
                self.run_config['columns'], self.run_config['transformations'], weather_columns.tolist())

        if self.run_config['destination'] == 'ignore':
            destination_columns = self.run_config['columns']['column'][self.run_config['columns']['column_group'] == 'destination_cluster']
            self.run_config['columns'], self.run_config['transformations'] = self._identify_cols_to_delete(
                self.run_config['columns'], self.run_config['transformations'], destination_columns.tolist())
        elif self.run_config['destination'] == 'ohe':
            destination_center_columns = ['destination_cluster_lat_center','destination_cluster_lon_center']
            self.run_config['columns'], self.run_config['transformations'] = self._identify_cols_to_delete(
                self.run_config['columns'], self.run_config['transformations'], destination_center_columns)
        elif self.run_config['destination'] == 'cluster_centers':
            ohe_destination_columns = self.run_config['columns']['column'][
                (self.run_config['columns']['column_group'] == 'destination_cluster')
                & (self.run_config['columns']['dtype'] == 'bool')
            ]
            self.run_config['columns'], self.run_config['transformations'] = self._identify_cols_to_delete(
                self.run_config['columns'], self.run_config['transformations'], ohe_destination_columns.tolist())

        # Add that we'll need to normalize
        self.run_config['transformations'] += [{'dataset':['x', 'y'], 'function': 'normalize'}]

        # If this is a fusion model, sort out which features will be processed by the recurrent portion
        # and which by the other portion
        if self.run_config['model_type'] == 'long_term_fusion':
            if self.run_config['extended_recurrent_idxs'] == 'all_non_weather':
                recurrent_cols = self.run_config['columns']['column'][
                    (self.run_config['columns']['column_group'] != 'weather')
                    & self.run_config['columns']['being_used']
                ].tolist()
            else:
                if self.run_config['sog_cog'] == 'raw':
                    recurrent_cols = ['lat', 'lon', 'sog', 'cog']
                else:
                    recurrent_cols = ['lat', 'lon']
                if self.run_config['extended_recurrent_idxs'] == 'vt_dst_and_time':
                    vt_dst_columns = self.run_config['columns']['column'][
                        ((self.run_config['columns']['column_group'] == 'vessel_group')
                        | (self.run_config['columns']['column_group'] == 'destination_cluster'))
                         & self.run_config['columns']['being_used']
                    ].tolist()
                    recurrent_cols += vt_dst_columns
                    if self.run_config['time_of_day'] == 'hour_day':
                        recurrent_cols += ['hour','day_of_week']
            self.run_config['recurrent_idxs'] = [loading._find_current_col_idx(c, self.run_config['columns']) for c in recurrent_cols]
            if len(self.run_config['recurrent_idxs']) != self.run_config['columns']['being_used'].sum():
                # Check if there are any non recurrent columns
                self.run_config['transformations'] += [{'dataset':['x'], 'function':'split_for_fusion',
                                                        'columns':recurrent_cols,
                                                        'indexes':self.run_config['recurrent_idxs']}]


                # Check if we need to change the shape of the weather data for a CNN
                if self.run_config['fusion_layer_structure'] == 'convolutions':
                    weather_columns = self.run_config['columns']['column'][
                        (self.run_config['columns']['column_group'] == 'weather')
                        ]
                    weather_idxs = [loading._find_current_col_idx(c, self.run_config['columns']) for c in weather_columns]
                    self.run_config['transformations'] += [{'dataset': ['x'], 'function': 'reshape_weather',
                                                            'columns': weather_columns.to_list(), 'indexes':weather_idxs}]


        # Slice to just the y cols
        if self.run_config['sog_cog'] == 'raw' and self.run_config['model_type'] == 'iterative':
            y_cols = ['lat','lon','sog','cog']
        else:
            y_cols = ['lat','lon']
        self.run_config['y_idxs'] = [loading._find_current_col_idx(c, self.run_config['columns']) for c in y_cols]

        if self.run_config['model_type'] in ('attention_seq2seq','iterative','long_term','long_term_fusion'):
            self.run_config['transformations'] += [{'dataset':['y'], 'function':'select_columns',
                                                    'indexes':self.run_config['y_idxs']}]

            self.run_config['transformations'] += [{'dataset':['y'], 'function': 'squeeze'}]


    def _identify_cols_to_add(self, columns, transformations, cols, dtype, extra_info=None):
        """
        A helper function for _identify_transformations

        Used for creating the step that will exist in the transformation list

        :param columns: DataFrame specifying current columns
        :param transformations: The current list of transformations, which will be appended to
        :param cols: The names of columns to add
        :param dtype: The datatypes of columns to add
        :param extra_info: Any supplementary information that needs to be included for the step to be completed
        :return:
        """
        columns = pd.concat([
            columns,
            pd.DataFrame({
                'column': cols,
                'dtype': dtype,
                'being_used': True
            })]).reset_index(drop=True)
        idxs = [loading._find_current_col_idx(c, columns) for c in cols]
        t = {'dataset':['x', 'y'], 'function': 'add_columns', 'columns': cols, 'indexes': idxs}
        if extra_info is not None:
            for k, v in extra_info.items():
                t[k] = v
        transformations += [t]
        return columns, transformations

    def _identify_cols_to_delete(self, all_columns, transformations, cols_to_delete):
        """
        A helper function for _identify_transformations

        Used for creating the step that will exist in the transformation list

        :param all_columns: DataFrame specifying current columns
        :param transformations: The current list of transformations, which will be appended to
        :param cols_to_delete: The names of columns to delete
        :return:
        """
        columns_to_delete = [loading._find_current_col_idx(c, all_columns) for c in cols_to_delete]
        transformations += [{'dataset':['x', 'y'], 'function':'remove_columns',
                                                'columns': cols_to_delete, 'indexes':columns_to_delete}]
        all_columns['being_used'] = np.where(all_columns['column'].isin(cols_to_delete),
                                                            False,
                                                            all_columns['being_used'])
        return all_columns, transformations

    def _apply_transformations(self, x_or_y, transformations):
        """
        Method for applying a list of transformations

        :param x_or_y: One of 'x' or 'y'
        :param transformations: List of transformations to apply
        :return:
        """
        if self.conserve_memory:
            self.dataset._apply_transformations(
                x_or_y, transformations,
                self.normalizer, self.run_config['normalization_factors']
            )
        else:
            self.dataset = loading.apply_transformations(
                self.dataset,
                x_or_y, transformations,
                self.normalizer, self.run_config['normalization_factors']
            )

    def _calculate_normalization_factors(self):
        """
        Calculate the minimum/maximum of each column, which will be used to normalize each data point to a 0/1 scale

        :return:
        """
        normalization_step = np.where([t['function'] == 'normalize' for t in self.run_config['transformations']])[0][0]
        # Only make transformations up until the normalization is supposed to happen
        transformations = self.run_config['transformations'][:normalization_step]

        data_dir = os.path.join(self.run_config['formatted_dir'],
                                f'train_long_term_train')
        self.dataset = loading.read_ts_data(data_dir, self.run_config['time'], 'x', dtype='float32', conserve_memory=self.conserve_memory)
        self._apply_transformations('x', transformations)
        self.run_config['normalization_factors'] = self.normalizer.get_normalization_factors(self.dataset, self.run_config['columns'])
        del self.dataset
        gc.collect()


    def _hard_reload_data(self, time_period, sliding_window_method, x_or_y):
        """
        Load a dataset from scratch.

        Only used for loading the training/evaluation sets (not for post-hoc analysis).

        :param time_period: Whether this is the training, validation, or test set
        :param sliding_window_method: What sliding window method was used for the dataset. Relevant as the validation
                                      set was processed with both the train and test sliding window method. (The
                                      training set was processed with the training sliding window method, while the
                                      test set was processed with the test sliding window method.)
        :param x_or_y: One of 'x' or 'y'
        :return:
        """
        data_dir = os.path.join(self.run_config['formatted_dir'],
                                f'{time_period}_long_term_{sliding_window_method}')

        self.dataset = loading.read_ts_data(data_dir, self.run_config['time'], x_or_y, dtype='float32', conserve_memory=self.conserve_memory)
        transformations = self.run_config['transformations']

        # If this is the test set for the iterative model, we want *ALL* Y timestamps, instead of just the most recent
        if self.run_config['model_type'] == 'iterative' and sliding_window_method == 'test':
            all_timestamps = list(range(self.run_config['original_y_shape'][1]))
            transformations = self._amend_y_timestamp_selection(transformations, all_timestamps)


        self._apply_transformations(x_or_y, transformations)
        self._sample_dataset()
        self._save_ds_to_cache(time_period, sliding_window_method, x_or_y)
        dataset = self.dataset
        del self.dataset
        if type(dataset) == DiskArray:
            dataset = dataset.compute()
        return dataset

    def _load_data_for_analysis(self, time_period, sliding_window_method, x_or_y):
        """
        Load a post-hoc dataset for analysis

        :param time_period: Whether this is the training, validation, or test set
        :param sliding_window_method: What sliding window method was used for the dataset. Relevant as the validation
                                      set was processed with both the train and test sliding window method. (The
                                      training set was processed with the training sliding window method, while the
                                      test set was processed with the test sliding window method.)
        :param x_or_y: One of 'x' or 'y'
        :return:
        """
        data_dir = os.path.join(self.run_config['formatted_dir'],
                                f'{time_period}_long_term_{sliding_window_method}')

        self.dataset = loading.read_ts_data(data_dir, self.run_config['time'], x_or_y, dtype='float32', conserve_memory=self.conserve_memory)
        analysis_columns = self.run_config['columns'][
            ~self.run_config['columns']['original_index'].isna()].drop(columns=['being_used']).reset_index(drop=True).copy()
        analysis_columns['being_used'] = True

        analysis_columns, analysis_transformations = self._identify_analysis_transformations(analysis_columns)

        self._apply_transformations(x_or_y, analysis_transformations)

        dataset = self.dataset
        del self.dataset
        return dataset, analysis_columns

    def _identify_analysis_transformations(self, analysis_columns):
        """
        Equivalent of _identify_transformations, but for identifying the steps that need to be taken
        for the post-hoc analysis data

        :param analysis_columns: DataFrame specifying the original columns available
        :return:
        """

        analysis_transformations = []

        base_datetime_idx = loading._find_current_col_idx('base_datetime', analysis_columns)
        analysis_columns, analysis_transformations = self._identify_cols_to_add(
            analysis_columns, analysis_transformations,
            ['hour', 'day_of_week'],'int16',
            extra_info ={'base_datetime_idx': base_datetime_idx}
        )

        # Delete the base_datetime column
        analysis_columns, analysis_transformations = self._identify_cols_to_delete(
            analysis_columns, analysis_transformations,
            ['base_datetime'])


        # Add bearing an
        extra_info = {'lat_idx': loading._find_current_col_idx('lat', analysis_columns),
                      'lon_idx':loading._find_current_col_idx('lon', analysis_columns)}
        analysis_columns, analysis_transformations = self._identify_cols_to_add(
            analysis_columns, analysis_transformations,
            ['mean_bearing_angle','std_bearing_angle'],'float32',
            extra_info
        )

        # Add distance traveled
        extra_info = {'lat_lon_idxs':
                          [loading._find_current_col_idx('lat', analysis_columns),
                           loading._find_current_col_idx('lon', analysis_columns)]}
        analysis_columns, analysis_transformations = self._identify_cols_to_add(
            analysis_columns, analysis_transformations,
            ['distance_traveled'], 'float32', extra_info
        )

        weather_columns = analysis_columns['column'][analysis_columns['column_group'] == 'weather'].to_list()
        water_u_columns = [w for w in weather_columns if 'water_u' in w]
        water_v_columns = [w for w in weather_columns if 'water_v' in w]

        analysis_columns, analysis_transformations = self._identify_cols_to_add(
            analysis_columns, analysis_transformations,
            ['water_u_mean','water_u_std'],'float64',
            extra_info = {'idx_to_summarize': [loading._find_current_col_idx(c, analysis_columns)
                                          for c in water_u_columns]}

        )
        analysis_columns, analysis_transformations = self._identify_cols_to_add(
            analysis_columns, analysis_transformations,
            ['water_v_mean','water_v_std'],'float64',
            extra_info = {'idx_to_summarize': [loading._find_current_col_idx(c, analysis_columns)
                                          for c in water_v_columns]}

        )

        water_u_idxs = [loading._find_current_col_idx(c, analysis_columns) for c in water_u_columns]
        water_u_idxs = np.array(water_u_idxs)[np.argsort(water_u_columns)].tolist()

        water_v_idxs = [loading._find_current_col_idx(c, analysis_columns) for c in water_v_columns]
        water_v_idxs = np.array(water_v_idxs)[np.argsort(water_v_columns)].tolist()

        analysis_columns, analysis_transformations = self._identify_cols_to_add(
            analysis_columns, analysis_transformations,
            ['mean_current_magnitude','std_current_magnitude'],'float64',
            extra_info = {'idx_pairs': list(zip(water_u_idxs, water_v_idxs))}
        )


        # Convert to pandas
        analysis_transformations += [{'dataset':['x','y'], 'function':'convert_to_pandas',
                                      'df_columns':analysis_columns['column'][analysis_columns['being_used']].to_list(),
                                      'lat_lon_idxs':[loading._find_current_col_idx('lat', self.run_config['columns']),
                                                      loading._find_current_col_idx('lon', self.run_config['columns'])]
                                                    }]
        analysis_columns = pd.concat([
            analysis_columns,
            pd.DataFrame({
                'column': ['lats','lons'],
                'dtype': 'object',
                'being_used': True
            })]).reset_index(drop=True)

        # Look at current *around the boat itself*
        analysis_columns, analysis_transformations = self._identify_cols_to_add(
            analysis_columns, analysis_transformations,
            ['50_closest_weather_magnitude_mean',
               '50_closest_weather_magnitude_std', '25_closest_weather_magnitude_mean',
               '25_closest_weather_magnitude_std', '10_closest_weather_magnitude_mean',
               '10_closest_weather_magnitude_std', '5_closest_weather_magnitude_mean',
               '5_closest_weather_magnitude_std', '1_closest_weather_magnitude_mean'],'float64'
        )

        analysis_columns, analysis_transformations = self._identify_cols_to_add(
            analysis_columns, analysis_transformations,
            ['closest_weather_us', 'closest_weather_vs', 'closest_weather_lats', 'closest_weather_lons'],
            'object'
        )


        analysis_columns, analysis_transformations = self._identify_cols_to_delete(
            analysis_columns, analysis_transformations,
            water_u_columns+water_v_columns)

        # Condense vessel group
        vg_columns = analysis_columns[analysis_columns['column_group'] == 'vessel_group']['column'].to_list()
        analysis_columns, analysis_transformations = self._identify_cols_to_add(
            analysis_columns, analysis_transformations,
            ['vessel_group'], 'object',
            extra_info = {'ohe_cols': vg_columns}
        )
        analysis_columns, analysis_transformations = self._identify_cols_to_delete(analysis_columns,
                                                                                   analysis_transformations,
                                                                                   vg_columns)


        # Condense destination cluster
        dst_columns = analysis_columns[
            (analysis_columns['column_group'] == 'destination_cluster')
            & ~analysis_columns['column'].str.contains('center')
        ]['column'].to_list()
        analysis_columns, analysis_transformations = self._identify_cols_to_add(
            analysis_columns, analysis_transformations,
            ['destination_cluster'], 'object',
            extra_info = {'ohe_cols': dst_columns}
        )

        analysis_columns, analysis_transformations = self._identify_cols_to_delete(analysis_columns,
                                                                                   analysis_transformations,
                                                                                   dst_columns)
        bins = [
            ('sog', 4, 16),
            ('cog', 60, None),
            ('mean_bearing_angle',60, None),
            ('std_bearing_angle',30, None),
            ('distance_traveled', 10, 70),
            ('water_u_mean',50, 300),
            ('water_u_std', 50, 300),
            ('water_v_mean',50, 300),
            ('water_u_std', 50, 300),
            ('mean_current_magnitude',50, 400),
            ('std_current_magnitude', 50, 300)
        ]


        analysis_columns, analysis_transformations = self._identify_cols_to_add(
            analysis_columns,analysis_transformations,
            [b[0] + '_bin' for b in bins], 'str',
            extra_info={'bins':bins}
        )

        return analysis_columns, analysis_transformations


    def _sample_dataset(self):
        """
        For the debugging dataset, sample a small number of rows

        :return:
        """
        if self.run_config['debug']:
            TO_SAMPLE = 2048 * 5
            if isinstance(self.dataset, list):
                for i in range(len(self.dataset)):
                    self.dataset[i] = self.dataset[i][:TO_SAMPLE]
            elif isinstance(self.dataset, DiskArray):
                self.dataset = self.dataset.head(TO_SAMPLE)
            else:
                self.dataset = self.dataset[:TO_SAMPLE]

    def _create_columns_df(self):
        """
        Create a DataFrame specifying the columns that are available perfore the data_loading does any processing.

        Includes some supplementary information, such as what group a one-hot encoded column belongs to (e.g. 'weather')

        :return:
        """
        self.run_config['columns'] = pd.read_csv(os.path.join(self.run_config['formatted_dir'], 'features.csv'))
        self.run_config['columns'].index.name = 'original_index'
        self.run_config['columns'] = self.run_config['columns'].reset_index()
        self.run_config['columns']['being_used'] = True
        self.run_config['columns']['column_group'] = np.where(
            ((self.run_config['columns']['column'] == 'weather_is_imputed')
             | (self.run_config['columns']['column'] == 'time_since_weather_obs')
             | (self.run_config['columns']['column'].str.contains('water_'))),
            'weather',
            np.nan
        )
        for col in self.config.categorical_columns:
            self.run_config['columns']['column_group'] = np.where(
                (self.run_config['columns']['column'].str.contains(col)),
                col,
                self.run_config['columns']['column_group']
            )

    def _find_shapes(self):
        """
        Find the number of timestamps and columns in the original datasets, before any processing steps have been performed

        :return:
        """
        data_dir = os.path.join(self.run_config['formatted_dir'],
                                f'test_long_term_test')
        X = loading.read_ts_data(data_dir, self.run_config['time'], 'x', dtype='float32', conserve_memory=self.conserve_memory)
        self.run_config['original_x_shape'] = [None] + list(X.shape[1:])
        del X

        Y = loading.read_ts_data(data_dir, self.run_config['time'], 'y', dtype='float32', conserve_memory=self.conserve_memory)
        self.run_config['original_y_shape'] = [None] + list(Y.shape[1:])
        del Y
        gc.collect()

