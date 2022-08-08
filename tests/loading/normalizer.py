import numpy as np

from loading.disk_array import DiskArray
from loading.loading import _find_current_col_idx
from config import config


class Normalizer():
    def __init__(self):
        pass

    @staticmethod
    def get_normalization_factors(X, columns):
        """
        Get the normalization factors for a specific dataset

        Find the min and max values for different columns, which are used when normalizing the dataset to a 0-1 range

        :param X: Dataset
        :param columns: The names and types of columns in X
        :return: Normalization factors
        """
        year_idx = _find_current_col_idx('year', columns)
        month_idx = _find_current_col_idx('month', columns)

        normalization_factors = {
            'lat': {
                'idx': _find_current_col_idx('lat', columns),
                'min': float(config.dataset_config.lat_1),
                'max': float(config.dataset_config.lat_2)
            },
            'lon': {
                'idx': _find_current_col_idx('lon', columns),
                'min': float(config.dataset_config.lon_1),
                'max': float(config.dataset_config.lon_2)
            },
            'year': {
                'idx': year_idx,
                'min': min(config.years),
                'max': max(config.years)
            },
            'month': {
                'idx': month_idx,
                'min': 1,
                'max': 12
            }

        }
        cols_to_use = columns['column'][columns['being_used']]
        non_bools = cols_to_use[columns.dtype != 'bool']
        multi_cols = ['speed','water','mmsi_neighbor','lat_neighbor','lon_neighbor','time_since_neighbor']
        ranges = {k: [np.Inf, -np.Inf] for k in multi_cols}
        if type(X) == DiskArray:
            mins, maxes = X._calculate_min_max()
        for col in non_bools:
            if col not in normalization_factors.keys():
                idx = _find_current_col_idx(col, columns)
                normalization_factors[col] = {
                    'idx': idx,
                    'min': mins[idx] if type(X) == DiskArray else float(X[:, :, idx].min()),
                    'max': maxes[idx] if type(X) == DiskArray else float(X[:, :, idx].max())
                }
                for mc in multi_cols:
                    if mc in col:
                        ranges[mc][0] = min(normalization_factors[col]['min'], ranges[mc][0])
                        ranges[mc][1] = max(normalization_factors[col]['max'], ranges[mc][1])

        for col in non_bools:
            for mc in multi_cols:
                if mc in col:
                    normalization_factors[col]['min'] = ranges[mc][0]
                    normalization_factors[col]['max'] = ranges[mc][1]

        return normalization_factors

    @staticmethod
    def normalize_data(data, normalization_factors):
        """
        Apply the normalization factors to a dataset

        Uses min/max normalization (and the mins/maxes specified in normalization_factors) to put variables on a 0 to 1 range

        Assumes data is 3D array of shape (# of timestamps, # of trajectories, # of columns)

        :param data: Dataset to normalize
        :param normalization_factors: Normalization factors
        :return: Normalized data
        """

        if len(data.shape) == 3:
            for col in normalization_factors.values():
                if col['idx'] < data.shape[-1]:
                    range = (col['max'] - col['min'])
                    if range != 0:
                        dist_above_min = (data[:, :, col['idx']] - col['min'])
                        data[:, :, col['idx']] = dist_above_min / range
                    else: # if the variable doesn't vary at all (which can happen when debugging), just set it to 0
                        data[:,:, col['idx']] = 0

        else:
            for col in normalization_factors.values():
                if col['idx'] < data.shape[-1]:
                    range = (col['max'] - col['min'])
                    if range != 0:
                        dist_above_min = (data[:, col['idx']] - col['min'])
                        data[:, col['idx']] = dist_above_min / range
                    else:
                        data[:, col['idx']] = 0
        return data

    @staticmethod
    def unnormalize(data, normalization_factors, idxs=None):
        """
        Move data from 0-1 scale back to original scale

        :param data: Data to unnormalize
        :param normalization_factors: Normalization factors
        :param idxs: Indexes to unnormalize
        :return: Unnormalized data
        """
        data = data.copy()
        if idxs is None:
            idxs = np.arange(0, data.shape[-1])

        if len(data.shape) == 3:
            for col in normalization_factors.values():
                if np.any(col['idx'] == idxs):
                    range = (col['max'] - col['min'])
                    data[:, :, col['idx']] = data[:, :, col['idx']] * range + col['min']
        else:
            for col in normalization_factors.values():
                if np.any(col['idx'] == idxs):
                    range = (col['max'] - col['min'])
                    data[:, col['idx']] = data[:, col['idx']] * range + col['min']

        return data