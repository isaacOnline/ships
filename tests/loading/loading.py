import os

from loading.disk_array import DiskArray
import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit


def read_ts_data(directory, time_gap, x_or_y, dtype=None, conserve_memory=False):
    """
    Read a dataset from disk

    Dataset should have been created using the processing code

    :param directory: Specific directory where data is stored. Should include which of the training/test/validation sets this is
    :param time_gap: The number of minutes between imputed AIS messages
    :param x_or_y: Whether this is the x or y dataset. Should either be 'x' or 'y'
    :param dtype: Datatype to read dataset in as
    :param conserve_memory: If the dataset is saved in chunks, passing in conserve_memory=True will use a DiskArray to
                            load in the dataset, while conserve_memory=False will concatenate all of the chunks together
                            into a single numpy array
    :return:
    """
    path = os.path.join(directory, f'{time_gap}_min_time_gap_{x_or_y}.npy')

    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        if dtype is not None:
            data = data.astype(dtype)

    else:
        path = os.path.join(directory, f'{time_gap}_min_time_gap_{x_or_y}')
        files = os.listdir(path)
        files = np.array(files)[np.argsort([int(n.split('.')[0]) for n in files])].tolist()
        if dtype is None:
            dtype = np.load(os.path.join(path, files[0]),allow_pickle=True).dtype
        if conserve_memory:
            data = DiskArray()
            for f in files:
                arr = np.load(os.path.join(path, f), allow_pickle=True).astype(dtype)
                data.add_array(arr)
        else:
            data = np.concatenate([np.load(os.path.join(path, f),allow_pickle=True).astype(dtype) for f in files])


    return data


def add_distance_traveled(X, lat_lon_idxs, dt_idx):
    """
    Calculate the distance traveled from the first time stamp in the dataset to the last time stamp

    Gives dt in km

    :param X: Dataset to add to
    :param lat_lon_idxs:  Indexes of the latitude and longitude columns
    :param dt_idx: Index to insert the distance traveled into
    :return:
    """
    start_location = X[:, 0, lat_lon_idxs]
    end_location = X[:, -1, lat_lon_idxs]
    distance_traveled = haversine_vector(start_location, end_location, Unit.KILOMETERS)

    # Get into correct shape
    distance_traveled = np.stack([distance_traveled] * X.shape[1], axis=1)
    X = np.insert(X, dt_idx, distance_traveled, axis=2)
    return X


def add_stats(X, col_idx, which):
    """
    Add a set of summary statistics to a dataset

    Works over the first index, which should represent timestamps

    :param X: Dataset to insert into
    :param col_idx: Index of column to summarize
    :param which: Either 'median' or 'min_median_max'
    :return:
    """
    if which == 'min_median_max':
        add_stat(X, col_idx, 'min')

    add_stat(X, col_idx, 'median')

    if which == 'min_median_max':
        add_stat(X, col_idx, 'max')

    return X


def add_stat(X, col_idx, which):
    """
    Add a summary statistic to a dataset

    Works over the first index, which should represent timestamps

    :param X: The dataset to add to
    :param col_idx: Column to be summarized
    :param which: either 'min', 'median' or 'max'
    :return:
    """
    if which == 'min':
        mins = np.stack([X[:,:,col_idx].min(axis=1)] * X.shape[1],axis=1)
        min_idx = X.shape[2]
        X = np.insert(X, min_idx, mins, axis=2)
    elif which == 'median':
        medians = np.stack([np.median(X[:,:,col_idx],axis=1)] * X.shape[1],axis=1)
        median_idx = X.shape[2]
        X = np.insert(X, median_idx, medians, axis=2)
    elif which == 'max':
        maxes = np.stack([X[:,:,col_idx].max(axis=1)] * X.shape[1],axis=1)
        max_idx = X.shape[2]
        X = np.insert(X, max_idx, maxes, axis=2)

    return X

def _calc_time_stats_1d(data, stat):
    """
    Extract a time component from a numpy array of unix times

    Data should be a 1d numpy array, containing UTC unix times

    stat should be a time attribute that pandas tracks:
    https://pandas.pydata.org/docs/user_guide/timeseries.html#time-date-components

    :param data: Times to extract component from
    :param stat: Component to extract
    :return:
    """
    data = pd.to_datetime(data * 1000000000)
    data = data.tz_localize('utc').tz_convert('US/Pacific')
    data = getattr(data, stat).to_numpy()
    return data


def add_time_stats(data, datetime_col, hour_idx, dow_idx):
    """
    Add the hour and day of week when messages occurred

    :param data: Data to add to
    :param datetime_col: Index of datetime column
    :param hour_idx: Index where hour information should be included
    :param dow_idx: Index where day of week information should be included
    :return:
    """
    hour = np.apply_along_axis(_calc_time_stats_1d, 0, data[...,datetime_col], 'hour')

    day_of_week = np.apply_along_axis(_calc_time_stats_1d, 0, data[...,datetime_col], 'day_of_week')

    data = np.insert(data, hour_idx, hour, axis=-1)
    data = np.insert(data, dow_idx, day_of_week, axis=-1)
    return data


def split_X_for_fusion(X, recurrent_idxs):
    """
    Split an input dataset in two

    X should be a 3d array, and will be split into two new arrays - a 3d array and a 2d array

    recurrent_idxs should specify the columns that will remain in 3d, while for the other columns,
    the value from the last timestamp will be taken (timestamps should be on the first axis)

    :param X: Dataset to split
    :param recurrent_idxs: Columns that should remain in 3d
    :return:
    """
    recurrent_part = X[:, :, recurrent_idxs]
    dense_part = X[:,-1,:]
    dense_part = np.delete(dense_part, recurrent_idxs, axis=-1)
    return [recurrent_part, dense_part]


def reshape_weather_data(weather_data, weather_only_cols):
    """
    Reshape flattened weather columns back into a grid

    Useful for trying convolutional layers for processing weather data

    :param weather_data: Weather data to reshape
    :param weather_only_cols: Columns that contain weather data
    :return:
    """
    imputed_col = weather_only_cols[weather_only_cols['column'] == 'weather_is_imputed'].copy()

    water_u_v = weather_only_cols[weather_only_cols['column'] != 'weather_is_imputed'].copy()
    water_u_v['lat_lons'] = water_u_v['column'].str.replace('_water_[uv]','',regex=True).str.split('_')
    water_u_v['lat'] = [float(ll[0]) for ll in water_u_v['lat_lons']]
    water_u_v['lon'] = [float(ll[1]) for ll in water_u_v['lat_lons']]

    water_u_columns = water_u_v[water_u_v['column'].str.contains('water_u')].copy()
    water_v_columns = water_u_v[water_u_v['column'].str.contains('water_v')].copy()

    data_len = len(weather_data)

    lats = (water_u_columns['lat']).unique()
    lats.sort()

    # sort descending
    lons = -np.sort((-water_u_columns['lon']).unique())
    num_lons = len(lons)

    water_u_data = []
    for lat in lats:
        slice = water_u_columns[water_u_columns['lat'] == lat]
        lat_row = np.empty((data_len, num_lons),dtype=weather_data.dtype)
        lat_row[:] = np.nan
        for i, lon in enumerate(lons):
            lat_lon_idx = slice[slice['lon'] == lon]
            if len(lat_lon_idx) == 0:
                continue
            elif len(lat_lon_idx) == 1:
                idx = lat_lon_idx.index[0]
                lat_row[:,i] = weather_data[:,idx]
        water_u_data.append(lat_row)
    water_u_data = np.stack(water_u_data,axis=-1)

    water_v_data = []
    for lat in lats:
        slice = water_v_columns[water_v_columns['lat'] == lat]
        lat_row = np.empty((data_len, num_lons),dtype=weather_data.dtype)
        lat_row[:] = np.nan
        for i, lon in enumerate(lons):
            lat_lon_idx = slice[slice['lon'] == lon]
            if len(lat_lon_idx) == 0:
                continue
            elif len(lat_lon_idx) == 1:
                idx = lat_lon_idx.index[0]
                lat_row[:,i] = weather_data[:,idx]
        water_v_data.append(lat_row)
    water_v_data = np.stack(water_v_data,axis=-1)


    # Create indicators for if a) the the location is over land, (represented by a 1) b) the current at the location has
    # been imputed, (represented by a 0.5) or c) the location is an actual value (represented by a 0)
    imputed = np.stack([np.stack([weather_data[..., imputed_col.index[0]].astype(bool)] * len(lons), axis = -1)] * len(lats), axis=-1)
    coast = np.isnan(water_u_data)

    imputation_channel = coast.astype(weather_data.dtype)
    imputation_channel[imputed & ~coast] = 0.5

    water_u_data[np.isnan(water_u_data)] = 0.5
    water_v_data[np.isnan(water_v_data)] = 0.5

    return np.stack([water_u_data, water_v_data, imputation_channel], axis=-1)



def _find_current_col_idx(col, columns):
    """
    Find the current idx of a column

    columns object should have a 'being_used' boolean column

    :param col: Name of column
    :param columns: DataFrame of columns
    :return:
    """
    # I'm finding out the current index of the column I want to delete by, essentially,
    # counting up how many columns (that we haven't deleted yet) are in the dataset before it
    idx_in_columns_df = np.where(columns['column'] == col)[0][0]
    if not columns['being_used'].iloc[idx_in_columns_df]:
        raise ValueError(f'Columns {col} is trying to be accessed even though it is not currently in the dataframe')
    current_idx = (columns['being_used'][:idx_in_columns_df]).sum()
    return int(current_idx)


def apply_transformations(dataset, x_or_y, transformations, normalizer, normalization_factors):
    """
    Apply a set of transformations to a dataset

    The transformations list should have been created by the data loader

    :param dataset: Dataset to transform
    :param x_or_y: Whether this is the input or output dataset
    :param transformations: List of transformations to apply, along with supplementary info
    :param normalizer: Normalizer object for dataset
    :param normalization_factors: Normalization factors, which should have been calculated by the normalizer
    :return:
    """
    for transformation in transformations:
        if x_or_y in transformation['dataset']:
            if transformation['function'] == 'select_timestamps':
                assert type(dataset) == np.ndarray
                ts_to_select = transformation['to_select']
                dataset = dataset.take(indices=ts_to_select, axis=1)
            elif transformation['function'] == 'add_columns':
                dataset = _add_columns(dataset, transformation)
            elif transformation['function'] == 'remove_columns':
                dataset = _remove_columns(dataset, transformation)
            elif transformation['function'] == 'normalize':
                assert type(dataset) == np.ndarray
                dataset = normalizer.normalize_data(dataset, normalization_factors)
            elif transformation['function'] == 'split_for_fusion':
                assert type(dataset) == np.ndarray
                dataset = split_X_for_fusion(dataset, transformation['indexes'])
            elif transformation['function'] == 'select_columns':
                assert type(dataset) == np.ndarray
                cols_to_select = transformation['indexes']
                dataset = dataset.take(indices=cols_to_select, axis=2)
            elif transformation['function'] == 'reshape_weather':
                assert type(dataset) == np.ndarray
                weather_cols = pd.DataFrame(transformation['columns'], index=transformation['indexes'],
                                            columns=['column'])
                dataset[1] = reshape_weather_data(dataset[1], weather_cols)
            elif transformation['function'] == 'squeeze':
                assert type(dataset) == np.ndarray
                dataset = dataset.squeeze()
            elif transformation['function'] == 'convert_to_pandas':
                assert type(dataset) == np.ndarray
                if len(dataset.shape) == 3:
                    trajs = dataset[:, :, transformation['lat_lon_idxs']]
                    lats = [trajs[i, :, 0].tolist() for i in range(len(trajs))]
                    lons = [trajs[i, :, 1].tolist() for i in range(len(trajs))]
                    dataset = pd.DataFrame(dataset[:,-1,:], columns=transformation['df_columns'])
                    dataset['lats'] = lats
                    dataset['lons'] = lons
                else:
                    dataset = pd.DataFrame(dataset, columns=transformation['df_columns'])

    return dataset

def get_bearing(lat1, long1, lat2, long2):
    """
    Get the bearing angle between arrays of latitudes and longitudes

    :param lat1: Latitudes for first set of points
    :param long1: Longitudes for first set of points
    :param lat2: Latitudes for second set of points
    :param long2: Longitudes for second set of points
    :return:
    """
    dLon = (long2 - long1)
    x = np.cos(np.radians(lat2)) * np.sin(np.radians(dLon))
    y = (np.cos(np.radians(lat1))
         * np.sin(np.radians(lat2))
         - np.sin(np.radians(lat1))
         * np.cos(np.radians(lat2))
         * np.cos(np.radians(dLon)))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)

    brng %= 360

    return brng

def _remove_columns(dataset, transformation):
    """
    Delete a column from a dataset

    :param dataset: Dataset to delete from
    :param transformation: Information about column to delete. Should have been created by DataLoader
    :return:
    """
    if type(dataset) == pd.DataFrame:
        return dataset.drop(columns = transformation['columns'])
    else:
        return np.delete(dataset, transformation['indexes'], axis=-1)

def _add_columns(dataset, transformation):
    """
    Add a column to a dataset

    :param dataset: Dataset to add to
    :param transformation: Information for column to add. Should have been created by DataLoader
    :return:
    """
    if transformation['columns'] == ['hour', 'day_of_week']:
        assert type(dataset) == np.ndarray
        dataset = add_time_stats(dataset,
                                 transformation['base_datetime_idx'],
                                 *transformation['indexes'])
    elif 'sog_median' in transformation['columns']:
        assert type(dataset) == np.ndarray
        for col, idx in zip(transformation['columns'],transformation['indexes']):
            col, summary = col.split('_')
            dataset = add_stat(dataset,
                               transformation[f'{col}_index'],
                               summary)
    elif transformation['columns'] == ['distance_traveled']:
        assert type(dataset) == np.ndarray
        dataset = add_distance_traveled(dataset,
                                        lat_lon_idxs=transformation['lat_lon_idxs'],
                                        dt_idx=transformation['indexes'][0])
    elif np.any([c in ['water_u_mean', 'water_v_mean'] for c in transformation['columns']]):
        assert type(dataset) == np.ndarray
        for c, idx in zip(transformation['columns'], transformation['indexes']):
            if 'mean' in c:
                stat = np.abs(dataset[...,transformation['idx_to_summarize']]).mean(axis=-1)
            elif 'std' in c:
                stat = dataset[...,transformation['idx_to_summarize']].std(axis=-1)
            else:
                raise ValueError (f'Unknown column to calculate {c}')
            dataset = np.insert(dataset, idx, stat, axis=-1)
    elif 'mean_current_magnitude' in transformation['columns']:
        assert type(dataset) == np.ndarray
        magnitudes = [np.sqrt(np.sum(np.power(dataset[...,p], 2) , axis = -1)) for p in transformation['idx_pairs']]
        for c, idx in zip(transformation['columns'], transformation['indexes']):
            if 'mean' in c:
                stat = np.mean(np.array(magnitudes), axis= 0)
            elif 'std' in c:
                stat = np.std(np.array(magnitudes), axis= 0)
            else:
                raise ValueError (f'Unknown column to calculate {c}')
            dataset = np.insert(dataset, idx, stat, axis=-1)
    elif transformation['columns'] in [['vessel_group'], ['destination_cluster']]:
        assert type(dataset) == pd.DataFrame
        vals = np.empty((len(dataset)), dtype='str')
        vals[:] = ''
        for with_prefix in transformation['ohe_cols']:
            without_prefix = with_prefix.replace(transformation['columns'][0] + '_', '')
            vals = np.char.add(vals, np.where(dataset[with_prefix], without_prefix, '').astype('str'))

        dataset[transformation['columns'][0]] = vals
    elif transformation['columns'] == ['mean_bearing_angle','std_bearing_angle']:
        assert type(dataset) == np.ndarray
        lats = dataset[...,transformation['lat_idx']]
        lons = dataset[...,transformation['lon_idx']]
        angles = []
        for i in range(lats.shape[1] - 1):
            angles += [get_bearing(lats[...,i],lons[...,i],
                                   lats[...,i+1],lons[...,i+1])]
        angles = np.stack(angles).T
        for c, idx in zip(transformation['columns'], transformation['indexes']):
            if c == 'mean_bearing_angle':
                stat = np.mean(angles, axis= 1)
            elif c == 'std_bearing_angle':
                stat = np.std(angles, axis= 1)

            stat = np.stack([stat] * dataset.shape[1]).T
            dataset = np.insert(dataset, idx, stat, axis=-1)
    elif 'sog_bin' in transformation['columns']:
        assert type(dataset) == pd.DataFrame
        for (original_c, bin_size, bin_cutoff), new_c in zip(transformation['bins'], transformation['columns']):
            binned = (dataset[original_c] // bin_size * bin_size).astype(int)
            if bin_cutoff is not None:
                binned = np.where(binned < bin_cutoff,
                                  '[' + binned.astype(str) + ', ' + (binned + bin_size).astype(str) + ')',
                                  f'[{bin_cutoff}+]')
            else:
                binned = '[' + binned.astype(str) + ', ' + (binned + bin_size).astype(str) + ')'
            dataset[new_c] = binned
    elif '50_closest_weather_magnitude_mean' in transformation['columns']:
        assert type(dataset) == pd.DataFrame
        dataset.index.name = 'id'
        dataset = dataset.reset_index()

        water_u_cols = [w for w in dataset.columns if 'water_u' in w and not ('mean' in w or 'std' in w)]
        water_v_cols = [w for w in dataset.columns if 'water_v' in w and not ('mean' in w or 'std' in w)]

        u = pd.melt(dataset, id_vars=['id', 'lat', 'lon'], value_vars=water_u_cols).rename(
            columns={'value': 'water_u'})
        u['variable'] = u['variable'].str.replace('_water_u','')
        v = pd.melt(dataset, id_vars=['id', 'lat', 'lon'], value_vars=water_v_cols).rename(
            columns={'value': 'water_v'})
        v['variable'] = u['variable'].str.replace('_water_v','')

        magnitudes = u.merge(v, on=['id','variable','lat','lon'])
        magnitudes['magnitude'] = np.sqrt(magnitudes['water_v'] ** 2 + magnitudes['water_u'] ** 2)
        magnitudes = magnitudes.drop(columns=['water_u','water_v'])
        magnitudes['weather_lat'] = magnitudes['variable'].str.split('_').str[0].astype(float)
        magnitudes['weather_lon'] = magnitudes['variable'].str.split('_').str[1].astype(float)
        magnitudes = magnitudes.drop(columns='variable')
        magnitudes['distance_to_weather'] = haversine_vector(magnitudes[['lat','lon']], magnitudes[['weather_lat','weather_lon']])
        magnitudes = magnitudes.sort_values(['id','distance_to_weather']).reset_index(drop=True)

        n = 50
        closest = magnitudes.groupby('id')[['id','magnitude']].head(n).groupby('id').agg(['mean','std'])
        closest.columns = [f'{n}_closest_weather_magnitude_mean',f'{n}_closest_weather_magnitude_std']

        for n in [25, 10, 5, 1]:
            c = magnitudes.groupby('id')[['id', 'magnitude']].head(n).groupby('id').agg(['mean', 'std'])
            c.columns = [f'{n}_closest_weather_magnitude_mean', f'{n}_closest_weather_magnitude_std']
            closest = closest.merge(c, on = 'id')
        closest = closest.drop(columns=['1_closest_weather_magnitude_std'])
        dataset = dataset.merge(closest, on = 'id')
        dataset = dataset.drop(columns='id')
    elif 'closest_weather_us' in transformation['columns']:
        assert type(dataset) == pd.DataFrame
        dataset.index.name = 'id'
        dataset = dataset.reset_index()

        water_u_cols = [w for w in dataset.columns if 'water_u' in w and not ('mean' in w or 'std' in w)]
        water_v_cols = [w for w in dataset.columns if 'water_v' in w and not ('mean' in w or 'std' in w)]

        u = pd.melt(dataset, id_vars=['id', 'lat', 'lon'], value_vars=water_u_cols).rename(
            columns={'value': 'water_u'})
        u['variable'] = u['variable'].str.replace('_water_u', '')
        v = pd.melt(dataset, id_vars=['id', 'lat', 'lon'], value_vars=water_v_cols).rename(
            columns={'value': 'water_v'})
        v['variable'] = u['variable'].str.replace('_water_v', '')

        magnitudes = u.merge(v, on=['id', 'variable', 'lat', 'lon'])
        magnitudes['weather_lat'] = magnitudes['variable'].str.split('_').str[0].astype(float)
        magnitudes['weather_lon'] = magnitudes['variable'].str.split('_').str[1].astype(float)
        magnitudes = magnitudes.drop(columns='variable')
        magnitudes['distance_to_weather'] = haversine_vector(magnitudes[['lat', 'lon']],
                                                             magnitudes[['weather_lat', 'weather_lon']])
        magnitudes = magnitudes.sort_values(['id', 'distance_to_weather']).reset_index(drop=True)

        n = 35
        closest = magnitudes.groupby('id')[['id','water_u', 'water_v', 'weather_lat', 'weather_lon','distance_to_weather']].head(n)
        closest['rank'] = closest.groupby('id')['distance_to_weather'].rank('min').astype(int)
        closest = closest.drop(columns='distance_to_weather')
        closest.columns = ['id','u', 'v', 'lat', 'lon','rank']

        closest = closest.pivot(index='id', columns='rank', values=['u', 'v', 'lat', 'lon'])
        closest.columns = [f'{i}_closest_weather_{v}' for v, i in closest.columns]
        for col in ['u','v','lat','lon']:
            closest[f'closest_weather_{col}s'] = closest[[f'{i}_closest_weather_{col}' for i in range(1, n + 1)]].values.tolist()
        closest = closest[transformation['columns']]
        dataset = dataset.merge(closest, on = 'id')
        dataset = dataset.drop(columns='id')



    return dataset

