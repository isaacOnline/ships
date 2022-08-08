import argparse
import logging
import os
import numpy as np

import pandas as pd
from pydap.client import open_url, open_dods

from config import config
from config.dataset_config import datasets
from processing_step import ProcessingStep


class Downloader(ProcessingStep):
    """
    Class for downloading ocean current data from NOAA
    """
    def __init__(self):
        super().__init__()
        self._define_directories(
            from_name=None,
            to_name='ocean_current_downloads'
        )
        self._initialize_logging(args.save_log, 'ocean_current_download')


    def _get_hycom_region(self):
        """
        Find which weather region to download data from

        The coordinates for the weather regions are given through the link below
        https://www.ncei.noaa.gov/products/weather-climate-models/frnmoc-navy-global-hybrid-ocean

        :return:
        """
        ranges = {
            # RegionNum: [[Lat_min, lat_max], [lon_min, lon_max]]
            1: [(0.0, 70.0), (-99.99996948242188, -50.0)],
            6: [(10.0, 70.0), (-150.00001525878906, -210.0)],
            7: [(10.0, 60.0), (-149.99996948242188,-100.0)],
            17: [(60.0, 80.0), (-179.99996948242188, -120.0)]
        }

        for region_num, [(lat_min, lat_max), (lon_min, lon_max)]  in ranges.items():
            if config.dataset_config.lat_1 >= lat_min and config.dataset_config.lat_2 <= lat_max:
                if config.dataset_config.lon_1 >= lon_min and config.dataset_config.lon_2 <= lon_max:
                    return region_num
        raise ValueError("Regional weather data not available for the lat/lon coordinates chosen. They may be "
                         "available in HYCOM's global surface currents dataset, which uses a slightly different"
                         "url format. See the link in the docstring to amend the code for that dataset. ")



    def _define_directories(self, from_name, to_name):
        """
        Save file paths to directory as member variable

        Override of the ProcessingStep's _define_directories, as the downloader does not have a from_dir.

        :param from_name: Should always be None, but included to keep signature in line with ProcessingStep's method
        :param to_name: Should always be 'ocean_current_downloads', but included to keep signature in line with
                        ProcessingStep's method
        :return:
        """
        self.box_and_year_dir = os.path.join(
            config.data_directory,
            f'{config.dataset_config.lat_1}_{config.dataset_config.lat_2}_'
            f'{config.dataset_config.lon_1}_{config.dataset_config.lon_2}_'
            f'{config.start_year}_{config.end_year}'
        )
        self.from_dir = from_name
        self.to_dir = os.path.join(self.box_and_year_dir, to_name)
        self.artifact_directory = os.path.join(self.box_and_year_dir, 'artifacts')

        self._create_directories()

    def _get_map_idxs(self, dataset, variable, map):
        if map == 'time':
            time_offset = pd.to_datetime(dataset['time'].attributes['units'].replace('hours since ', ''))
            min = pd.to_datetime(f'{config.start_year}-01-01').tz_localize(time_offset.tzname())
            max = pd.to_datetime(f'{config.end_year + 1}-01-01').tz_localize(time_offset.tzname())
        else:
            max = getattr(config.dataset_config, f'{map}_2')
            min = getattr(config.dataset_config, f'{map}_1')

        if map in ['lat', 'lon']:
            lat_lon_margin = 1
            max += lat_lon_margin
            min -= lat_lon_margin
        if hasattr(dataset[map], 'modulo'):
            if dataset[map].modulo == '360 degrees':
                max %= 360
                min %= 360
            else:
                raise ValueError(f'Unknown module: {dataset[variable][map].modulo} for dataset with id {dataset.id}')

        map_vals = np.array(dataset[map][:])
        if map == 'time':
            map_vals = pd.to_datetime([time_offset + pd.Timedelta(hours=h) for h in map_vals])

        idxs = np.where((map_vals <= max) & (map_vals >= min))[0]


        if len(idxs) == 0:
            raise ValueError('No surface current observations are in target range')

        if len(idxs) > 1:
            continuous = len(np.unique(idxs[1:] - idxs[:-1])) == 1
            if not continuous:
                raise ValueError(f'Slice for map {map} with dataset {dataset.id} is not continuous')

        min_idx = idxs.min()
        max_idx = idxs.max()

        map_vals = map_vals[(map_vals <= max) & (map_vals >= min)]
        return min_idx, max_idx, map_vals


    def download(self):
        """
        Download relevant dataset from NOAA

        This accesses the aggregated NetCDF using OPENDAP. It only downloads the ocean current U/V values for the water
        surface (i.e. it doesn't download the currents below the surface). It downloads time chunks so as to not
        overload the THREDDS server, e.g. requesting the first two months, then the next two, and so on.

        :return:
        """
        logging.info('Starting downloads')

        # Get the url to query
        region = self._get_hycom_region()
        aggregated_url = ('https://www.ncei.noaa.gov/'
                          f'thredds-coastal/dodsC/hycom/hycom_reg{region}_agg/'
                          f'HYCOM_Region_{region}_Aggregation_best.ncd')

        # Open connection to url
        sample_ds = open_url(aggregated_url)

        # Find the correct indexes that we want to filter down to
        time_min, time_max, time_vals = self._get_map_idxs(sample_ds, 'water_u', 'time')
        depth_min, depth_max, depth_vals = self._get_map_idxs(sample_ds, 'water_u', 'depth')
        lat_min, lat_max, lat_vals = self._get_map_idxs(sample_ds, 'water_u', 'lat')
        lon_min, lon_max, lon_vals = self._get_map_idxs(sample_ds, 'water_u', 'lon')

        # Only download this many time points at once. This number can be changed if there are timeout issues.
        time_points_to_download_at_once = 1000
        time_slices = np.arange(time_min, time_max, time_points_to_download_at_once)

        for var in config.currents_variables:
            for j, min_time in enumerate(time_slices):
                max_time = min(time_max, min_time + time_points_to_download_at_once - 1)

                # Add a filter to the url so we just download for the desired coordinates/time period/depth
                filtered_url = aggregated_url + (
                    f'.dods?{var}.{var}'
                    f'[{min_time}:1:{max_time}]'
                    f'[{depth_min}:1:{depth_max}]' 
                    f'[{lat_min}:1:{lat_max}]'
                    f'[{lon_min}:1:{lon_max}]'
                )
                # Open connection to filtered url
                data = open_dods(filtered_url)

                # Download data
                data = (np.array(data.data))[0,0,:,0]
                # I was getting an error about little endian/big endian mismatch that the byteswapping fixes
                data = [pd.DataFrame(data[i].byteswap().newbyteorder(), index=lat_vals, columns=lon_vals) for i in range(len(data))]

                times = time_vals[min_time - time_min: max_time-time_min + 1]
                # Reshape downloaded data
                for i in range(len(data)):
                    data[i]['time'] = times[i]
                    data[i].index.name = 'latitude'
                    data[i] = data[i].reset_index()
                    data[i] = pd.melt(
                        data[i],
                        id_vars=['latitude','time'],
                        value_vars=lon_vals
                    )
                    data[i] = data[i].rename(columns={'variable':'longitude','value':'speed'})
                    if (data[i]['longitude'] > 180).all():
                        data[i]['longitude'] -= 360
                data = pd.concat(data)
                for col in ['year','month','day','hour']:
                    data[col] = getattr(data['time'].dt, col)
                del data['time']

                # Save this downloaded dataset to csv
                data.to_csv(os.path.join(self.to_dir, f'{var}_{j}.csv'),index=False)
                logging.info(f'{var} data downloaded for times from {times[0].strftime("%Y-%m-%d %H:%M")} to '
                             f'{times[-1].strftime("%Y-%m-%d %H:%M")}')

        logging.info(f'All downloads complete for coordinates {config.dataset_config.corner_1},'
                     f' {config.dataset_config.corner_2}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_name', choices=datasets.keys())
    # Tool for debugging
    parser.add_argument('-l', '--log_level', type=int,
                        default=2, choices=[0, 1, 2, 3, 4],
                        help='Level of logging to use')
    parser.add_argument('-s', '--save_log', action='store_true')

    args = parser.parse_args()

    config.dataset_config = datasets[args.dataset_name]
    config.set_log_level(args.log_level)

    downloader = Downloader()
    downloader.download()
