import argparse
import logging
import os
import dask

import dask.dataframe as dd
import pandas as pd
import numpy as np

from config import config
from config.dataset_config import datasets
from processing_step import ProcessingStep


class WeatherAggregator(ProcessingStep):
    """
    Class for preprocessing the ocean currents dataset

    """
    def __init__(self):
        super().__init__()
        self._define_directories(
            from_name='ocean_current_downloads',
            to_name='ocean_currents_aggregated'
        )
        self._initialize_logging(args.save_log, 'aggregate_currents')

    def load(self):
        """
        Load the downloaded ocean currents datasets from disk

        The ocean currents datasets are split by time, but this loads them in all at once (one dataset for each weather
        variable)

        :return:
        """
        for var in config.currents_variables:
            self.datasets[var] = dd.read_csv(os.path.join(self.from_dir, f'{var}_*.csv'))


    def calculate_and_save(self):
        """
        Perform the aggregation and save to disk

        The ocean currents are forecasted at every three hours. This reformats the
        dataset so that there is a row for each of these three hour timestamps, and the
        u/v observations are stored as columns. This performs the downsampling of the u/v vectors
        so that we only keep those observed at every 0.16, 0.24, or 0.32 degrees, in order to keep
        the size of the dataset manageable. It also filters out any coordinates that are over land,
        so that they do not need to be recorded/take up space.


        :return:
        """
        if config.currents_window != 'stable':
            raise ValueError('Moving currents windows have been deprecated')
        else:
            for var in config.currents_variables:
                # Make sure that weather variables are filtered to correct region.
                self.datasets[var] = self.datasets[var][
                    (self.datasets[var]['latitude'] >= config.dataset_config.lat_1)
                    & (self.datasets[var]['latitude'] <= config.dataset_config.lat_2)
                    & (self.datasets[var]['longitude'] >= config.dataset_config.lon_1)
                    & (self.datasets[var]['longitude'] <= config.dataset_config.lon_2)
                    ]

                partition = self.datasets[var]._partitions(0).compute()

                # Even though the weather data was originally recorded at every 0.08 degrees,
                # we were unable to use data of this size on our machines, so instead created
                # a grid of approximate dimension 14 x 14 (no matter the size of the region).
                approx_grid_dim = 14
                latitudes = partition['latitude'].unique()
                longitudes = partition['longitude'].unique()

                # Calculate the stride for latitude
                lat_stride = np.round(len(latitudes) /(approx_grid_dim -1))
                latitude_idx = np.arange(0,len(latitudes), lat_stride, dtype=int)
                latitudes = latitudes[latitude_idx]

                # Calculate the stride for longitude
                lon_stride = np.round(len(longitudes) / (approx_grid_dim - 1))
                longitude_idx = np.arange(0,len(longitudes), lon_stride, dtype=int)
                longitudes = longitudes[longitude_idx]

                # Select the lat/lon coordinates we're interested in
                self.datasets[var] = self.datasets[var][self.datasets[var]['latitude'].isin(latitudes)
                                                        & self.datasets[var]['longitude'].isin(longitudes)
                                                        & ~self.datasets[var]['speed'].isna()]

                self.datasets[var]['coord'] = (
                        self.datasets[var]['latitude'].round(2).astype(str)
                        + '_'
                        + self.datasets[var]['longitude'].round(2).astype(str)
                )

                self.datasets[var] = self.datasets[var].compute()

                # Reshape dataset
                self.datasets[var] = self.datasets[var].pivot(index=['year', 'month', 'day', 'hour'],
                                                              columns='coord', values='speed')

                # In my dataset there were a few timestamps where currents were reported over land, which this filters
                # out. This also filters out locations that were only sporadically reported to.
                likely_land_coords = self.datasets[var].columns[self.datasets[var].isna().mean() > 0.5]
                self.datasets[var] = self.datasets[var].drop(columns=likely_land_coords)
                self.datasets[var] = (self.datasets[var] * 1000).astype(np.int16)


                self.datasets[var].columns = self.datasets[var].columns + '_' + var

            # Join weather variables together
            self.datasets['complete'] = self.datasets[config.currents_variables[0]].copy()
            for i in range(len(config.currents_variables) - 1):
                new_var = config.currents_variables[i+1]
                self.datasets['complete'] = pd.merge(self.datasets['complete'], self.datasets[new_var],
                                                     left_index=True,right_index=True)
            self.datasets['complete'].to_csv(os.path.join(self.to_dir, 'weather_aggregated.csv'))
            logging.info('Weather Aggregation complete')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_name', choices=datasets.keys())
    # Tool for debugging
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-l', '--log_level', type=int,
                        default=2, choices=[0, 1, 2, 3, 4],
                        help='Level of logging to use')
    parser.add_argument('-s', '--save_log', action='store_true')

    args = parser.parse_args()
    config.dataset_config = datasets[args.dataset_name]
    config.set_log_level(args.log_level)


    dask.config.set(scheduler='single-threaded')

    aggregator = WeatherAggregator()
    aggregator.load()
    aggregator.calculate_and_save()
