import argparse
import logging
import os
import gc
import dask
import re

import dask.dataframe as dd
import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit

from config import config
from config.dataset_config import datasets
from processing_step import ProcessingStep
from utils import get_info_from_specifier, pd_append, to_snake_case, clear_path, all_specifiers, get_min_max_times


class Cleaner(ProcessingStep):
    """
    Class for performing cleaning steps of preprocessing
    """
    def __init__(self, test_fraction, validation_fraction):
        super().__init__()
        self._define_directories(
            from_name='filtered',
            to_name='cleaned' + ('_debug' if args.debug else '')
        )
        self.from_dir = os.path.join(self.box_and_year_dir, 'downloads','filtered')
        self._initialize_logging(args.save_log, 'clean')

        if not args.seed:
            args.seed = np.random.randint(100000000)
            logging.warning(f'No speed specified. Using randomly generated seed {args.seed}')
        else:
            logging.info(f'Using random seed {args.seed} specified by user')
        np.random.seed(args.seed)

        self.test_fraction = test_fraction
        self.validation_fraction = validation_fraction
        self.original_lengths = {}
        self.unneeded_columns = [
            'vessel_type',
            'status',
            'length',
            'width',
            'draft',
            'cargo',
            'transceiver_class'
        ]

        logging.info(f'Not using columns {self.unneeded_columns}')


    def load(self):
        """
        Load the test and train files

        Will first determine, based on the years specified, which downloaded files will go into the test set and
        which will go into the train set. Will then read and begin preprocessing these files.

        :return:
        """
        test_files, train_files = self._get_test_train_files()
        if args.debug:
            test_files = test_files[:8]
            train_files = train_files[:8]
        logging.debug('Loading test set')
        self.datasets['test'] = self._load_files(test_files, 'test')
        self.original_lengths['test'] = len(self.datasets['test'])

        logging.info('Loading test set complete, now loading train set')

        self.datasets['train'] = self._load_files(train_files, 'train')
        self.original_lengths['train'] = len(self.datasets['train'])

        logging.info('Loading train set complete')

    def _get_test_train_files(self):
        """
        Determine which files will be used for testing set and which for training set

        The test/train split is done by time, meaning that the training and testing sets come from non-overlapping time
        periods, with the training set using data from before the testing period. This method looks through the years
        that were specified, and finds the month to split on that best match the test_fraction specified during
        initialization. So e.g. if you specified only using date from 2018 with a test fraction of 1/6, this would
        calculate that you will use November/December for testing and all earlier months for training.

        :return: Files to use for testing, files to use for training
        """
        files_to_read = all_specifiers(self.zones, config.years, 'parquet', self.from_dir)['paths']

        # Determine which months to use for test and which for train
        all_months = []
        for y in config.years:
            for m in range(1,13):
                all_months.append((y,m))
        number_of_test_months = len(all_months) * self.test_fraction
        if number_of_test_months % 1 != 0:
            number_of_test_months = int(number_of_test_months)
            true_test_fraction = number_of_test_months / len(all_months)
            logging.warning(f'Test fraction was rounded from {self.test_fraction:0.6} to {true_test_fraction:0.5} '
                            f'because there were only {len(all_months)} months to split up')
        else:
            number_of_test_months = int(number_of_test_months)
        train_months = all_months[:-number_of_test_months]
        test_months = all_months[-number_of_test_months:]

        # Log the split that was found
        last_train_year, last_train_month = train_months[-1]
        first_test_year, first_test_month = test_months[-0]
        logging.info(f'The last month being used in the training set is zone from '
                     f'{last_train_month}/{last_train_year}, while the first data file being used in the test '
                     f'set is from {first_test_month}/{first_test_year}')

        # Split the files
        train_datasets = [fn for fn in files_to_read if
                          (int(get_info_from_specifier(fn)[0]), int(get_info_from_specifier(fn)[1])) in train_months]
        test_datasets = [fn for fn in files_to_read if
                          (int(get_info_from_specifier(fn)[0]), int(get_info_from_specifier(fn)[1])) in test_months]

        return test_datasets, train_datasets

    def _drop_unwanted_columns(self, partition):
        """
        Drop any columns that are not needed after this step

        :param partition: dataset (pd.DataFrame)
        :return: dataset with columns dropped
        """
        # transceiver_class may or may not be in the dataset prior to this step, so here we are making sure it only gets
        # dropped if it was indeed in the dataset
        if 'transceiver_class' in partition.columns:
            unneeded_columns = self.unneeded_columns
        else:
            unneeded_columns = [c for c in self.unneeded_columns if c != 'transceiver_class']

        partition = partition.drop(columns=unneeded_columns)
        return partition

    def _load_files(self, files, name):
        """
        Read in files

        Goes through each file and performs any possible filtering (in order to cut down on dataset size), then
        saves these as temporary versions to disk. Next reads in all temp files into a single dataset, and sorts them
        by MMSI and datetime (this is slow but only needs to be done once). This is again saved as a temp file, then
        loaded in as the test/train set.

        For the individual filtering to work properly, all messages from the same time period must be in the same
        dataset. For this reason, monthly files from 2015-2017, (which are separated by Zone), are combined. (E.g.
        if the dataset contains messages from UTM zones 10 and 11, then this will combine the January zone 10/11 files
        into a single file before doing the initial filtering.)

        :param files: Names of files to read
        :return: Dataset
        """
        datasets = []

        # The filtered versions of the individual dataset
        tmp_dir = os.path.join(self.to_dir, f'.tmp_{name}_renamed')

        # The unified version of the filtered individual datasets
        tmp_dir_2 = os.path.join(self.to_dir, f'.tmp_{name}.parquet')

        # If the unified version already exists, then no need to recreate it (otherwise, do so).
        if not os.path.exists(tmp_dir_2):
            if os.path.exists(tmp_dir):
                # Since the Zone unification has already been done, amend the file names to reflect this
                # Replace 'Zone10' or 'Zone11' with 'Zone*'
                files = [re.sub(r'Zone[0-9]{1,2}','Zone*', f) for f in files]
                # Take the unique file names
                files = np.unique(files).tolist()
                # Replace 'Zone*' with 'AllZones'
                files = [re.sub('Zone\*','AllZones', f) for f in files]
                tmp_files = [os.path.join(tmp_dir, os.path.basename(p)) for p in files]
            else:
                # Keep track of the new file names
                tmp_files = []
                os.mkdir(tmp_dir)

                # Keep track of the number of messages that are thrown out for different reasons
                original_len = 0
                total_invalid_mmsis = 0
                total_unwanted_vts = 0
                total_unwanted_statuses = 0
                total_stationary = 0
                total_sog_cog_heading = 0
                total_empirical_speed = 0
                total_short_tracks = 0

                # Condense zones so that they are read in as one (dask can handle a wildcard character)
                files = [re.sub(r'Zone[0-9]{1,2}','Zone*', f) for f in files]
                files = np.unique(files).tolist()

                # Iterate through the files, filtering each one
                for file in files:
                    # Read in the files
                    data = dd.read_parquet(file, engine='pyarrow').compute()
                    gc.collect()

                    # Track the original number of messages
                    original_len += len(data)
                    data = self._clean_column_names(data)
                    data = data.set_index('mmsi')
                    data = self._fill_nas(data)

                    # Remove any messages from vessels with invalid mmsis
                    data, invalid_mmsis = self._remove_invalid_mmsis(data)
                    total_invalid_mmsis += invalid_mmsis

                    # Correct any negatives that can be corrected per MarineCadastre.gov's FAQ
                    data = self._correct_negatives(data)

                    # Remove any messages from vessel types that we aren't using
                    data, unwanted_vts = self._remove_unwanted_vessel_types(data)
                    total_unwanted_vts += unwanted_vts

                    # Remove any messages that don't have status codes we are interested in
                    data, unwanted_statuses = self._remove_unwanted_statuses(data)
                    total_unwanted_statuses += unwanted_statuses

                    # Drop columns we aren't using
                    data = self._drop_unwanted_columns(data)
                    data = data.reset_index()

                    # Sort by MMSI and datetime
                    data['base_datetime'] = pd.to_datetime(data['base_datetime'])
                    data = data.sort_values(['mmsi','base_datetime']).reset_index(drop=True)

                    # Remove messages that have invalid values, or where the vessel is stationary or moving too fast
                    data, stationary, sog_cog_heading, empirical_speed = self._remove_invalid_messages(data, trajectories_are_complete=False)
                    total_stationary += stationary
                    total_sog_cog_heading += sog_cog_heading
                    total_empirical_speed += empirical_speed

                    # Remove tracks that cannot possibly be long enough to match requirements specified in the config
                    data = self._create_track_ids(data)
                    specifier = os.path.join(os.path.basename(os.path.dirname(file)), os.path.basename(file))
                    min_time, max_time = get_min_max_times(specifier)
                    data, short_tracks = self._remove_unwanted_tracks(data, trajectories_are_complete=False,
                                                        min_time = min_time, max_time = max_time)
                    total_short_tracks += short_tracks

                    # Drop temporary columns
                    data = data.drop(columns=['new_ship', 'new_track', 'track'])

                    # Save to temporary path
                    fname = os.path.basename(file)
                    fname = re.sub('Zone\*', 'AllZones', fname)
                    tmp_path = os.path.join(tmp_dir, fname)
                    data.to_parquet(tmp_path, engine='pyarrow',index=False)
                    tmp_files += [tmp_path]

                del data
                # Log how messages were removed in total for various reasons
                correct_mmsi_len = original_len - total_invalid_mmsis
                logging.info(f'Dataset started with {original_len:,} messages.')
                logging.info(f'{total_invalid_mmsis:,} messages were dropped because they did not have valid MMSIs '
                             f'({total_invalid_mmsis / original_len * 100:0.3}%). Dataset now contains {correct_mmsi_len:,} '
                             f'messages.')
                wanted_vt_len = correct_mmsi_len - total_unwanted_vts
                logging.info(f'{total_unwanted_vts:,} messages removed ({total_unwanted_vts / correct_mmsi_len * 100:0.3}%) '
                             f'that did not have vessel type group: {config.vessel_types}. Dataset now contains '
                             f'{wanted_vt_len:,} messages.')
                wanted_status_len = wanted_vt_len - total_unwanted_statuses
                logging.info(f'{total_unwanted_statuses:,} messages removed ({total_unwanted_statuses / wanted_vt_len * 100:0.3}%) '
                             f'that did not have statuses: {config.desired_statuses}. Dataset now '
                             f'contains {wanted_status_len:,} messages.')
                have_moved = wanted_status_len - total_stationary
                logging.info(f'{total_stationary:,} messages removed ({total_stationary / wanted_status_len * 100:0.3}%) '
                             f'because they did not move since the previous message. Another pass at removing '
                             f'stationary ships will be made later, once all datasets are joined and can be '
                             f'processed in unison. Dataset now contains {have_moved:,} messages.')
                good_sog_cog_heading = have_moved - total_sog_cog_heading
                logging.info(f'{total_sog_cog_heading:,} messages removed ({total_sog_cog_heading / have_moved * 100:0.3}%) '
                             f'that did not have valid SOGs, COGs, or headings. Dataset now '
                             f'contains {good_sog_cog_heading:,} messages.')
                good_empircal_speed = good_sog_cog_heading - total_empirical_speed
                logging.info(f'{total_empirical_speed:,} messages removed ({total_empirical_speed / good_sog_cog_heading * 100:0.3}%) '
                    f'that did not have valid empirical speeds. Another pass at removing invalid empirical speeds '
                             f'will be made later, once all datasets are joined and can be processed in unison. '
                             f'Dataset now contains {good_empircal_speed:,} messages.')
                good_or_unknown_track_length = good_empircal_speed - total_short_tracks
                logging.info(f'{total_short_tracks:,} messages removed ({total_short_tracks / good_empircal_speed * 100:0.3}%) '
                    f'because they were apart of trajectories that were known to be shorter than '
                             f'{config.min_track_length / 60 /60} hours. Another pass at removing short trajectories '
                             f'will be made later, once all datasets are joined and can be processed in unison. '
                             f'Dataset now contains {good_or_unknown_track_length:,} messages.')

            gc.collect()

            # Read filtered files from temp paths into a single dataframe
            dataset = dd.read_parquet(tmp_files)
            if args.debug:
                npartitions = 2
            else:
                npartitions = os.cpu_count()*6

            # Sort by MMSI
            dataset = dataset.set_index('mmsi',npartitions=npartitions)
            # If the above uses up too much memory and gets killed, you can try using the below line instead.
            # It may be a good deal slower, but hopefully won't eat up as much RAM
            # dataset = dataset.set_index('mmsi', npartitions=npartitions, shuffle='disk')

            # Partitions will be split by MMSI, so the below just further sorts them by time
            dataset = dataset.map_partitions(func = (lambda p: p.sort_values(['mmsi','base_datetime'])),
                                             meta = dataset.partitions[0].compute())

            # Save sorted df to temp path
            dd.to_parquet(dataset, tmp_dir_2, schema='infer')
            clear_path(tmp_dir)
            del dataset
            gc.collect()

        # Load in dataset
        dataset = dd.read_parquet(tmp_dir_2)

        return dataset

    def _clean_column_names(self, dataset):
        """
        Clean column names

        Rename to snake case. There is also a typo in some of the MarineCadastre.gov's raw data files (a misspelling of
        transceiver), which this corrects.

        :param dataset: Dataset to rename
        :return: Dataset with renamed columns
        """
        dataset = dataset.rename(columns={'TranscieverClass': 'TransceiverClass', 'BaseDateTime': 'BaseDatetime'})
        dataset.columns = dataset.columns.map(to_snake_case)
        return dataset

    def _remove_invalid_mmsis(self, partition):
        """
        Remove messages with invalid MMSIs

        For the MMSI to be valid, it must have 9 digits, and its first three digits must be between 201 and 775
        (inclusive), per MMSI page from the Coast Guard's Navigation Center, a copy of which is stored in the
        resources_and_information directory

        There are other MMSIs for other AIS transceivers that we aren't interested in, e.g. land-based radio stations or
        search and rescue aircraft. These are filtered out.

        :param dataset_name: Whether this is the test or training set
        :return: Dataset with invalid MMSIs missing
        """
        original_len = len(partition)
        partition = partition[partition.index.str.len() == 9]

        # The first three digits of the MMSI must be between 201 and 775 (inclusive)
        first_three = partition.index.str.slice(0, 3).astype(int)
        partition = partition[
            (first_three >= 201) &
            (first_three <= 775)
            ]
        num_invalid_mmsis = original_len - len(partition)
        return partition, num_invalid_mmsis


    def _correct_negatives(self, partition):
        """
        Correct negative SOG and COG values, as specified in the MarineCadastre.gov FAQ

        :param dataset_name: Whether this is the test or training set
        :return: Dataset with negatives corrected
        """

        # Per MarineCadastre.gov FAQ, SOG values less than 0 can be corrected for by adding 102.6
        sog_correction = np.where(partition['sog'] < 0,
                                  102.6,
                                  0)
        partition['sog'] += sog_correction

        # Per MarineCadastre.gov FAQ, COG values less than 0 can be corrected for by adding 409.6
        cog_correction = np.where(partition['cog'] < 0,
                                  409.6,
                                  0)
        partition['cog'] += cog_correction
        return partition

    def _fill_nas(self, partition):
        """
        Replace heading values of 511 with NAs, per MarineCadastre.gov's FAQ

        :param partition: Dataset to process
        :return: Processed dataset
        """

        # Replace heading values that are equal to 511, as this is the code they use for unknown values, per the marine
        # cadastre FAQ
        partition['heading'] = partition['heading'].replace(511, np.nan)

        return partition

    def _remove_invalid_messages(self, partition, trajectories_are_complete = True):
        """
        Remove messages that fail to meet various conditions

        To be kept in the dataset, the message must:
            1) Not be at the exact same lat/long as the previous message
            2) Have an SOG that is below the cutoff defined in config.sog_cutoff
            3) Have a Heading that is between 0 and 360
            4) Have a COG that is between 0 and 360
            5) Have an empirical speed that is below the cutoff defined by config.empirical_speed_cutoff and above 0.01

        The empirical speed step is a bit more involved than the others. First of all it is done iteratively, because
        removing a message for having an invalid empirical speed may result in the surrounding messages having invalid
        empirical speeds the next round. To be removed from the dataset for having too large of an empirical speed, the
        empirical speeds between the message and *both* its preceding and following messages must be too large. This is
        with the idea that if only one of these is too large, then we don't know whether the first message or the second
        message has the error. Alternatively, the empirical speed can also be too small (less than 0.01 knots),
        indicating the ship was standing still.

        Even if trajectories_are_complete is set to False, the dataset must contain a contiguous set of messages - i.e.
        there cannot be chunks missing from the middle. For example if the trajectory contains messages
        1 2 3 4 5 6 7 8, this function will not work properly if the partition contains 1 3 4 6 8, but it will work
        properly if the trajectory contains 1 2 3 4 or 3 4 5 6 7.


        :param partition: Whether this is the train or test set
        :param trajectories_are_complete: Whether this partition contains all message for a trajectory, or if it's
                                          possible only some of the messages in a trajectory are contained. This is
                                          relevant when we are processing files one at a time, since in that case
                                          trajectories may be split across different files.
        :return: The dataset with invalid messages removed
        """
        if partition.index.name == 'mmsi':
            mmsi_index = True
            partition = partition.reset_index()
        else:
            mmsi_index = False

        # Create an indicator for whether this is a new ship
        mmsi = partition['mmsi'][1:].reset_index(drop=True)
        prev_mmsi = partition['mmsi'][:-1].reset_index(drop=True)
        new_ship = mmsi != prev_mmsi
        new_ship = pd_append([True, new_ship])
        partition['new_ship'] = new_ship
        del new_ship, mmsi, prev_mmsi

        # See if, as of right now, this timestamp would be the start of a new track anyway
        # (Since even if the ship hasn't moved since the last timestamp, this is part of a different track, and thus
        # offers information to the new track)
        prev_timestamp = partition['base_datetime'][:-1].reset_index(drop=True)
        timestamp = partition['base_datetime'][1:].reset_index(drop=True)
        prev_time_gap = (timestamp - prev_timestamp).dt.total_seconds()
        this_is_start_of_new_track = partition['new_ship'] | pd_append(
            [True, (prev_time_gap > config.new_trajectory_time_gap)])

        original_len = len(partition)
        # Filter out messages where ship hasn't moved
        lat_lon = partition[['lat', 'lon']][1:].reset_index(drop=True)
        prev_lat_lon = partition[['lat', 'lon']][:-1].reset_index(drop=True)
        has_moved = (lat_lon - prev_lat_lon).abs().sum(axis=1) != 0
        has_moved = pd_append([False, has_moved])
        partition = partition[has_moved | this_is_start_of_new_track].reset_index(drop=True)
        del this_is_start_of_new_track, prev_time_gap, timestamp, prev_timestamp, lat_lon, prev_lat_lon, has_moved
        stationary_ships = original_len - len(partition)

        # Filter out bad SOGs
        valid_sog = (partition['sog'] <= config.sog_cutoff)
        partition = partition[valid_sog].reset_index(drop=True)
        del valid_sog

        # Filter out bad headings
        valid_heading = (
                ((partition['heading'] <= 360) &
                 (partition['heading'] >= 0)) |
                partition['heading'].isna()
        ).reset_index(drop=True)
        partition = partition[valid_heading].reset_index(drop=True)
        del valid_heading

        # Filter out bad COGs
        valid_cog = (
                (partition['cog'] <= 360) &
                (partition['cog'] >= 0)
        ).reset_index(drop=True)
        partition = partition[valid_cog].reset_index(drop=True)
        del valid_cog

        bad_sog_cog_or_heading = original_len - len(partition) - stationary_ships

        # this is done iteratively as it may be the case that when an outlier message is removed, the message before
        # or message after now becomes an outlier message as well
        previous_len = -1
        while len(partition) != previous_len:
            previous_len = len(partition)

            # Calculate how fast ship was going
            prev_lat_lon = partition[['lat', 'lon']][:-2].reset_index(drop=True)
            lat_lon = partition[['lat', 'lon']][1:-1].reset_index(drop=True)
            next_lat_lon = partition[['lat', 'lon']][2:].reset_index(drop=True)
            prev_nm_traveled = haversine_vector(prev_lat_lon, lat_lon, unit=Unit.NAUTICAL_MILES)
            next_nm_traveled = haversine_vector(lat_lon, next_lat_lon, unit=Unit.NAUTICAL_MILES)
            del prev_lat_lon, next_lat_lon, lat_lon

            prev_time_gap = (partition['base_datetime'][1:-1].reset_index(drop=True)
                             - partition['base_datetime'][:-2].reset_index(drop=True)).dt.total_seconds() / 60 / 60 # in hours
            next_time_gap = (partition['base_datetime'][2:].reset_index(drop=True)
                             - partition['base_datetime'][1:-1].reset_index(drop=True)).dt.total_seconds() / 60 / 60 # in hours

            prev_empirical_knots = pd_append([0, prev_nm_traveled / prev_time_gap, 0])
            next_empirical_knots = pd_append([0, next_nm_traveled / next_time_gap, 0])
            del prev_nm_traveled, next_nm_traveled

            prev_time_gap = pd_append([0, prev_time_gap, 0])
            next_time_gap = pd_append([0, next_time_gap, 0])

            new_ship = partition['mmsi'][1:].reset_index(drop=True) != partition['mmsi'][:-1].reset_index(drop=True)
            new_ship = pd_append([True, new_ship])
            partition['new_ship'] = new_ship

            this_is_first_ts = partition['new_ship'] | (prev_time_gap > (config.new_trajectory_time_gap / 60 / 60))
            this_is_last_ts = pd_append([partition['new_ship'][1:], True]) | (
                    next_time_gap > (config.new_trajectory_time_gap / 60 / 60))
            del prev_time_gap, next_time_gap

            # Remove if the empirical speed between this and the previous message
            # was > config.empirical_speed_cutoff knots AND so is the empirical speed to the next message, since it
            # is more likely to be an outlier if it's different from both surrounding timestamps, rather than just one
            # Or if the ship has only moved a very small amount, to remove trajectories where the ship is basically
            # anchored
            if trajectories_are_complete:
                valid_speed = ~(
                        (
                                (
                                        (prev_empirical_knots > config.empirical_speed_cutoff) | this_is_first_ts)
                                        & ((next_empirical_knots > config.empirical_speed_cutoff) | this_is_last_ts)
                        ) |
                        ((prev_empirical_knots < 0.01) & ~(this_is_first_ts | this_is_last_ts))
                )
            else:
                # If the trajectories aren't full yet, this means that a message having a true value for
                # this_is_first_ts or this_is_last_ts does not necessarily mean it was the first or last timestamp
                # (since in reality there may have been a ts before/after it). This means that being the first/last ts
                # here isn't sufficient for making one of the two invalid empirical speeds that a message
                # must have to be considered invalid overall. (If it is the case that the message actually *was* the
                # first/last, this will be caught later when the full trajectories are run through all together).
                # The second usage of this_is_first_ts/this_is_last_ts is okay to keep in, because it is making sure
                # first_ts/last_ts messages are considered valid, which is what we want in the first past.
                valid_speed = ~(
                        (
                                (prev_empirical_knots > config.empirical_speed_cutoff)
                                & (next_empirical_knots > config.empirical_speed_cutoff)
                        ) |
                        ((prev_empirical_knots < 0.01) & ~(this_is_first_ts | this_is_last_ts))
                )

            del prev_empirical_knots, next_empirical_knots
            partition = partition[valid_speed].reset_index(drop=True)

        if mmsi_index:
            partition = partition.set_index('mmsi')

        bad_empirical_speed = original_len - len(partition) - bad_sog_cog_or_heading - stationary_ships
        if trajectories_are_complete:
            return partition
        else:
            return partition, stationary_ships, bad_sog_cog_or_heading, bad_empirical_speed

    def _remove_unwanted_statuses(self, partition):
        """
        Remove messages with statuses we did not consider.

        The statuses we used were 'under way sailing' and 'under way using engine'. We also kept in null statuses
        and statuses that where the vessel was being pushed/pulled.

        This means we removed messages if the vessel was fishing, if it was moored/at anchor, etc.

        Because statuses are in the MarineCadastre's dataset as a code, this uses the mapping in
        config/navigation_statuses.csv

        :param dataset_name: Dataset to process
        :return: The dataset with unwanted messages removed
        """
        original_len = len(partition)

        # Replace text status with numeric ones
        partition = partition.copy()
        replacement_dict = {L.lower(): S for i, (S, L) in config.statuses.iterrows()}
        partition['status'] = partition['status'].fillna('-1').str.lower().replace(replacement_dict).astype(int)

        # Filte to only statuses we are interested in
        desired_statuses_numeric = [replacement_dict[s] for s in config.desired_statuses]
        partition = partition[partition['status'].isin(desired_statuses_numeric)]

        num_unwanted_statuses = original_len - len(partition)
        return partition, num_unwanted_statuses

    def _remove_unwanted_vessel_types(self, partition):
        """
        Remove messages from unwanted vessel types.

        Vessels must be of type Pleasure Craft/Sailing, Tug Tow, Passenger, Cargo, Tanker, or Fishing

        Because vessel types are in the MarineCadastre's dataset as a code, this uses the mapping in
        config/vessel_type_codes.csv

        :param partition: Dataset to process
        :return: The dataset with unwanted messages removed
        """
        original_len = len(partition)

        # Replace text numeric type with text ones
        replacement_dict = {VT: G.lower().replace('/', ' or ') for i, (G, VT, d) in config.types.iterrows()}
        partition['vessel_type'] = partition['vessel_type'].fillna(-1).astype(int).astype(str)
        partition['vessel_group'] = partition['vessel_type'].replace(replacement_dict)

        # Filter out vessel types we don't want
        partition = partition[partition['vessel_group'].isin(config.vessel_types)]

        num_unwanted_vts = original_len - len(partition)
        return partition, num_unwanted_vts

    def _create_track_ids(self, partition):
        """
        Go through dataset and identify trajectories.

        Prior to this, the dataset is sorted by MMSI and message timestamp. A new trajectory ID is created when either
        1) the message has a different MMSI than the previous one, or 2) The new message occurs more than
        config.new_trajectory_time_gap seconds since the last message (i.e. it's been so long since the last message
        that we're considering this a new trajectory).

        To conserve memory this function is called on each partition of the dask dataframe separately, which requires
        it to be smart about creating unique IDs.

        :param partition: Dataset to process
        :return: The dataset, with trajectory_ids
        """
        if partition.index.name == 'mmsi':
            partition = partition.reset_index()
            mmsi_index = True
        else:
            mmsi_index = False
        # Check if it has been more than config.new_trajectory_time_gap seconds
        timestamp = partition['base_datetime'][1:].reset_index(drop=True)
        prev_timestamp = partition['base_datetime'][:-1].reset_index(drop=True)
        time_gap = timestamp - prev_timestamp

        mmsi = partition['mmsi'][1:].reset_index(drop=True)
        prev_mmsi = partition['mmsi'][:-1].reset_index(drop=True)
        new_ship = mmsi != prev_mmsi
        new_ship = pd_append([True, new_ship])
        partition['new_ship'] = new_ship

        new_trajectory_id = partition['new_ship'] | pd_append([
            True,
            (time_gap.dt.total_seconds() > config.new_trajectory_time_gap)
        ])

        partition['new_track'] = new_trajectory_id
        partition['within_mmsi_id'] = partition.groupby('mmsi')['new_track'].apply(np.cumsum)

        # Because track IDs are being created within each partition (without knowledge of the track IDs created in
        # other partitions), we need to make sure that they are not repeated across different partitions. I'm doing this
        # by using the MMSI (which the partitions are split on, meaning a single partition will contain all the messages
        # from the MMSIs it contains) to create the track ids. Each track ID is a combination of the track's MMSI,
        # with a within-MMSI ID. E.g. for a vessel with MMSI '2', the track IDs will be something like '2000', '2001',
        # '2002', etc. (Although in reality they'll be much longer, to make sure there aren't any overlaps.)
        MAX_NUMBER_OF_TRACKS_PER_MMSI = 1000000
        if (partition['within_mmsi_id'] > MAX_NUMBER_OF_TRACKS_PER_MMSI).any():
            raise ValueError(f'A MMSI has resulted in more than {MAX_NUMBER_OF_TRACKS_PER_MMSI:,} tracks. Please ' 
                             f'increase the MAX_NUMBER_OF_TRACKS_PER_MMSI constant in order to guarantee that all '
                             f'track IDs are unique')
        partition['track'] = partition['mmsi'].astype(int) * MAX_NUMBER_OF_TRACKS_PER_MMSI + partition['within_mmsi_id']
        partition = partition.drop(columns = ['within_mmsi_id'])

        if mmsi_index:
            partition = partition.set_index('mmsi')

        return partition

    def _remove_unwanted_tracks(self, partition, trajectories_are_complete=True, min_time = None, max_time = None):
        """
        Remove tracks that do not meet the minimum time length requirement

        :param partition: Dataset to process
        :param trajectories_are_complete: Whether the dataset contains all messages from the trajectories, or if it is
                                          possible that portions of the trajectories are contained in other datasets
        :param min_time: If the trajectories in this dataset are not complete, then the time bound (minimum) for
                         this dataset
        :param max_time: If the trajectories in this dataset are not complete, then the time bound (maximum) for
                         this dataset
        :return: The dataset, with short tracks removed
        """

        aggregate = partition.groupby('track').agg({'base_datetime': ['min', 'max']})
        aggregate['length'] = aggregate[('base_datetime', 'max')] - aggregate[('base_datetime', 'min')]

        if trajectories_are_complete:
            long_tracks = aggregate.index[aggregate['length'].dt.total_seconds() > config.min_track_length]
            partition = partition[partition['track'].isin(long_tracks)]
            return partition
        else:
            original_len = len(partition)
            time_since_beginning_of_dataset = aggregate[('base_datetime', 'min')] - min_time
            time_until_end_of_dataset = max_time - aggregate[('base_datetime', 'max')]

            trajectory_could_be_incomplete = (
                    (time_since_beginning_of_dataset.dt.total_seconds() <= config.new_trajectory_time_gap)
                    | (time_until_end_of_dataset.dt.total_seconds() <= config.new_trajectory_time_gap))
            long_or_unknown_tracks = aggregate.index[
                (aggregate['length'].dt.total_seconds() > config.min_track_length)
                | trajectory_could_be_incomplete
            ]
            partition = partition[partition['track'].isin(long_or_unknown_tracks)]
            short_tracks = original_len - len(partition)
            return partition, short_tracks

    def _process_partition(self, partition):
        """
        Once the initial filtering pass has been performed, and the partitions have been sorted and split by MMSI,
        perform all the necessary steps that can be done at the partition level

        :param partition: The partition to process
        :return: The processed partition
        """
        partition = self._remove_invalid_messages(partition)
        partition = self._create_track_ids(partition)
        partition = self._remove_unwanted_tracks(partition)
        partition = partition.drop(['new_track', 'new_ship'], axis=1)
        return partition

    def clean_split_and_save(self):
        """
        After the initial filtering/cleaning/sorting have been been done, perform the rest of the
        cleaning on the full dataset

        :return:
        """
        for dataset_name in ['test', 'train']:
            logging.info(f'{dataset_name} dataset is starting with {len(self.datasets[dataset_name]):,} messages.')
            partition = self.datasets[dataset_name]._partitions(0).compute()
            output_meta = self._process_partition(partition)
            self.datasets[dataset_name] = self.datasets[dataset_name].map_partitions(self._process_partition,
                                                                                     meta=output_meta)
            self.datasets[dataset_name] = self.datasets[dataset_name].reset_index().set_index('track', sorted=True)

            if dataset_name == 'test':
                self._save_dataset(dataset_name)
            elif dataset_name == 'train':
                self.split_train_validation()
                self._save_dataset('valid')
                self._save_dataset('train')

        clear_path(os.path.join(self.to_dir, '.tmp_test.parquet'))
        clear_path(os.path.join(self.to_dir, '.tmp_train.parquet'))

    def split_train_validation(self):
        """
        Randomly sample trajectories from the training set to become apart of the validation set.

        :return:
        """
        tracks = self.datasets['train'].index.unique().compute()
        tracks = np.random.choice(tracks, len(tracks), replace=False) # Reorder

        validation_size = int(self.validation_fraction * len(tracks))
        logging.info(f'Using {validation_size:,}/{len(tracks):,} tracks for the validation set'
                     f' ({validation_size / len(tracks) * 100:0.4}%)')
        validation_tracks = tracks[:validation_size]
        validation_tracks = np.sort(validation_tracks)
        train_tracks = tracks[validation_size:]
        train_tracks = np.sort(train_tracks)

        self.datasets['valid'] = self.datasets['train'].loc[validation_tracks]
        self.datasets['train'] = self.datasets['train'].loc[train_tracks]

    def _save_dataset(self, dataset_name):
        """
        Save  to disk

        Each dataset is saved as a series of parquet files, which can later be read by dask for multiprocessing

        :return:
        """

        out_path = os.path.join(self.to_dir, f'{dataset_name}.parquet')
        self.current_file = out_path
        clear_path(out_path)
        dd.to_parquet(self.datasets[dataset_name], out_path, schema='infer')
        logging.info(f'{dataset_name} set saved to {out_path}')
        self.current_file = None
        self.datasets[dataset_name] = dd.read_parquet(out_path)
        logging.info(f'{dataset_name} ended with {len(self.datasets[dataset_name]):,} messages')
        del self.datasets[dataset_name]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_name', choices=datasets.keys())

    # How to split between test/train/validation
    parser.add_argument('--test_fraction', '-tf', default=1 / 5, type=float,
                        help='Percentage of months to use for test set. Will take the most recent months for this set')
    parser.add_argument('--validation_fraction', '-vf', default=1 / 5, type=float,
                        help='Percentage of the non-test tracks to use for validation. Randomly sampled')
    parser.add_argument('--seed', type=int,
                        help='Seed to use when randomly splitting up training and validation dataset_names.')

    # Logging
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-l', '--log_level', type=int,
                        default=2, choices=[0, 1, 2, 3, 4],
                        help='Level of logging to use')
    parser.add_argument('-s', '--save_log', action='store_true')
    parser.add_argument('--memory',type=str,choices=['conserve',None],default=None)

    args = parser.parse_args()

    config.dataset_config = datasets[args.dataset_name]

    config.set_log_level(args.log_level)

    dask.config.set(scheduler='single-threaded')

    cleaner = Cleaner(args.test_fraction, args.validation_fraction)
    cleaner.load()
    cleaner.clean_split_and_save()
