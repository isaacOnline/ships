import argparse
import logging
import os
import shutil
import urllib
import urllib.error
import urllib.request
import zipfile
import urllib3
import pandas as pd

from config.dataset_config import datasets
from config import config
from processing_step import ProcessingStep
from utils import all_specifiers


class Downloader(ProcessingStep):
    """
    Class for downloading AIS data files from MarineCadastre.gov

    The AIS messages will likely be split across multiple files (based on the region and time period). This class
    will first identify the necessary files to download for the specified region/time. For each file, it will then
    retrieve the raw file from the MarineCadastre.gov, unzip the file, filter it to the correct region, and delete
    the raw/unzipped files. (The deletion is done in order to save space.)
    """
    def __init__(self):
        super().__init__()
        self._define_directories(
            from_name=None,
            to_name='downloads'
        )
        self._initialize_logging(args.save_log, 'download')

        # Set types for columns
        self.dtypes = {
            'MMSI': 'object',
            'BaseDateTime': 'object',
            'LAT': 'float64',
            'LON': 'float64',
            'SOG': 'float64',
            'COG': 'float64',
            'Heading': 'float64',
            'VesselName': 'object',
            'IMO': 'object',
            'CallSign': 'object',
            'VesselType': 'float64',
            'Status': 'object',
            'Length': 'float64',
            'Width': 'float64',
            'Draft': 'float64',
            'Cargo': 'float64',
            'TransceiverClass': 'object'
        }

        # For keeping track of how many files were downloaded
        self.original_length = 0
        self.new_length = 0

    def _define_directories(self, from_name, to_name):
        """
        Define the directory that this class will save files to.

        This overrides the ProcessingStep's _define_directories method, because the downloader does not have a
        "from_dir", since it is downloading files from MarineCadastre.gov.

        :param from_name: Should always be None, but included to keep signature in line with ProcessingStep's method
        :param to_name: Should always be 'download', but included to keep signature in line with ProcessingStep's method
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

    def _create_directories(self):
        """
        Create directories to move data into, if they don't already exist.

        Override of ProcessingStep's _create_directories. The difference is that the Downloader
        has two steps to it - downloading and unzipping - and it creates subdirectories in its to_dir for
        each of these steps.

        :return:
        """
        super()._create_directories()
        for subdir in 'raw', 'unzipped', 'filtered':
            subdir = os.path.join(self.to_dir, subdir)
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            for year in config.years:
                year_dir = os.path.join(subdir, str(year))
                if not os.path.exists(year_dir):
                    os.mkdir(year_dir)

    def _files_left_to_filter(self):
        """
        Determine whether there are any files left to download and filter

        This iterates through all of the file specifiers for the bounding box/set of years, and
        checks if they exist yet.

        If a file was only partially downloaded or unzipped, this issue will NOT be caught here, as the method only
        checks for the file's existence.

        :return: Whether any files still need to be filtered
        """
        dir = 'filtered'
        extension = 'parquet'
        remaining = 0
        total = 0
        dir = os.path.join(self.to_dir, dir)
        to_paths = all_specifiers(self.zones, config.years, extension, dir)['paths']
        for path in to_paths:
            total += 1
            if os.path.exists(path):
                continue
            else:
                remaining += 1
        logging.info(f'{remaining} of {total} files left to be processed into {dir}')
        return remaining > 0

    def _remove_unneeded_downloads(self, specifier):
        """
        Once the file has been filtered to the correct region, delete the raw and unzipped files to save storage space.

        :return:
        """
        for subdir in ['raw', 'unzipped']:
            extension = 'zip' if subdir == 'raw' else 'csv'
            path = os.path.join(self.to_dir, subdir, specifier.replace('zip', extension))
            os.remove(path)
            logging.info(f'File "{specifier.replace("zip", extension)}" removed from "{subdir}" subdirectory to save '
                         f'storage space')

    def download(self):
        """
        Download all relevant datasets from MarineCadastre.gov

        Iterates through each of the files that includes messages for the bounding box/years specified, downloading,
        unzipping, and filtering each one. If there are any errors with downloading, this will retry to download the
        file twice. It will log how many files were downloaded, how many had already been downloaded previously, and
        how many files had issues downloading.

        :return:
        """

        retries = 2
        while self._files_left_to_filter() and (retries > 0):
            # Track different processing errors
            download_unsuccessful = 0
            unzip_unsuccessful = 0
            filter_unsuccessful = 0

            # Track files that were previously processed, in case the downloader has been run before
            total_already_downloaded = 0
            total_already_unzipped = 0
            total_already_filtered = 0

            # Track the work being done during this pass
            total_downloaded_this_round = 0
            total_unzipped_this_round = 0
            total_filtered_this_round = 0

            logging.debug('Starting downloads')

            for specifier in all_specifiers(self.zones, config.years, 'zip')['specifiers']:
                filtered_path = os.path.join(self.to_dir, 'filtered', specifier.replace('zip', 'parquet'))
                already_filtered = os.path.exists(filtered_path)
                if already_filtered:
                    total_already_downloaded += 1
                    total_already_unzipped += 1
                    total_already_filtered += 1
                else:
                    has_been_downloaded = self._download_single_file(specifier)
                    if has_been_downloaded == -1:
                        download_unsuccessful += 1
                    elif has_been_downloaded == 0:
                        total_already_downloaded += 1
                    else:
                        total_downloaded_this_round += 1

                    # If download was successful, or file had previously been downloaded:
                    if has_been_downloaded in [0, 1]:
                        has_been_unzipped = self._unzip_single_file(specifier)
                        if has_been_unzipped == -1:
                            unzip_unsuccessful += 1
                        elif has_been_unzipped == 0:
                            total_already_unzipped += 1
                        else:
                            total_unzipped_this_round += 1

                        # If unzip was successful, or file had previously been unzipped:
                        if has_been_unzipped in [0, 1]:
                            has_been_filtered = self._filter_single_file(specifier.replace('zip','csv'))

                            if has_been_filtered in [-1, -2]:
                                filter_unsuccessful += 1
                            elif has_been_filtered == 0:
                                total_already_filtered += 1
                            elif has_been_filtered == 1:
                                total_filtered_this_round += 1
                                self._remove_unneeded_downloads(specifier)

            # Log how many files were processed
            logging.info(f'{total_downloaded_this_round} files were downloaded this round')
            logging.info(f'{total_already_downloaded} had been downloaded previously')
            if download_unsuccessful > 0:
                logging.warning(f'Retrying downloads for {download_unsuccessful} files that had issues downloading')
            else:
                logging.info(f'{download_unsuccessful} files had issues with downloading.')

            logging.info(f'{total_unzipped_this_round} files were unzipped')
            logging.info(f'{total_already_unzipped} had been unzipped previously')

            if unzip_unsuccessful > 0:
                logging.warning(f'{unzip_unsuccessful} files had unzipping errors')
            else:
                logging.info(f'{unzip_unsuccessful} files had unzipping errors')

            logging.info(
                f'{total_filtered_this_round} files were filtered. {self.new_length:,}/{self.original_length:,} messages in these files were kept')
            logging.info(f'{filter_unsuccessful} had errors being filtered')
            logging.info(f'{total_already_filtered} had been filtered previously')

            retries -= 1

        if retries == 0:
            logging.exception('Some files could not be downloaded. Please check logs.')
        else:
            logging.info(f'All downloads complete for coordinates {config.dataset_config.corner_1}, '
                         f'{config.dataset_config.corner_2}')

    def _download_single_file(self, specifier):
        """
        Download a specific file from MarineCadastre.gov.

        A return of 0 means the file is already present and so an attempt to download was not made. Return of -1 means
        there was an error downloading the file using both urllib and wget. (If there is an error downloading the file
        using urllib, then this will attempt to download the file via the command line and wget.) Return of 1 means the
        file was downloaded successfully.

        :param specifier: The specifier for the file to download. Include the year, e.g. '2018/AIS_2018_01_01.zip'
        :return: exit code
        """
        from_path = urllib.parse.urljoin(config.base_url, specifier)
        to_path = os.path.join(self.to_dir, 'raw', specifier)

        if os.path.exists(to_path):
            logging.debug(f'{specifier} already present')
            return 0
        else:
            logging.debug(f'Retrieving {specifier}')
            self.current_file = to_path
            try:
                self._retrieve(from_path, to_path, 'urllib')
                logging.debug(f'Download {specifier} retrieved')
                self.current_file = None
                return 1
            except urllib.error.ContentTooShortError:
                if os.path.exists(to_path):
                    os.remove(to_path)
                logging.exception(f"Exception occurred with downloading {specifier} via urllib, retrying via wget")
                status = self._retrieve(from_path, to_path, 'wget')
                if status == 0:
                    self.current_file = None
                    return 1
                else:
                    logging.error(f'wget failed to retrieve file {specifier}')
                    self.current_file = None
                    return -1

    def _retrieve(self, url, path, method):
        """
        Download a file from url and save it to path.

        There are multiple options for carrying out the download: urllib, urllib3, and wget. I found that urllib would
        sometimes stop downloading prematurely with specific files, and that using wget on the command line did not have
        this issue.

        :param url: url to download from
        :param path: path to save the file to
        :param method: library to use for performing the download
        :return:
        """
        if method == 'urllib':
            urllib.request.urlretrieve(url, path)
        elif method == 'wget':
            download_directory = os.path.dirname(path)
            return os.system(f'wget {url} -P {download_directory}')
        elif method == 'urllib3':
            http = urllib3.PoolManager()

            with http.request('GET', url, preload_content=False) as r, open(path, 'wb') as out_file:
                shutil.copyfileobj(r, out_file)
            r.release_conn()
        else:
            ValueError('Unknown download method')


    def _unzip_single_file(self, from_name):
        """
        Unzip a single file

        There are a number of outcomes for the unzipping. If the return is 0, this means the file already exists as it
        had been unzipped previously. If the return is 1, this means the file was successfully unzipped. If the return
        is -2, this means the file was never downloaded. If the return is -1, this means the download is not complete,
        so you may want to delete the download and try again.

        The names of zip files downloaded from MarineCadastre.gov have an unneeded base path, which is
        removed here. For example, AIS_2015_01_Zone10.zip contains one file, which is named
        AIS_ASCII_by_UTM_Month/2015/AIS_2015_01_Zone10.csv. Instead of creating the
        AIS_ASCII_by_UTM_Month/2015/ directory, this just unzips the single file as AIS_2015_01_Zone10.csv.

        :param from_name: The name of the file to unzip (should include .zip extension)
        :return: exit code
        """
        # Get from/to paths/names
        from_path = os.path.join(self.to_dir, 'raw', from_name)
        to_path = os.path.join(self.to_dir, 'unzipped', from_name.replace('.zip', '.csv'))
        file_name = os.path.basename(from_path)

        if os.path.exists(to_path):
            logging.debug(f'File {file_name} already unzipped')
            return 0
        else:
            logging.debug(f'Unzipping {file_name}')
            self.current_file = to_path
            try:
                with zipfile.ZipFile(from_path, 'r') as zip_ref:
                    for file_info in zip_ref.infolist():
                        # The files have an unneeded directory when downloaded, which is removed here
                        file_info.filename = os.path.basename(file_info.filename)
                        zip_ref.extract(file_info, os.path.dirname(to_path))
                logging.debug(f'File {file_name} unzipped')
                self.current_file = None
                return 1
            except zipfile.BadZipFile as e:
                logging.exception(f'Exception occurred unzipping {file_name}, moving to next file')
                if os.path.exists(to_path):
                    os.remove(to_path)
                self.current_file = None
                return -1

    def _filter_single_file(self, specifier):
        """
        Filter a single file

        This method will return 1 if the file was filtered, 0 if it had been filtered prior, -1 if the file couldn't
        be found, and -2 if there was a read error

        :param specifier: The name of the file to filter
        :return: exit code
        """
        from_path = os.path.join(self.to_dir, 'unzipped', specifier)
        to_path = os.path.join(self.to_dir,'filtered', specifier.replace('.csv', '.parquet'))

        if os.path.exists(to_path):
            logging.debug(f'File {specifier} already filtered to box {config.dataset_config.corner_1}, '
                          f'{config.dataset_config.corner_2}')
            return 0
        else:
            logging.debug(f'Filtering {specifier} to box {config.dataset_config.corner_1}, '
                          f'{config.dataset_config.corner_2}')
            self.current_file = to_path
            if os.path.exists(from_path):
                try:
                    data = pd.read_csv(from_path, dtype=self.dtypes, low_memory=False)

                    original_len = len(data)
                    self.original_length += original_len

                    data_idx = (
                            (data['LAT'] < self.lat_max) &
                            (data['LAT'] > self.lat_min) &
                            (data['LON'] < self.lon_max) &
                            (data['LON'] > self.lon_min)
                    )
                    data = data[data_idx]

                    data = data.drop(['VesselName', 'IMO', 'CallSign'], axis=1)
                    data = data.sort_values(['MMSI', 'BaseDateTime'])

                    new_len = len(data)
                    self.new_length += new_len

                    data.to_parquet(to_path)

                    logging.debug(f'{specifier} sorted and filtered. {new_len:,}/{original_len:,} messages remain')
                    self.current_file = None

                    return 1
                except FileNotFoundError:
                    logging.exception(f"Exception occurred when reading {specifier}")
                    self.current_file = None
                    return -2
            else:
                logging.error(f'File {specifier} not found')
                self.current_file = None
                return -1


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
