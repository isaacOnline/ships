import logging
import os
import signal

from abc import ABC

from config import config
from utils import get_zones_from_coordinates


class ProcessingStep(ABC):
    """
    Base class for members in the processing directory. Includes member functions for saving data passed in during
    initialization, defining/creating the necessary directories where data will be located, and initializing logging.
    """
    def __init__(self):
        self.lat_min = config.dataset_config.lat_1
        self.lat_max = config.dataset_config.lat_2
        self.lon_min = config.dataset_config.lon_1
        self.lon_max = config.dataset_config.lon_2
        self.current_file = None
        self.zones = get_zones_from_coordinates(config.dataset_config.corner_1, config.dataset_config.corner_2)
        self.datasets = {}

    def _define_directories(self, from_name, to_name):
        """
        Save file paths for from/to directories as member variables.

        Most processing steps take a dataset saved in one location, process them, then save them to another
        location. This function simply saves the paths for the from directory and to directory to the object. (The
        downloaders do not have from directories, as they source the data from NOAA or MarineCadastre.gov.)

        :param from_name: Name of directory that data is coming from
        :param to_name: Name of directory that data is going to
        :return:
        """
        self.box_and_year_dir = os.path.join(
            config.data_directory,
            f'{config.dataset_config.lat_1}_{config.dataset_config.lat_2}_'
            f'{config.dataset_config.lon_1}_{config.dataset_config.lon_2}_'
            f'{config.start_year}_{config.end_year}'
        )
        self.from_dir = os.path.join(self.box_and_year_dir, from_name)
        self.to_dir = os.path.join(self.box_and_year_dir, to_name)
        self.artifact_directory = os.path.join(self.box_and_year_dir, 'artifacts')

        self._create_directories()

    def _create_directories(self):
        """
        Create directory to move data into, if it doesn't already exist.

        Must call _define_directories before _create_directories.

        :return:
        """
        if not os.path.exists(config.data_directory):
            os.mkdir(config.data_directory)
        if not os.path.exists(self.box_and_year_dir):
            os.mkdir(self.box_and_year_dir)
        if not os.path.exists(self.to_dir):
            os.mkdir(self.to_dir)
        if hasattr(self, 'artifact_directory'):
            if not os.path.exists(self.artifact_directory):
                os.mkdir(self.artifact_directory)

    def _initialize_logging(self, save_log=False, log_file_name=None):
        """
        Kick off the logging process

        This adds a logging handler, which will write logging messages to stdout. (If save_log=True, it will also
        write them to disk.) Once this has been run, any other module can import logging, then write messages using
        functions like logging.info, logging.warning, etc.

        If the save_log option is specified, this will to save logs to
         f'{self.box_and_year_directory}/logs/{log_file_name}.log'

        :param save_log: Whether or not the log should be saved to disk
        :param log_file_name: The name of the log file, if the log should be saved to disk
        :return:
        """
        logging_directory = os.path.join(
            self.box_and_year_dir,
            'logs'
        )
        if not os.path.exists(logging_directory):
            os.mkdir(logging_directory)

        # remove any other handlers that have been added by imported python libraries
        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)

        # Reset format
        format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        c_handler = logging.StreamHandler()
        c_handler.setLevel(config.log_level)
        c_handler.setFormatter(format)

        if save_log:
            log_file = os.path.join(logging_directory, f'{log_file_name}.log')
            f_handler = logging.FileHandler(log_file)
            f_handler.setLevel(logging.DEBUG)
            f_handler.setFormatter(format)
            logging.basicConfig(handlers=[c_handler, f_handler], datefmt='%m/%d/%Y %I:%M:%S', level=logging.DEBUG)
            logging.info(f'Logs being saved to {log_file}')
            logging.info('New run beginning')
        else:
            logging.basicConfig(handlers=[c_handler], datefmt='%m/%d/%Y %I:%M:%S', level=logging.DEBUG)

        self._add_exit_handling()

    def _add_exit_handling(self):
        """
        Initialize exit handling.

        If signal is interrupted and run does not complete, then write this to log. Also delete a file that was in the
        middle of being unzipped, if there was one.
        """

        def log_sigint(a, b):
            # this gets fed two inputs when called, neither of which are needed
            if self.current_file is not None:
                if os.path.exists(self.current_file):
                    os.remove(self.current_file)
                logging.error(f'Process ended prematurely by signal interruption. File {self.current_file} '
                              'was being processed when interruption occurred and an attempt to remove '
                              'the incomplete file was made. You may still need to remove it manually.')
            else:
                logging.error(f'Process ended prematurely by signal interruption.')

        signal.signal(signal.SIGINT, log_sigint)

