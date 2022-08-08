import logging

import pandas as pd

# Set a base directory to use for data storage. You should change this value.
global data_directory
data_directory = '/home/isaac/data/'

global dataset_config

# Define start/end years to look at. Currently, the earliest supported year is 2015. Data prior to 2015 uses a different
# url and will also need to be preprocessed slightly differently - check the ais_data_faq_from_marine_cadastre.pdf in
# resources_and_information for details.
global start_year
start_year = 2015
assert start_year >= 2015

global end_year
end_year = 2019

global years
years = range(start_year, end_year + 1)

# Url to download data from. This should not be changed.
global base_url
base_url = 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/'

# The length of time between AIS messages for the trajectory to be considered a new one, in seconds. Currently
# set to two hours
global new_trajectory_time_gap
new_trajectory_time_gap = 120 * 60

# The maximum sog that a message can have without being removed from the dataset
global sog_cutoff
sog_cutoff = 30

# The maximum empirical speed that a message can have without being removed from the dataset, in knots
global empirical_speed_cutoff
empirical_speed_cutoff = 40

# The number of seconds between timestamps when interpolating. Currently set to 5 minutes
global interpolation_time_gap
interpolation_time_gap = 5 * 60

# Number of *timestamps* to use for prediction, and to predict into the future. The current setting uses three hours of
# history and predicts three hours into the future
global length_of_history
length_of_history = int(3 * 60 * 60 / interpolation_time_gap) + 1

global length_into_the_future
length_into_the_future = int(3 * 60 * 60 / interpolation_time_gap) - 1

# The length of time needed for a track to be kept in the dataset, in seconds. Should not be edited directly
global min_track_length
min_track_length = (length_of_history + length_into_the_future) * interpolation_time_gap

# The vessel groups to keep in the analysis.
# Other valid vessel types that may be used are 'other' and 'military'
global vessel_types
vessel_types = [
    'cargo',
    'passenger',
    'fishing',
    'tug tow',
    'tanker',
    'pleasure craft or sailing',
]

# Statuses to be kept in the analysis.
# See columns in navigation_statuses.csv for other possible values
global desired_statuses
desired_statuses = [
    'under way sailing',
    'under way using engine',
    'undefined'
]

# The different time gaps to create datasets for. Currently only testing 15 and 30 minute time gaps
global time_gaps
time_gaps = [min * 60 for min in [15, 30]]

# Variables to take from the ocean currents dataset.
# Other possible values are 'salinity' and 'water_temp'. See NOAA website for more.
global currents_variables
currents_variables = ['water_u', 'water_v']



# Categorical columns that need to be one hot encoded. Only change this if you change preprocessing to add in other
# columns.
global categorical_columns
categorical_columns = ['vessel_group','destination_cluster']

# Used for preprocessing of currents dataset. All other values have been deprecated
global currents_window
currents_window = 'stable'

# Files to use for data cleaning. Do not change.
global statuses
statuses = pd.read_csv('config/navigation_statuses.csv')
global types
types = pd.read_csv('config/vessel_type_codes.csv')

# Function for setting the log level
def set_log_level(level):
    global log_level
    if level == 0:
        log_level = logging.CRITICAL
    elif level == 1:
        log_level = logging.ERROR
    elif level == 2:
        log_level = logging.WARNING
    elif level == 3:
        log_level = logging.INFO
    elif level == 4:
        log_level = logging.DEBUG
