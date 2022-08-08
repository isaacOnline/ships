from socket import gethostname
import os

global dataset_config

# Define start/end years to look at. Currently, the earliest supported year is 2015. Data prior to 2015 uses a different
# url and will also need to be preprocessed slightly differently - check the ais_data_faq_from_marine_cadastre.pdf in
# resources_and_information for details.
global start_year
start_year = 2015

global end_year
end_year = 2019

global years
years = range(start_year, end_year + 1)

# The number of seconds between timestamps when interpolating. Currently set to 5 minutes
global interpolation_time_gap
interpolation_time_gap = 5 * 60

# Number of *timestamps* to use for prediction, and to predict into the future. The current setting uses three hours of
# history and predicts three hours into the future
global length_of_history
length_of_history = int(3 * 60 * 60 / interpolation_time_gap) + 1

global length_into_the_future
length_into_the_future = int(3 * 60 * 60 / interpolation_time_gap) - 1

# Set a base directory to use for data storage. You should change this value.
global data_directory
data_directory = '/home/isaac/data/'

global box_and_year_dir

# Name of dataset being used
global dataset_name
dataset_name = 'formatted_with_currents_stride_3'


# Whether logging should be used
global logging
logging = True

# The host machine
global machine
host = gethostname()

# Categorical columns that are one hot encoded. Only change this if you change preprocessing to add in other
# columns.
global categorical_columns
categorical_columns = ['vessel_group','destination_cluster']