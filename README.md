# Vessel Trajectory Prediction

## Repository Overview
This repository repository contains code for downloading and processing AIS and weather data, as well as for fitting 
vessel trajectory predictions models.

### Organization
* The [processing](processing) directory contains all python files used for creating a dataset of AIS messages, as well as 
a script for doing so. See [Creating a New Dataset](#creating-a-new-dataset) below for instructions.
* The [tests](tests) directory contains all Python files used for fitting models. See [Fitting and Evaluating 
Models](#fitting-and-evaluating-models) below for more details. 
* The [experiment_scripts](experiment_scripts) directory contains shell scripts that can serve as examples for fitting models.
* The [resources_and_information](resources_and_information) directory contains documents with extra information about the datasets we used. 

### Python Environments
There are two distinct python environments to use with this repository, [one for processing](processing_environment.yml), 
and [one for fitting models](model_fitting_environment.yml).
The yaml specification files for these environments are provided at the root level. It will be necessary to create the 
processing environment from the yaml file yourself if you want to run anything in the processing directory, but if you 
use the below instructions to train models, mlflow will create the model fitting environment for you. 

## Fitting and Evaluating Models 
Before fitting any models, make sure that you have set the paths in the [config file](tests/config/config.py) to point 
to where the data is stored on your machine. 

This project uses [mlflow](https://www.mlflow.org) for model training, saving model artifacts, and storing data 
about their performance. The mlflow library can be installed using conda, and can then be run from the command line. 
The first time you train a model, mlflow will recreate the conda environment that we have been using for model fitting,
which it will then reload when you run models in the future. (If you'd like to recreate this environment 
without mlflow, you can use the model_fitting_environment.yml specifications file).

The script for training and evaluating models can be kicked off using the 'test_time_gaps' mlflow entry point. This is 
done by running the following from the command line. **Other command line arguments will need to be passed in for the 
entry point to work properly. Please use the shell scripts in [experiment_scripts/examples](
experiment_scripts/examples) as templates.**

> mlflow run . -e test_time_gaps 

Mlflow must be kicked off from the project's root directory. Running the command will have mlflow run the 
fit_and_evaluate_model.py script. Provided that the correct preprocesing has been run beforehand, all of the features of 
the experiment can be controlled using command line arguments. Mlflow uses the syntax -P argname=argvalue, for example 
'-P weather=ignore'. For a full list of possible arguments and values, see the 
[TestArgParser](tests/utils/test_arg_parser.py) object in tests/utils, or run the following command from the root 
directory:
> python fit_and_evaluate_model.py -h 

Some arguments are dependent on one another, but the script should warn you if you enter an invalid combination.

## Creating a New Dataset
The first step in creating a new dataset is to add specifications to the 
configuration files found in [processing/config/dataset_config.py](processing/config/dataset_config.py) and 
[tests/config/dataset_config.py](tests/config/dataset_config.py). You will need 
to specify a name for the dataset, the latitude/longitude boundaries, the desired amount of time to move 
the window forward when performing sliding window calculations, and an initial set of min_pts values to try when using
DBSCAN to create destination clusters. See current examples in the *dataset_config.py* files 
for reference. Make sure to include the same information in both of the *dataset_config.py* files. Also make sure to 
update the paths in [processing/config/config.py](processing/config/config.py) and 
[tests/config/config.py](tests/config/config.py)

The second step is to change the dataset names in [process.sh](processing/process.sh), then to run the first section of 
[process.sh](processing/process.sh). Make sure that you have created and activated the processing conda environment before doing so, and that you 
run [process.sh](processing/process.sh) from the processing directory. Once the first section is done, you will then need to specify parameters 
for DBSCAN to use when creating destination clusters (see [Using the Destination 
Appender](#using-the-destination-appender) 
below for details). The final step is to run the second section of [process.sh](processing/process.sh), which will 
calculate the destination clusters and perform the rest of the preprocessing steps.

If you are interested in changing other features of the experiment that aren't defined in dataset_config.py, most 
relevant constants have been defined in [processing/config/config.py](processing/config/config.py) and 
[tests/config/config.py](tests/config/config.py). For example if you
want to change the vessel types to also include military vessels or the maximum SOG value that messages can 
have without getting filtered out, you can do so here. Make sure that these two config files are kept in sync. 

## Using the Destination Appender
Because we did not have access to reliable destination data, we elected to calculate our own destination clusters for 
trajectories using DBSCAN. DBSCAN requires the selection of two parameters: min_pts and eps. I would recommend reviewing
[these](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf?source=post_page) 
[papers](https://dl.acm.org/doi/pdf/10.1145/3068335) for information on how to best chose these parameters. The 
destination appender is set up to create plots to help with the choice, if parameter values have not already 
been selected. 

If you are not interested in using destination data, you can edit the from_dir in 
[current_appender.py](processing/current_appender.py#L22) from "interpolated_with_destination" to "interpolated". You 
can then remove the "destination_appender.py" lines in [process.sh](processing/process.sh). 
This will simply skip over the destination appending step. Otherwise, feel free to follow the steps below for choosing 
the min_pts/eps values for DBSCAN, or to set them on your own. 

1. Before doing anything else, set a range of min_pts values to try in 
[processing/config/dataset_config.py](processing/config/dataset_config.py). I tried the values: 4, 10, 20, 50, 100, 
250, and 500 as defaults, but you may want to switch these if your dataset is of a significantly different size.
2. Run the destination appender script with these values. This will create an elbow plot, (in the *artifacts/
destination_appending* subdirectory of your data directory), which will help you set a range of eps values to try. 
The range of eps values should also be set in [dataset_config.py](processing/config/dataset_config.py). 
3. Once you've selected both parameter ranges, run the destination appender again. This will create plots of a number of 
cluster quality metrics for each of the min_pts/eps values, which you can use to select a min_pts value to use. (Again,
set this in [dataset_config.py](processing/config/dataset_config.py).)
4. Run the destination appender again. This will create plots showing the clusters. Use these plots to 
select a final value of eps to use. 
5. Run the destination appender a final time - this will actually calculate and append the destination values.

## Memory Issues
I encountered a number of memory issues when creating datasets, so the repository is currently optimized to conserve 
memory when possible. (The data processing is fairly slow for this reason.) Despite this, if the desired region or time 
period is large enough, users may still run out of memory and have their scripts killed when creating their own 
datasets. The below summarizes some of the changes you can make to get around these problems.

* If the dataset is too large at the cleaning step, you can try changing the shuffle method that dask uses to 
'disk' [here](processing/cleaner.py#L301). The dataset needs to be sorted by MMSI and timestamp during this step, which 
is what creates the bottleneck. 
* If the dataset is too large at the formatting step, you can try changing the 
[partition size](processing/formatter.py#L254) to a smaller value. This will make the partitions smaller. 
* If the dataset is small enough at the formatting step, you will be able to change the 
[conserve_memory flag](processing/formatter.py#L66) in  *formatter.py* to False, which will significantly speed up the 
final processing step. After the windowing step, the dataset is stored in chunks, and if the conserve_memory flag is set
to True (the default) these chunks will be processed iteratively, saving the processed versions to disk separately. 
Otherwise, the chunks can be processed in parallel, then later combined together. 

 