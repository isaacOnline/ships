## RUN AFTER dataset_config.py HAS BEEN UPDATED:
python downloader.py california_coast -l 4 -s &&
python cleaner.py california_coast -l 4 -s --seed 47033218 --memory conserve &&
python interpolator.py california_coast -l 4 -s &&
python current_downloader.py california_coast -l 4 -s &&
python current_aggregator.py california_coast -l 4 -s &&
python destination_appender.py california_coast -l 4 -s

## RUN ONLY AFTER SETTING DBSCAN PARAMETER VALUES
#python destination_appender.py california_coast -l 4 -s &&
#python current_appender.py california_coast -l 4 -s &&
#python sliding_window.py california_coast -l 4 -s &&
#python formatter.py california_coast -l 4 -s