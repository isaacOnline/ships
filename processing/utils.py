import os
import re
import shutil
import urllib

import pandas as pd
import utm

from calendar import monthrange
from dateutil import rrule
from datetime import datetime

def get_zones_from_coordinates(corner_1, corner_2):
    """
    Get UTM zones to download, based on lat/lon coordinates

    :param corner_1: Lat/lon pair
    :param corner_2: Lat/lon pair
    :return: range of zones to download
    """
    _, _, zone_1, _ = utm.from_latlon(*corner_1)
    _, _, zone_2, _ = utm.from_latlon(*corner_2)
    if zone_1 > 19:
        raise ValueError(f'Corner 1 {corner_1} is outside data available on MarineCadastre.gov')
    if zone_2 > 19:
        raise ValueError(f'Corner 2 {corner_2} is outside data available on MarineCadastre.gov')
    zones_to_download = range(min(zone_1, zone_2), max(zone_1, zone_2) + 1)
    return zones_to_download


def get_file_specifier(year, month, zone_or_day, extension):
    """
    Get the specific file name for this year, month, and zone or day

    Files from 2017 and prior are split by utm zone, while files from 2018 and on are split by day.

    This unfortunately means that all 2018 AIS messages need to be downloaded.

    :param year:
    :param month:
    :param zone:
    :param extension:
    :return:
    """
    if year in (2015, 2016, 2017):
        specifier = f'AIS_{year}_{month:02d}_Zone{zone_or_day:02d}.{extension}'
    elif year in (2018, 2019, 2020, 2021):
        specifier = f'AIS_{year}_{month:02d}_{zone_or_day:02d}.{extension}'
    else:
        raise ValueError(f"I'm not sure how to format the specifier for year {year}; "
                         f"Check https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/index.html to edit "
                         f"me to do so, and make sure to also edit get_info_from_specifier()")

    specifier = urllib.parse.urljoin(f'{year}/', specifier)
    return specifier


def get_info_from_specifier(file_name):
    """
    Split the file specifier into its constituent info

    The file specifier contains the year, month, zone/day, and file extension for the file in question. This splits up
    a file specifier into these parts. Whether the third piece of information is the zone or day depends on what year
    the file is from (2015-2017 will contain the zone, will 2018+ will contain the day, as this is how the files are
    organized on MarineCadastre.gov).

    :return: year, month, zone or day, extension
    """
    split = re.search('[0-9]{4}.+AIS_([0-9]{4})_([0-9]{2})_(Zone)?([0-9]{2}|\*)\.(.+)', file_name)
    if split:
        year = split.group(1)
        month = split.group(2)
        zone_or_day = split.group(4)
        extension = split.group(5)
    else:
        raise ValueError('This file does not have a known specifier format; the year, month, zone/day, and extension '
                         'cannot be found')

    return year, month, zone_or_day, extension


def all_specifiers(zones, years, extension, dir=None):
    """
    Get all file specifiers for the relevant zones and years

    A specifier is a string formatted something like '2017/AIS_2017_01_Zone01.zip'

    :param zones: The UTM zones to look at
    :param years: The years to consider
    :param extension: The file extension to use for the specifier
    :param dir: The directory, if one is desired at the start of the specifier,
    :return: paths
    """
    specifiers = []
    if dir is not None:
        paths = []
    for year in years:
        if year in (2015, 2016, 2017):
            for month in range(1, 13):
                for zone in zones:
                    specifier = get_file_specifier(year, month, zone, extension)
                    specifiers.append(specifier)

                    if dir is not None:
                        path = os.path.join(dir, specifier)
                        paths.append(path)
        elif year in (2018, 2019, 2020, 2021):
            for dt in rrule.rrule(rrule.DAILY,
                                  dtstart=datetime.strptime(f'{year}-01-01', '%Y-%m-%d'),
                                  until=datetime.strptime(f'{year}-12-31', '%Y-%m-%d')):
                specifier = get_file_specifier(dt.year, dt.month, dt.day, extension)
                specifiers.append(specifier)

                if dir is not None:
                    path = os.path.join(dir, specifier)
                    paths.append(path)

    if dir is not None:
        all_zym = {'paths': paths, 'specifiers': specifiers}
    else:
        all_zym = {'specifiers': specifiers}

    return all_zym


def pd_append(values):
    """
    Append values together into a pandas series

    values should be a list of different things to append together, e.g. the first item might be the integer, the second
    a pd.Series of integers.

    :param values: A list of different things to append together
    :return:
    """
    v1 = values[0]
    if len(values) > 2:
        if type(v1) == pd.Series:
            series = pd.concat([
                v1,
                pd_append(values[1:])
            ]).reset_index(drop=True)
        else:
            series = pd.concat([
                pd.Series([v1]),
                pd_append(values[1:])
            ]).reset_index(drop=True)
    elif len(values) == 2:
        v2 = values[1]
        if type(v1) == pd.Series:
            series = pd.concat([
                v1,
                pd.Series([v2])
            ]).reset_index(drop=True)
        elif type(v2) == pd.Series:
            series = pd.concat([
                pd.Series([v1]),
                v2
            ]).reset_index(drop=True)
        else:
            series = pd.concat([
                pd.Series([v1]),
                pd.Series([v2])
            ]).reset_index(drop=True)
    return series


def to_snake_case(name):
    """
    Convert a string to snake case

    :param name: Name to convert
    :return: Converted name
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


def clear_path(path):
    """
    Delete any files or directories from a path

    :param path: Path to remove
    :return:
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)

def get_min_max_times(specifier):
    """
    Get the first/last possible time for AIS messages contained in a file

    :param specifier: file information
    :return:
    """
    year, month, zone_or_day, extension = get_info_from_specifier(specifier)
    year = int(year)
    month = int(month)

    if year in (2015, 2016, 2017):
        min_time = pd.to_datetime(f'{year}-{month}-01 00:00:00')
        _, last_day = monthrange(year, month)
        max_time = pd.to_datetime(f'{year}-{month}-{last_day} 23:59:59')


    elif year in (2018, 2019, 2020, 2021):
        day = zone_or_day
        min_time = pd.to_datetime(f'{year}-{month}-{day} 00:00:00')
        max_time = pd.to_datetime(f'{year}-{month}-{day} 23:59:59')

    else:
        raise ValueError('Year unaccounted for')

    return min_time, max_time
