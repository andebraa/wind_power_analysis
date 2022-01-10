"""
Script that loops over the places_norway list of places and call nominatim
for the longitude and latitude of these places. writes to list which
is then used to locate tweets in the dataset
"""
import pandas as pd
import numpy as np

places = pd.read_csv('data/places_norway.csv')
