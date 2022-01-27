"""
Pandas often adds an index collumn, which it then can interperet as an unnamed collumn
if it is changed and then written. datasets then end up with multiple unnamed collumns.
This temporary script removed the first few collumnss
"""

import pandas as pd
import re

data = pd.read_csv('second_rendition_data/second_rendition_geolocated_noemoji_anonymous.csv')

data = data.drop(columns=data.columns[:2])
data.to_csv('second_rendition_data/second_rendition_geolocated_noemoji_anonymous.csv', index = False)
print(data)
~             
