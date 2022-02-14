"""
Pandas often adds an index collumn, which it then can interperet as an unnamed collumn
if it is changed and then written. datasets then end up with multiple unnamed collumns.
This temporary script removed the first few collumnss
"""

import pandas as pd
import re
infile = input('infile: ')

data = pd.read_csv(infile)

print(data.columns)
col = int(input('index of collumn to remove '))

data = data.drop(columns=col)
data.to_csv(infile, index = False)
print(data)             
