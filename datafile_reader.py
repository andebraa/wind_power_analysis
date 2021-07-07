import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ast

dictlist = []
with open('havvind_2.txt') as file:
    for elem in file:

        line_dict = ast.literal_eval(elem)

        dictlist.append(line_dict)



df = pd.DataFrame(dictlist)
print(df.columns)
print(type(df))
print(df['geo'])
print(type(df['geo']))
places=df['place']
print(type(places))
