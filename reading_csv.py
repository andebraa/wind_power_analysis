"""
Script for reading csv files made by lese_txt.py. The order of csv collumns is very ridgid.
"""
import re
import json
import csv
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np

test_hist = pd.read_csv('twitterdata.csv', parse_dates=True)

tiems = test_hist['created_at']
tiems= pd.to_datetime(tiems, errors='coerce', format = "%Y-%m-%dT%H:%M:%S.%fZ")
tiems_sorted = np.sort(tiems)

week_nums = pd.date_range(tiems_sorted[0], tiems_sorted[-1], freq='W-MON')
print(week_nums.week)


#test_hist['created_at'].groupby(test_hist["created_at"].dt.week).count()#.plot(kind="bar")

#bins = int(np.ceil(np.log(n)+1))
occurences = np.zeros(len(week_nums))
print(week_nums.week[0])
print('twat')
for i in range(len(tiems)):
    print(int(tiems[i].week) -int(week_nums.week[0]-1))
    occurences[ int(tiems[i].week) -int(week_nums.week[0]) ]  += 1

print(occurences)
print(week_nums)
print(len(occurences))
print(len(week_nums))
plt.plot(week_nums, occurences)

#bins = 30
#print(test_hist['created_at'])
#test_hist['created_at'].hist(bins=week_nums.week)
#plt.hist(test_hist['created_at'])
plt.show()
