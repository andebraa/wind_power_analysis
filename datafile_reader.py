import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ast

tweets_data = []
for line in open('havvind_2.json','r'):
    tweets_data.append(json.loads(line))

print(tweets_data[:2])
# print(dictlist)
# df = pd.DataFrame(dictlist, encoding="utf-8")
# print(df.columns)
# print(df)
#print(df['text'])
#
#print(df['created_at'])
