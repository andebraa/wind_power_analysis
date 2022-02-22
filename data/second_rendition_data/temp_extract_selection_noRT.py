import pandas as pd


data = pd.read_csv('second_rendition_geolocated_noemoji_anonymous.csv') 

no_rt = data.loc[~data.text.str.contains('RT')] 

selection = no_rt.sample(n=10000)

selection.to_csv('natalya_noRT_10000.csv')
