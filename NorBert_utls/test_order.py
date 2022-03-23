import pandas as pd 
import numpy as np

def test_order():
    old_df = pd.read_csv('~/wind_power_analysis/data/second_rendition_data/second_rendition_geolocated_noemoji_anonymous.csv')
    new_df = pd.read_csv('~/wind_power_analysis/data/second_rendition_data/second_rendition_geolocated_noemoji_anonymous_predict.csv')

    for i, line in old_df.iterrows():
        print(line['text'] == new_df.iloc[i,0]) 


if __name__ == '__main__':
    test_order()
