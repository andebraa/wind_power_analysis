import geopandas 
import pandas as pd 


data = pd.read_csv('full_geodata_longlat.csv',
                    usecols = ['username', 
                               'text', 
                               'loc', 
                               'created_at', 
                               'like_count', 
                               'quote_count', 
                               'city', 
                               'latitude', 
                               'longitude']
                    )


