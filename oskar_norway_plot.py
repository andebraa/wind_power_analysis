import re
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point

gdf_nuts3 = gpd.read_file(
    'data/NUTS_RG_01M_2021_3857_LEVL_3.shp.zip',
    crs = "EPSG:4326"
)



# Filter out all countries except Norway
gdf_NOnuts3 = gdf_nuts3[gdf_nuts3.CNTR_CODE.str.contains('NO')]

# We manually filter out Svalbard and Jan Mayen as we are not interested in them
gdf_NOnuts3 = gdf_NOnuts3.query('FID != "NO0B1" and FID != "NO0B2"')

# Return to sequential index from 0 to 13
gdf_NOnuts3 = gdf_NOnuts3.reset_index()

# Set the coordinate reference system to EPSG:4326 - same as our twitter data.
gdf_NOnuts3 = gdf_NOnuts3.to_crs("EPSG:4326")

## import and process twitter data 

# We load the Twitter data
df_twitter = pd.read_csv(
    'final_dataset.csv',parse_dates=['created_at']
)

# Transform it into a GeoDataFrame with point geometry based on longitude and latitude.
gdf_twitter = gpd.GeoDataFrame(
    df_twitter,
    crs = "EPSG:4326",
    geometry=gpd.points_from_xy(df_twitter.longitude, df_twitter.latitude)
)
print(gdf_twitter.head())


#getting boolean label values
delete_rows = []
bool_label = np.zeros(len(df_twitter))

for i, row in df_twitter.iterrows():
    try:
        search = re.search(r"\[\s*((?:-){0,1}\d.\d+(?:e-\d+)*)\s+((?:-){0,1}\d.\d+(?:e-\d+)*)\s*\]", row['labels']).groups()
        labels = (float(search[0]), float(search[1]) )
        if labels[0] < labels[1]:
            bool_label[i] = 1

    except:
        print("delete row {}".format(i))
        delete_rows.append(i) #rows with nan labels are saved for later deletion

df_twitter.drop(df_twitter.index[delete_rows]) # delete rows with nan indexes

#adding boolean label to dataframe
df_twitter['label'] = bool_label 

print(df_twitter.head())
# Create a new list with one row for each city
gdf_twitter_grouped = gdf_twitter[gdf_twitter.city != "Longyearbyen"]
gdf_twitter_grouped = gdf_twitter.groupby('city').first()
gdf_twitter_sentiment = gdf_twitter.groupby('label').first()

# Add a new column 'twitter_frequency' with the number of times each city occur.
# This amounts to the twitter frequency.
gdf_twitter_grouped['twitter_frequency'] = gdf_twitter['city'].value_counts()
gdf_twitter_grouped['label'] = gdf_twitter['label'].value_counts()

gdf_twitter.info()

# Only select the relevant columns
gdf_twitter_grouped_sorted = gdf_twitter_grouped[["geometry","twitter_frequency"]]
gdf_twitter_grouped_sorted = gdf_twitter_grouped_sorted.reset_index()

print(gdf_twitter_grouped_sorted.head())

## TESTING --------------------------------------------------




# Spatial joins https://geopandas.org/docs/user_guide/mergingdata.html#spatial-joins
# Basically, we add an attribute for each city related to the county the coordinate is located inside.
gdf_twitter_with_county = gdf_twitter_grouped_sorted.sjoin(gdf_NOnuts3, how = "inner", predicate = 'intersects')
# Summate the different cities within each county
gdf_frequency_county = gdf_twitter_with_county.groupby('NUTS_NAME').sum()
gdf_frequency_county.head(2)

# Spatial joins https://geopandas.org/docs/user_guide/mergingdata.html#spatial-joins
# Basically, we add an attribute for each city related to the county the coordinate is located inside.
gdf_twitter_with_county = gdf_twitter_grouped_sorted.sjoin(gdf_NOnuts3, how = "inner", predicate = 'intersects')
# Summate the different cities within each county
gdf_frequency_county = gdf_twitter_with_county.groupby('NUTS_NAME').sum()
gdf_frequency_county.head(2)













