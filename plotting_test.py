import pandas as pd 
import matplotlib.pyplot as plt 
import geopandas


norway = geopandas.read_file('kommuner_komprimert.json')

data = pd.read_csv('full_geodata_longlat_noforeign.csv',
                  usecols = ['user', 'username','text','loc','created_at',
                             'like_count','quote_count','city','latitude',
                             'longitude']
                  )

norway.plot()
plt.plot(data['latitude'], data['longitude'])
plt.show()
"""
colorscale=[
            [0, 'rgb(31,120,180)'], 
            [0.35, 'rgb(166, 206, 227)'], 
            [0.75, 'rgb(251,154,153)'], 
            [1, 'rgb(227,26,28)']
           ]

data = dict(type='choropleth',
            colorscale = colorscale,
            reversescale=True,
            locations = df_state_sentiment['states'],
            z = df_state_sentiment['total_sentiment'],
            locationmode = 'USA-states',
            text = df_state_sentiment['states'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
                          colorbar = {'title':"Twitter Sentiment"}
            )

layout = dict(title = 'Twitter Sentiment: GOP House Speaker: Paul Ryan',
                       geo = dict(scope='usa')
             )

choromap_us = go.Figure(data = [data],layout = layout)

# plotly.offline.plot(choromap_us, filename='img_map.html')  # save html map
IFrame('img_map.html', width=950, height=700)  # view saved map html file

"""
