import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('seaborn')
#%matplotlib inline 
import datetime as dt 


#defining colours
blue = '#0072B2'
bluegreen = '#009E73'
yellow = '#F0E442'
skyblue = '#56B4E9'
orange = '#E69F00'
vermilion = '#D55E00'


df_twitter = pd.read_csv(
    "../data/third_rendition_data/third_rendition_geolocated_anonymous_posneutral_predict.csv",
    parse_dates=[
        'created_at'
    ]
)

df_twitter_monthly = pd.DataFrame(df_twitter.copy())

df_twitter_monthly.index = df_twitter_monthly[
    'created_at'
].dt.to_period(
    'M'
)
df_twitter_monthly.groupby(
    level=0
).count(
)#.plot(
#    xlabel = 'Year',
#    ylabel = 'Number of tweets',
#    legend = None,
#    #title = 'Number of tweets over time'
#)

#now with sentiment 
df_twitter_monthly_senti = pd.DataFrame(df_twitter.copy())
df_twitter_monthly_senti.index = df_twitter_monthly_senti['created_at'].dt.to_period('M')



# We first calculate the total number of values for each months
total_sentiments = df_twitter_monthly_senti.groupby(level=0).count()

# then next, we calculate the sum of the label values, which will correspond to the total amount of non-negative tweets
average_sentiments = df_twitter_monthly_senti.groupby(level=0).sum()



# 1 = positive, 0 = negative
# the amount of negative sentiments is calculated by the total amount of sentiments minus the amount of non-negative
# the amount of non-negative is calculated before
df_twitter_monthly_senti['negative_sentiment'] = total_sentiments['label'] - average_sentiments['label']
df_twitter_monthly_senti['non-negative_sentiment'] = average_sentiments['label']

df_twitter_monthly_senti = df_twitter_monthly_senti.rename(columns={'created_at':'year_month'}).sort_values('year_month')
xs = [0,100]

fig, ax = plt.subplots(figsize=(16,10))
ax2 = ax.twinx()
df_twitter_monthly_senti.plot(
    ax=ax,
    y = ['negative_sentiment','non-negative_sentiment'],
    title = 'Amount of tweets over time (aggregated monthly), with binary classification',
    color = [vermilion,blue]
)

ax.axvline(
    dt.datetime(2019, 4, 1),
    color="#E69F00",
    label='NVE present proposed National Framework for Wind Power',
    linestyle = "--"
)

ax.axvline(
    dt.datetime(2019, 9, 1),
    color="#000000",
    label='Municipal and regional election',
    linestyle = "--"
)

ax.axvline(
    dt.datetime(2020, 6, 1),
    color="#CC79A7",
    label='Meld. St. 28 (2019-2020) - Tighter rules for wind power introduced',
    linestyle = "--"
)

ax.axvline(
    dt.datetime(2021, 9, 1),
    color="#56B4E9",
    label='Parliamentary election and Fosen Wind controversy',
    linestyle = "--"
)

plt.legend(fontsize=14)
plt.xlabel('Year')
plt.ylabel('Number of tweets')
#plt.show()
#plt.savefig('figures/tweets_over_time_monthly_agg.eps')

# Read the wind power data from NVE
df_wp = pd.read_csv(
    '../data/norway_wp_nve.csv',
    parse_dates = ['Produksjon oppstart']
).groupby(
    'Produksjon oppstart',
    as_index = False
).sum(
).drop(
    0
).rename(
    columns=
    {
        'Middelproduksjon [GWh]': 'average_generation_GWh',
        'Installert effekt [MW]': 'installed_capacity_MW'
    }
).assign(
    cumulative_annually_average_production_GWh = lambda x: x.average_generation_GWh.cumsum(),
    cumulative_installed_capacity_MW = lambda x: x.installed_capacity_MW.cumsum()
)
print(df_wp)
prod_start = df_wp['Produksjon oppstart'].iloc[0]
prod_slutt = df_wp['Produksjon oppstart'].iloc[-1]

daterange = pd.date_range(prod_start, prod_slutt)
plot_elems = pd.DataFrame()
plot_elems['daterange'] = daterange
plot_elems['cap'] = df_wp['installed_capacity_MW']
plot_elems['gen'] = df_wp['average_generation_GWh']
print(df_twitter_monthly_senti['year_month'])
print(plot_elems)

#ax2.plot(df_wp['Produksjon oppstart'], df_wp['installed_capacity_MW'], label='installert kapasitet MW')
#ax2.plot(df_wp['Produksjon oppstart'], df_wp['average_generation_GWh'], label='snitt produksjon GWh')
ax2.plot(plot_elems['daterange'], plot_elems['cap'])
ax2.grid(None)

plt.legend()

plt.show()

