# WIND POWER ANALYSIS #

- [ ] Gather data with search.py by looping over dates in the script
- [ ] convert data to csv using lese_txt.py
- [ ] read the location attribute of users using geo_locating.py. produces tweets all from norway and with a city and longitude and latitude.


## Structure of the repo
```bash
├── NorBert_utls
│   ├── apply_model.py
│   ├── collumn_reverse.py
│   ├── csv_appender.py
│   ├── natalya_finetuning.py
│   ├── renamer.py
│   ├── retweet_remover.py
│   ├── roberta_solar_sentiment.py
│   └── various_methods
│       ├── finetuning_example.py
│       ├── roberta_threelabel.py
│       ├── sentiment_analysis_sklearn.py
│       └── stackabuse_nlp.py
├── data
│   ├── annotation_012label.csv
│   ├── anon_identifier.py
│   ├── data_utils.py
│   ├── emoji_remover.py
│   ├── fetch_longlat.py
│   ├── retweet_fulfiller.py
│   ├── search.py
│   ├── trainingdata_maker.py
│   ├── wordcloud_test.py
│   ├── anotation_data
│   │   └── manually annotated three label tweets
│   ├── first_rendition_data
│   │   └── csv data from first data fetch
│   ├── second_rendition_data
│   │   └── csv data from second data fetch
│   └─── third_rendition_data
│       └── csv data from third data fetch
├── fig
│   └─── various figures
├── results_analysis
│   ├── geography_occurence.py
│   ├── oskar_plot_utils.py
│   ├── plot_histogram_tweets_time.py
│   ├── plot_histogram_tweets_time_foursquare.py
│   ├── sentiment_histogram.py
│   ├── sentiment_over_time.py
│   └── user_analysis.py
├── backup
│   └── various backup data
├── apply_model.py
├── follower_followee_lookup.R
├── geo_locating.py
├── kommuner_komprimert.json
├── lese_txt.py
├── plot_utils.py
├── test_retweet_values.py
└── third_rendition_geolocated_anonymous_usernamedict.json
```

## search.py ##
script for searching twitter API with a bearer token and a search query. Max number of tweets at a time is 500, and it will not loop. Thus one has to do 500 tweets at a time, and manually update the search period.

outputs a .txt file with twitter data.

loop_request and connect_to_endpoint functions are non functional as twitter doesn't allow loops. 

## lese_txt.py ##
reads the lines of the search.py txt file, and organizes this into a csv.
First we look for a geo tag in the tweet element. If this is lacking then we fetch the geo tag of the user.
NOTE user geo element is self attributed, meaning you can write anything. This is handled in geo_locating.

outpus a .csv file with the specified elements from the tweet.

## plot_histogram_tweet_time.py ##
Reads the .csv output of lese_txt to plot a timeline of weekly tweets.
output is a jpg saved in the script location:w

## anon_identifier.py ##
reads the lese_txt csv and attributes each unique user a random identifier. The username and ID is written to a json file in scritp location. new anonymized dataset and the dictionary of keys is written to file.


outputs a .json of keys and usernames, a .csv of anonymized data and a histogram of unique users and amount of times they have tweeted. 


## geo_locating.py ##

Attempts to read location elements in the user metadata. This info is written by the user themselves, so a lot of them are not valid locations, and simply 'the couch' or something similar.

uses GeoText to identify actual place names, then uses nominatim to fetch the coordinates to these. Nominatim has a 1 request per second, meaning this takes a lot of time.

In reading location data it would simply take too long to loop through tens of thousands of tweets. This script therefore manually assigns locations to Oslo, Bergen and Trondheim. In my case this reduces number of searches from 44 to 14 thousand.

Produces csv with only norwegian and geolocated tweets with longitude and lattidude.

Note; if multiple place names are found in the users geo tag then we assume the first one is the main one.
## trainingdata_maker.py ## 

Script for anotating tweets. Takes a full_geodata_longlat_noforeign csv file and prints out the text. 
the user is promted with an option to return 0 or 1, and this is added to the given tweet as a label.
NOTE does not handle other integers than 0 or 1. 

## rename_collumn.py ## 

The NorBert algorithm accepts trainingdata on the form [labels, text]. I work with label instead of the plural labels. this scripts looks for any collumn named label and renames it labels. not very complicated. 

## emoji_remover.py ##
Script that runs through the dataset, and matches characters in a regex search, and then loops over a list of emojis emoji_table.txt
https://github.com/seandolinar/socialmediaparse/blob/master/data/emoji_table.txt


