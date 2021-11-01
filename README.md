# WIND POWER ANALYSIS #

- [ ] Gather data with search.py by looping over dates in the script
- [ ] convert data to csv using lese_txt.py
- [ ] read the location attribute of users using geo_locating.py. produces tweets all from norway and with a city and longitude and latitude.



## search.py ##
script for searching twitter API with a bearer token and a search query. Max number of tweets at a time is 500, and it will not loop. Thus one has to do 500 tweets at a time, and manually update the search period.

outputs a .txt file with twitter data.

loop_request and connect_to_endpoint functions are non functional as twitter doesn't allow loops. 

## lese_txt.py ##
reads the lines of the search.py txt file, and organizes this into a csv.
If Has:geo is not a search query it will look for wether the tweet element or user element has a geo attribute. Thus all elements returned from lese_txt will have some sort of geo tag.
NOTE user geo element is self attributed, meaning you can write anything.

outpus a .csv file with the specified elements from the tweet.

## reading_csv.py ##
Reads the output of lese_txt to plot a timeline of weekly tweets.
output is a jpg saved in the script location:w

## anon_identifier.py ##
reads the lese_txt csv and attributes each unique user a random identifier. The username and ID is written to a json file in scritp location. new anonymized dataset and the dictionary of keys is written to file.


outputs a .json of keys and usernames, a .csv of anonymized data and a histogram of unique users and amount of times they have tweeted. 


## geo_locating.py ##

Attempts to read location elements in the user metadata. This info is written by the user themselves, so a lot of them are not valid locations, and simply 'the couch' or something similar.

uses GeoText to identify actual place names, then uses nominatim to fetch the coordinates to these. Nominatim has a 1 request per second, meaning this takes a lot of time.

In reading location data it would simply take too long to loop through tens of thousands of tweets. This script therefore manually assigns locations to Oslo, Bergen and Trondheim. In my case this reduces number of searches from 44 to 14 thousand.

Produces csv with only norwegian and geolocated tweets with longitude and lattidude.

## trainingdata_maker.py ## 

Script for anotating tweets. Takes a full_geodata_longlat_noforeign csv file and prints out the text. 
the user is promted with an option to return 0 or 1, and this is added to the given tweet as a label.
NOTE does not handle other integers than 0 or 1. 

## rename_collumn.py ## 

The NorBert algorithm accepts trainingdata on the form [labels, text]. I work with label instead of the plural labels. this scripts looks for any collumn named label and renames it labels. not very complicated. 
