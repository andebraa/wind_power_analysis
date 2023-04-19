#https://cran.r-project.org/web/packages/rtweet/vignettes/intro.html
install.packages("rtweet")
#when it asks for compilation say no

## quick overview of rtweet functions
vignette("auth", package = "rtweet")

## store api keys
api_key <- "" #ADD TOKENS TO THESE LINES
api_secret_key <- ""
access_token <- ""
access_token_secret <- ""

## authenticate via web browser
token <- create_token(
  app = "", #APP NAME
  consumer_key = api_key,
  consumer_secret = api_secret_key,
  access_token = access_token,
  access_secret = access_token_secret)

token

## check to see if the token is loaded next time (must restart R studio if you change tokens)
library(rtweet)
get_token()

#GET FRIENDS
#Retrieve a list of all the accounts a user follows.

#example from https://cran.r-project.org/web/packages/rtweet/vignettes/intro.html
## get user IDs of accounts followed by CNN
#cnn_fds <- get_friends("cnn")
## lookup data on those accounts
#cnn_fds_data <- lookup_users(cnn_fds$user_id)

#trying it out on my Twitter profile
#get user IDs of accounts followed by a user
me_fds = get_friends("jessixarobinson")
#lookup data on those accounts
me_fds_data = lookup_users(me_fds$user_id)

# extract the user data from the scrape
me_fds_user_data=users_data(me_fds_data)

# create datafrome with all rows and user id (1), screen name (2), name (3), location (4), and description (5), and followers (8)
me_friends=me_fds_user_data[, c(1:5, 8)]
me_friends

#it turns out the description fields sometimes contain new lines, which messes up the csv
#there's probably a way to fix this when writing to csv, but I don't want to figure it out now
#so I'm dropping description
# create datafrome with all rows and user id (1), screen name (2), name (3), location (4), and followers (8)
me_friends=me_fds_user_data[, c(1:4, 8)]
me_friends

#write them to a csv file
write.csv(me_friends,"/Users/jessicyr/my_friends.csv", row.names = FALSE)


##GET FOLLOWERS
#Same idea, just with ##get_followers## function

##Example from https://cran.r-project.org/web/packages/rtweet/vignettes/intro.html
## get user IDs of accounts following CNN
cnn_flw <- get_followers("cnn", n = 75000)
## lookup data on those accounts
cnn_flw_data <- lookup_users(cnn_flw$user_id)

