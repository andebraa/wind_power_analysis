'''
Script for visualizing sentiment over time. Loops through all tweets and
fetches those in a given time interval. Plots histogram of those tweets
over different time intervals.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
plt.style.use('seaborn')


df = pd.read_csv('../data/third_rendition_data/third_rendition_geolocated_anonymous_posneutral_predict.csv', parse_dates=True)

intervals_neg = [[],[],[],[]]
intervals_pos = [[],[],[],[]]
intervals_sum = [[],[],[],[]]

start0 = datetime.datetime.strptime("01-01-2016", "%d-%m-%Y")
start1 = datetime.datetime.strptime("13-03-2019", "%d-%m-%Y")
start2 = datetime.datetime.strptime("01-01-2021", "%d-%m-%Y")
start3 = datetime.datetime.strptime("01-11-2021", "%d-%m-%Y")

end0 = datetime.datetime.strptime("01-01-2017", "%d-%m-%Y")
end1 = datetime.datetime.strptime("13-05-2019", "%d-%m-%Y")
end2 = datetime.datetime.strptime("01-04-2021", "%d-%m-%Y")
end3 = datetime.datetime.strptime("01-02-2022", "%d-%m-%Y")
for i, tweet in df.iterrows():
    #2008-12-08T07:25:53.000Z
    if start0 <= datetime.datetime.strptime(tweet['created_at'],'%Y-%m-%dT%H:%M:%S.%fz') <= end0:
        intervals_neg[0].append(tweet['logits0'])
        intervals_pos[0].append(tweet['logits1'])
        intervals_sum[0].append(tweet['logits1']+tweet['logits0'])
        
    if start1 <= datetime.datetime.strptime(tweet['created_at'],'%Y-%m-%dT%H:%M:%S.%fz') <= end1:
        intervals_neg[1].append(tweet['logits0'])
        intervals_pos[1].append(tweet['logits1'])
        intervals_sum[1].append(tweet['logits1']+tweet['logits0'])
    
    if start2 <= datetime.datetime.strptime(tweet['created_at'],'%Y-%m-%dT%H:%M:%S.%fz') <= end2:
        intervals_neg[2].append(tweet['logits0'])
        intervals_pos[2].append(tweet['logits1'])
        intervals_sum[2].append(tweet['logits1']+tweet['logits0'])

    if start3 <= datetime.datetime.strptime(tweet['created_at'],'%Y-%m-%dT%H:%M:%S.%fz') <= end3:
        intervals_neg[3].append(tweet['logits0'])
        intervals_pos[3].append(tweet['logits1'])
        intervals_sum[3].append(tweet['logits1']+tweet['logits0'])


fig, ax = plt.subplots(2,2)
ax = ax.ravel()

for i,a in enumerate(ax):
    a.hist(intervals_sum[i], label = 'non-negative', bins=30)
    a.hist(intervals_neg[i], label = 'negative', bins=30, alpha = 0.7)
    a.hist(intervals_pos[i], label = 'non-negative', bins=30, alpha = 0.7)
    a.set_xlabel('logit sentiment score')
    a.legend()
    
    
ax[0].set_title(f'{start0.year} {start0.month} - {end0.year} {end0.month}')
ax[1].set_title(f'{start1.year} {start1.month} - {end1.year} {end1.month}')
ax[2].set_title(f'{start2.year} {start2.month} - {end2.year} {end2.month}')
ax[3].set_title(f'{start3.year} {start3.month} - {end3.year} {end3.month}')
plt.tight_layout()
plt.savefig('../fig/negativity_histogram.png', dpi = 200)
