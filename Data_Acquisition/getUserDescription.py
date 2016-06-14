import sys
import string
import simplejson
from twython import Twython

import datetime
now = datetime.datetime.now()
day = int(now.day)
month = int(now.month)
year= int(now.year)

t = Twython(app_key='D8XxzNAHM6uUPRlnQJUIYDhex', #REPLACE 'APP_KEY' WITH YOUR APP KEY, ETC., IN THE NEXT 4 LINES
    app_secret='bubnRJsyEVFchIzY4078GgrUCEZCtPgR83VKctSwYajurpHOrb',
    oauth_token='512739542-yp06o7mXg6Yrexxz7zdGqpETL5k44PxOYhVKN2bw',
    oauth_token_secret='cw9Z5PeK5YG7zuDtXpUZD40Apf2VFrUMo1XzWUHN6dn7F')

ids = "512739542,"
    
users = t.lookup_user(user_id = ids)

outfn = "twitter_user_data_%i.%i.%i.txt" % (now.month, now.day, now.year)

fields = "id screen_name name created_at url followers_count friends_count statuses_count \
    favourites_count listed_count \
    contributors_enabled description protected location lang expanded_url".split()
 
#INITIALIZE OUTPUT FILE AND WRITE HEADER ROW   
outfp = open(outfn, "w")
outfp.write(string.join(fields, "\t") + "\n")  # header

#THE VARIABLE 'USERS' CONTAINS INFORMATION OF THE 32 TWITTER USER IDS LISTED ABOVE
#THIS BLOCK WILL LOOP OVER EACH OF THESE IDS, CREATE VARIABLES, AND OUTPUT TO FILE
for entry in users:
    #CREATE EMPTY DICTIONARY
    r = {}
    for f in fields:
        r[f] = ""
    #ASSIGN VALUE OF 'ID' FIELD IN JSON TO 'ID' FIELD IN OUR DICTIONARY
    r['id'] = entry['id']
    #SAME WITH 'SCREEN_NAME' HERE, AND FOR REST OF THE VARIABLES
    r['screen_name'] = entry['screen_name']
    r['name'] = entry['name']
    r['created_at'] = entry['created_at']
    r['url'] = entry['url']
    r['followers_count'] = entry['followers_count']
    r['friends_count'] = entry['friends_count']
    r['statuses_count'] = entry['statuses_count']
    r['favourites_count'] = entry['favourites_count']
    r['listed_count'] = entry['listed_count']
    r['contributors_enabled'] = entry['contributors_enabled']
    r['description'] = entry['description']
    r['protected'] = entry['protected']
    r['location'] = entry['location']
    r['lang'] = entry['lang']
    #NOT EVERY ID WILL HAVE A 'URL' KEY, SO CHECK FOR ITS EXISTENCE WITH IF CLAUSE
    if 'url' in entry['entities']:
        r['expanded_url'] = entry['entities']['url']['urls'][0]['expanded_url']
    else:
        r['expanded_url'] = ''
    print r
    #CREATE EMPTY LIST
    lst = []
    #ADD DATA FOR EACH VARIABLE
    for f in fields:
        lst.append(unicode(r[f]).replace("\/", "/"))
    #WRITE ROW WITH DATA IN LIST
    outfp.write(string.join(lst, "\t").encode("utf-8") + "\n")

outfp.close()    