# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# This version is based off Greg's original script. Wayne added annotations for Python beginners
"""
"""
# Importing Python packages. Notice that the following packages are required for this operation: sqlite3, simplejson,sqlalchemy
import sys
import string
import simplejson
import sqlite3
import time
import datetime
from pprint import pprint
import sqlalchemy
from sqlalchemy.orm import mapper, sessionmaker
from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, Unicode, Float # importing Unicode is important! If not, you likely encounter data type error.
from sqlalchemy.ext.declarative import declarative_base
from types import *
from datetime import datetime, date, time
##### FIRST BLOCK OF MODIFIED CODE --> ADDED TO IMPORT TWYTHON AND ADD OAUTH AUTHENTICATION #####
import twython
import time
from tweepy import *

#t = twython.Twython(app_key='D8XxzNAHM6uUPRlnQJUIYDhex',       #REPLACE 'APP_KEY' WITH YOUR APP KEY, ETC., IN THE NEXT 4 LINES
#    app_secret='bubnRJsyEVFchIzY4078GgrUCEZCtPgR83VKctSwYajurpHOrb',
#    oauth_token='512739542-yp06o7mXg6Yrexxz7zdGqpETL5k44PxOYhVKN2bw',
#     oauth_token_secret='cw9Z5PeK5YG7zuDtXpUZD40Apf2VFrUMo1XzWUHN6dn7F')

#auth = OAuthHandler('D8XxzNAHM6uUPRlnQJUIYDhex', 'bubnRJsyEVFchIzY4078GgrUCEZCtPgR83VKctSwYajurpHOrb')
#auth.set_access_token('512739542-yp06o7mXg6Yrexxz7zdGqpETL5k44PxOYhVKN2bw', 'cw9Z5PeK5YG7zuDtXpUZD40Apf2VFrUMo1XzWUHN6dn7F')

auth2 = OAuthHandler('s5GmeO4csHRtOoC1Ot3KIy7YJ', 'cCDDSJ1MV50FQwQaIZJZjFDDY9zr8DMebCjUVQcUVjyVFul4Cg')
auth2.set_access_token('731905408063946752-tvixRN0FKndxJjGs8vNflurAZQNKTDc', 'XwZMI8FFxWjUjZrd08xxgfWluLXTQjZZ0IgzOtRVrdWnL')

#api = API(auth)
api = API(auth2)

Base = declarative_base()
class TWEET(Base):
    __tablename__ = 'tweets1' # This is table name.
    rowid = Column(Integer, primary_key=True)
    query = Column(String)
    user_type = Column(Text)
    tweet_id = Column(String)
    language = Column(String)
    #year = Column(String)
    content = Column(Text)
    from_user_screen_name = Column(Text)
    entities_urls = Column(Unicode(255))
    entities_hashtags = Column(Unicode(255))
    entities_hashtags_count = Column(Integer)
    entities_mentions = Column(Unicode(255))
    entities_mentions_count = Column(Integer)
    in_reply_to_screen_name = Column(String)
    in_reply_to_status_id = Column(String)
    num_words = Column(Integer) # might be useful if we can get it to work
    num_followers = Column(String)
    num_friends = Column(String)

    def __init__(self, query, user_type, tweet_id, language,
    content, from_user_screen_name, entities_urls, entities_hashtags, entities_hashtags_count,
    entities_mentions, entities_mentions_count, in_reply_to_screen_name, in_reply_to_status_id,
    num_words, num_followers, num_friends):
        self.query = query
        self.user_type = user_type
        self.tweet_id = tweet_id
        self.language = language
        #self.year = year
        self.content = content
        self.from_user_screen_name = from_user_screen_name
        self.entities_urls = entities_urls
        self.entities_hashtags = entities_hashtags
        self.entities_hashtags_count = entities_hashtags_count
        self.entities_mentions = entities_mentions
        self.entities_mentions_count = entities_mentions_count
        self.in_reply_to_screen_name = in_reply_to_screen_name
        self.in_reply_to_status_id = in_reply_to_status_id
        self.num_words = num_words
        self.num_followers = num_followers
        self.num_friends = num_friends

    def __repr__(self):
       return "<sender, created_at('%s', '%s')>" % (self.from_user_screen_name,self.created_at)

class ACCOUNT(Base):
    __tablename__ = 'accounts' # this is the table name for a list of scree names to be mined. You need to go to SQLite Database browser and create a new DB (make sure that DB's name matches the one defined in this script); within that DB, create a table, make sure the table name and field names match the ones defined here.
    rowid = Column(Integer, primary_key=True)
    screen_name = Column(String)
    user_type = Column(String)

    def __init__(self, screen_name, user_type):
        self.screen_name = screen_name
        self.user_type = user_type

    def __repr__(self):
       return "<Company, CSR_account('%s', '%s')>" % (self.rowid, self.screen_name)


def get_data(kid, page_num):
    try:
        for page in Cursor(api.user_timeline, id=kid, count="200", page=page_num, include_rts="0").pages():
            d = page
         #d = t.get_user_timeline(screen_name=kid, count="200", page=page, include_entities="true", include_rts="0")  #NEW LINE
            print (str(len(d)) + " tweets on page " + str(page_num))
            return d

    except TweepError as e:
        if e.message == "Twitter error response: status code = 429":
            print ('Naptime -- last query: ' + kid)
            time.sleep(15*60)

        elif e.message == "Twitter error response: status code = 401":
            pass

        else:
            print e.message


#def limit_handled(cursor):
#    while True:
#        try:
#            yield cursor.next()
#        except RateLimitError:
#            time.sleep(15*60)


def write_data(self, d, screen_name, user_type):

    query = screen_name

    user_type = user_type

    for entry in d:
        print user_type
        tweet_id = entry.id
        #tweet_id = entry['id']
        #language = entry['lang']
        language = entry.lang
        #content = entry['text']
        content = entry.text
        #created_at_text = entry['created_at']
        #created_at_text = entry.created_at
        #created_at = datetime.strptime(created_at_text, '%a %b %d %H:%M:%S +0000 %Y')
        #year = created_at.strftime('%Y')
        #from_user_screen_name = entry['user']['screen_name']
        from_user_screen_name = entry.user.screen_name
        #in_reply_to_screen_name = entry['in_reply_to_screen_name']
        in_reply_to_screen_name = entry.in_reply_to_screen_name
        #in_reply_to_status_id = entry['in_reply_to_status_id']
        in_reply_to_status_id = entry.in_reply_to_status_id
        #entities_hashtags_count = len(entry['entities']['hashtags'])
        entities_hashtags_count = len(entry.entities['hashtags'])
        #entities_mentions_count = len(entry['entities']['user_mentions'])
        entities_mentions_count = len(entry.entities['user_mentions'])
        #num_friends = entry['friends_count']
        num_friends = entry.user.friends_count
        #num_followers = entry['followers_count']1
        num_followers = entry.user.followers_count


        entities_urls = []
        entities_hashtags, entities_mentions = [], []

        for link in entry.entities['urls']:
            if 'url' in link:
                url = link['url']
                entities_urls.append(url)
            else:
                print "No urls in entry"

        for hashtag in entry.entities['hashtags']:
            if 'text' in hashtag:
                tag = hashtag['text']
                entities_hashtags.append(tag)
            else:
                print "No hashtags in entry"

        for at in entry.entities['user_mentions']:
            if 'screen_name' in at:
                mention = at['screen_name']
                entities_mentions.append(mention)
            else:
                print "No mentions in entry"

        entities_mentions = string.join(entities_mentions, u", ")
        entities_hashtags = string.join(entities_hashtags, u", ")
        entities_urls = string.join(entities_urls, u", ")
        entities_urls = unicode(entities_urls)
        entities_hashtags = unicode(entities_hashtags)
        entities_mentions = unicode(entities_mentions)

        updates = self.session.query(TWEET).filter_by(query=query, from_user_screen_name=from_user_screen_name,
                content=content).all()
        if not updates:
            print "inserting, query:", query
            upd = TWEET(query, user_type, tweet_id, language, content,
                        from_user_screen_name, entities_urls, entities_hashtags, entities_hashtags_count,
                        entities_mentions, entities_mentions_count,in_reply_to_screen_name, in_reply_to_status_id, None,
                        num_followers, num_friends)
            self.session.add(upd)
        else:
            if len(updates) > 1:
                print "Warning: more than one update matching to_user=%s, text=%s"\
                    % (to_user, content)
            else:
                print "Not inserting, dupe.."
class Scrape:
    def __init__(self):
        engine = sqlalchemy.create_engine("sqlite:///tweets_from_corpus4.db", echo=False)  # different DB name here
        Session = sessionmaker(bind=engine)
        self.session = Session()
        Base.metadata.create_all(engine)

    def main(self):

        all_ids = self.session.query(ACCOUNT).all()

        keys = []
        for i in all_ids[1322:]:
            for page_num in range(1, 17): # 3200 possible tweets from each person, 200 tweets per call
                screen_name = i.screen_name
                kid = screen_name
                rowid = i.rowid
                user_type = i.user_type
                print "\rprocessing id %s/%s  --  %s" % (rowid, len(all_ids), screen_name),
                sys.stdout.flush()

                d = get_data(kid, page_num)

                if not d:
                    continue

                if len(d)==0:
                    print "THERE WERE NO STATUSES RETURNED........MOVING TO NEXT ID"
                    continue

                write_data(self, d, screen_name, user_type)

                self.session.commit()

        self.session.close()


if __name__ == "__main__":
    s = Scrape()
    s.main()