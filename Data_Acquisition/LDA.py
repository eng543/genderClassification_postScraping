from gensim import corpora, models
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import bigrams, trigrams
import numpy as np
from collections import Counter, defaultdict
import string
import re
#from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

t1 = pd.read_csv("scraped_tweets1.csv", encoding="utf-8")
t2 = pd.read_csv("scraped_tweets2.csv", encoding="utf-8")
t3 = pd.read_csv("scraped_tweets3.csv", encoding="utf-8")
t4 = pd.read_csv("scraped_tweets4.csv", encoding="utf-8")
t5 = pd.read_csv("scraped_tweets5.csv", encoding="utf-8")

frames = [t1, t2, t3, t4, t5]
tweets = pd.concat(frames)

# remove non-english tweets, reset indices
tweets_eng = tweets[tweets['language'].isin(['en', 'und'])]
tweets_eng = tweets_eng.reset_index(drop = True)

# filter users who have less than 100 tweets
tweets_eng['count'] = 1
tweet_count_grouped = tweets_eng.groupby('query')
tweet_count = tweet_count_grouped['count'].agg([np.sum])
tweet_count['keep'] = tweet_count['sum'] >= 100
tweet_keep = tweet_count[tweet_count['keep'] == True]
users_tweet_keep = list(tweet_keep.index)
tweets_eng_keep = tweets_eng[tweets_eng['query'].isin(users_tweet_keep)]

#len(tweets_eng_keep), len(tweets_eng)

# filter users who have 3x as many friends as followers (potential bots)
friend_count_grouped = tweets_eng_keep.groupby('query')
followers_count = friend_count_grouped['num_followers'].agg([np.mean])
friends_count = friend_count_grouped['num_friends'].agg([np.mean])
friends_count['ratio'] = friends_count['mean']/friends_count['mean']
friends_count['keep'] = friends_count['ratio'] <= 3
ratio_keep = friends_count[friends_count['keep'] == True]
users_ratio_keep = list(ratio_keep.index)
tweets_trimmed = tweets_eng_keep[tweets_eng_keep['query'].isin(users_ratio_keep)]
#len(tweets_trimmed), len(tweets_eng_keep), len(tweets_eng)
tweets_trimmed = tweets_trimmed.reset_index(drop = True)

# deal with NaN
tweets_trimmed['entities_hashtags'] = tweets_trimmed['entities_hashtags'].fillna(0)
tweets_trimmed['entities_urls'] = tweets_trimmed['entities_urls'].fillna(0)

tweets_subset1 = tweets_trimmed[tweets_trimmed['query'] == "ForeverMoreVids"]
tweets_subset2 = tweets_trimmed[tweets_trimmed['query'] == "brianjdixon"]
subsets = [tweets_subset1, tweets_subset2]
tweets_subset = pd.concat(subsets)
tweets_subset = tweets_subset.reset_index(drop = True)

# expressions to remove punctuation, urls, hashtags, @mentions, numbers, emoticons
punctuation = set(string.punctuation)
blank = ["", " "]
url = re.compile(r'^http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')
hashtags = re.compile(r'^(?:\#+[\w_]+[\w\'_\-]*[\w_]+)')
mentions = re.compile(r'^(?:@[\w_]+)')
numbers = re.compile(r'(\d+)\D*(\d*)\D*(\d*)\D*(\d*)') # includes phone numbers
smiley = re.compile(r'[:=;\|\)\(\[\]\{\}][oO\-]?[D\)\]\(\[/\\OpP\|\{\}:]')
remove_regex = [url, hashtags, mentions, numbers, smiley]

tknzr = TweetTokenizer(strip_handles = True, reduce_len = True)
p_stemmer = PorterStemmer()
en_stop = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']

terms_all = []
users = []

for user in list(set(tweets_trimmed['query'])):
    user_sub = tweets_trimmed[tweets_trimmed['query'] == user]
    tweets = list(set(user_sub['content']))
    # unigrams only
    user_terms = []
    for tweet in tweets:
        terms = tknzr.tokenize(tweet.lower())
        for term in terms:
        	if (not any(rr.search(term) for rr in remove_regex)) and (term not in punctuation) and not (term.startswith('www')):
        		if term not in en_stop:
        			stemmed_term = p_stemmer.stem(term)
        			user_terms.append(stemmed_term.encode('ascii', 'ignore'))
    terms_all.append(user_terms)
    users.append(user)


dictionary = corpora.Dictionary(terms_all)

corpus = [dictionary.doc2bow(term) for term in terms_all]

ldamodel = models.ldamodel.LdaModel(corpus, num_topics=200, id2word = dictionary, passes=100)

ldamodel.save('savedModel')