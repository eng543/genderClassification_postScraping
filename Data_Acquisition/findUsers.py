""""
#!/usr/bin/python

#https://dev.twitter.com/rest/reference/get/users/search

from twitter import *
from tweepy import *
import time

twitter = Twitter(auth = OAuth('512739542-yp06o7mXg6Yrexxz7zdGqpETL5k44PxOYhVKN2bw',
                               'cw9Z5PeK5YG7zuDtXpUZD40Apf2VFrUMo1XzWUHN6dn7F',
                               'D8XxzNAHM6uUPRlnQJUIYDhex',
                               'bubnRJsyEVFchIzY4078GgrUCEZCtPgR83VKctSwYajurpHOrb'))

auth = OAuthHandler('D8XxzNAHM6uUPRlnQJUIYDhex', 'bubnRJsyEVFchIzY4078GgrUCEZCtPgR83VKctSwYajurpHOrb')
auth.set_access_token('512739542-yp06o7mXg6Yrexxz7zdGqpETL5k44PxOYhVKN2bw', 'cw9Z5PeK5YG7zuDtXpUZD40Apf2VFrUMo1XzWUHN6dn7F')

api = API(auth)

users = {}

# start with most informative queries
queries = ['mom', 'dad', 'mother', 'father', 'sister', 'brother', 'aunt', 'uncle', 'gramma', 'grampa', 'grandma', 'grandpa',
           'grandmother', 'grandfather', 'mama', 'momma', 'papa', 'male', 'female', 'girl', 'boy', 'man', 'woman',
           'trans', 'queer', 'gender', 'binary', 'fluid', 'andro',
           'eunuch', 'MTF', 'FTM', 'MTX', 'FTX']

query_sub = ['mother', 'father', 'male', 'female', 'girl', 'boy', 'man', 'woman', 'trans', 'gender']

wf = open("results2.txt", "w")

# 50 pages per call (1000 total users)
page = 1
while True:
    try:
        results = twitter.users.search(q='"mother"', page=page, count=20)
        if results:
            for user in results:
                if user["lang"] == "en":
                    users[user["id_str"].encode('utf-8')] = [user["location"].encode('utf-8'), user["lang"].encode('utf-8'), user["description"].encode('utf-8')]
                    user2 = {user["id_str"].encode('utf-8'): [user["location"].encode('utf-8'), user["lang"].encode('utf-8'), user["description"].encode('utf-8')]}
                    wf.write(str(user2))
                    wf.write("\n")
        else:
            break

        page += 1


    except TweepError: ## some error
        time.sleep(60 * 15)
        continue

    except StopIteration:
        break

#wf = open("results.txt", "w")
#for i in range(0, len(users)):
#    wf.write(i)
#    wf.write("\n")

wf.close()
"""

#!/usr/bin/python

#https://dev.twitter.com/rest/reference/get/users/search

from twitter import *
from tweepy import *
import time

# Due to authorization, presumably only one script can be running at a time due to call limits. Not totally sure on this.

twitter = Twitter(auth = OAuth('512739542-yp06o7mXg6Yrexxz7zdGqpETL5k44PxOYhVKN2bw',
                               'cw9Z5PeK5YG7zuDtXpUZD40Apf2VFrUMo1XzWUHN6dn7F',
                               'D8XxzNAHM6uUPRlnQJUIYDhex',
                               'bubnRJsyEVFchIzY4078GgrUCEZCtPgR83VKctSwYajurpHOrb'))

auth = OAuthHandler('D8XxzNAHM6uUPRlnQJUIYDhex', 'bubnRJsyEVFchIzY4078GgrUCEZCtPgR83VKctSwYajurpHOrb')
auth.set_access_token('512739542-yp06o7mXg6Yrexxz7zdGqpETL5k44PxOYhVKN2bw', 'cw9Z5PeK5YG7zuDtXpUZD40Apf2VFrUMo1XzWUHN6dn7F')

#api = API(auth, retry_count = 0, retry_delay = 5, retry_errors = set([401, 404, 500, 503]), monitor_rate_limit = True, wait_on_rate_limit = True)
api = API(auth)
users = {}

# start with most informative queries. Added additional query due to an imbalance in retrievals. The additional query distributes the calls evenly over time.
queries = ['mom', 'dad', 'mother', 'father', 'sister', 'brother', 'aunt', 'uncle', 'gramma', 'grampa', 'grandma', 'grandpa',
           'grandmother', 'grandfather', 'mama', 'momma', 'papa', 'male', 'female', 'girl', 'boy', 'man', 'woman',
           'trans', 'queer', 'gender', 'binary', 'fluid', 'andro',
           'eunuch', 'MTF', 'FTM', 'MTX', 'FTX', 'fem']

query_sub = ['mother', 'father', 'male', 'female', 'girl', 'boy', 'man', 'woman', 'trans', 'gender']

wf = open("results.txt", "a")

# 50 pages per call (1000 total users)
num_searches = 5

for search in range(num_searches):
    entry = 0
    for entry in range(len(queries)):
        keyword = queries[entry]
        retrievals = 0
        try:
            for user in Cursor(api.search_users, q=keyword).items():
            #if user.lang == "en":
                users[user.id_str] = [keyword, user.location.encode('utf-8'), user.lang.encode('utf-8'), user.description.encode('utf-8')]
                user2 = {user.id_str: [keyword, user.location.encode('utf-8'), user.lang.encode('utf-8'), user.description.encode('utf-8')]}
                wf.write(str(user2))
                wf.write("\n")
                retrievals += 1
                if retrievals >= 1000:
                    print('1000 retrievals for ' + keyword)
                    break
                else:
                    pass
            entry += 1
        #else:
        #    break

        except TweepError: ## some error
            print('Naptime: last query = ' + keyword)
            time.sleep(60 * 15)
            continue

        except StopIteration:
            break
            

#wf = open("results.txt", "w")
#for i in range(0, len(users)):
#    wf.write(i)
#    wf.write("\n")

wf.close()