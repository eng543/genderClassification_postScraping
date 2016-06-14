from twython import Twython

t = Twython(app_key='D8XxzNAHM6uUPRlnQJUIYDhex',       #REPLACE 'APP_KEY' WITH YOUR APP KEY, ETC., IN THE NEXT 4 LINES
    app_secret='bubnRJsyEVFchIzY4078GgrUCEZCtPgR83VKctSwYajurpHOrb',
    oauth_token='512739542-yp06o7mXg6Yrexxz7zdGqpETL5k44PxOYhVKN2bw',
    oauth_token_secret='cw9Z5PeK5YG7zuDtXpUZD40Apf2VFrUMo1XzWUHN6dn7F')

users = {}
for i in range(1, 2):
    results = t.search_users(q = '"dog" OR "man"')
    for user in results:
        users[user["id_str"].encode('utf-8')] = [user["location"].encode('utf-8'), user["lang"].encode('utf-8'), user["description"].encode('utf-8')]


print users