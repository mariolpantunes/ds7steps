# coding: utf-8
import tweepy
from tweepy import OAuthHandler
import csv

consumer_key = '2lDgkNXdm03bxodf55vlY5IHo'
consumer_secret = 'w5SaNzPCLyaBL1ieyGpm4uwjan5Y2GDqQjbbSUoBTT5Fl3cLP4'
access_token = '276620312-cHKy6zL3MRQy9VgC320fGWJQLBM5ALsm0ULc8Pvf'
access_secret = '9MZhoILO9T9BRtq4ciXzTHOq0hoQARCMDTSB3Nv98amkN'
QUERY = 'data science'
CSV_DATASET = 'twitter.csv'
MAX_TWEETS = 1000


def main():
    with open(CSV_DATASET, 'w') as csvfile:
        fieldnames = ['user', 'tweet']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)

        api = tweepy.API(auth, wait_on_rate_limit=True)
        data = tweepy.Cursor(api.search, q=QUERY).items(MAX_TWEETS)

        writer.writeheader()
        for d in data:
            j = d._json
            text = j['text']
            user = j['user']['screen_name']
            print(user, ": ", text)
            writer.writerow({'user': user, 'tweet': text})


if __name__ == "__main__":
    main()
