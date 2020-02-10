# Hunter Berry
# HON 450
# Twitter API Testing 

import tweepy;
import pandas as pd;
import re;
import time;
from textblob import TextBlob;
import sys;
from datetime import datetime;
import subprocess;


def clean_tweet(tweet): 
   return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())  

def diff(first, second):
   second = set(second)
   return [item for item in first if item not in second];

def getCategories(tweet):
   notedCategories = []; #partisanship, abortion, immigration, economics, millitary, etc;
   economicTerms = ['stock', 'economy'];
   immigrationTerms = ['wall', 'immigration', 'immigrants', 'sanctuary cities',];

   for term in economicTerms:
      if term in tweet:
         notedCategories.append("economics");
         break;

   for term in immigrationTerms:
      if term in tweet:
         notedCategories.append('immigration');
         break;

   return notedCategories;
   

def limit_handled(cursor, finished, total, diction):
   while True:
      try:
         yield cursor.next();
      except tweepy.RateLimitError:
         print("Rate limit reached. Preparing new process...");
         toBeCompleted = diff(total, finished);
         with open("finishText.txt", "w") as txt_file:
            for line in toBeCompleted:
               txt_file.write(line + "\n");
         tweets = pd.DataFrame(diction, columns=['user', 'tweetText', 'simplePolarity', 'nouns']);
         print("Exiting file...");
         subprocess.Popen(["py", "twitterAPI.py", "finishText.txt"], shell=True);
         sys.exit();
      except tweepy.TweepError:
         print("Error processing the user... Skipping now");
         break;
      except StopIteration:
         break;

def createFrame(api, tweetsToPull, users):
   diction = [];
   finishedUsers = [];

   for user in users:
      params = user.split(", ");
      if len(params) < 2:
         continue;
      userName = params[0];
      party = params[1];
      state = params[2];
      cursor = tweepy.Cursor(api.user_timeline, screen_name=userName, include_rts=True, tweet_mode='extended').items(tweetsToPull);
      for post in limit_handled(cursor, finishedUsers, users, diction):
         cleanedTweet = clean_tweet(post.full_text);
         if cleanedTweet[0:2] == 'RT':
            cleanedTweet = cleanedTweet[2:];
         #cats = getCategories(cleanedTweet);
         blob = TextBlob(cleanedTweet);
         polarity = blob.sentiment.polarity;
         nouns = blob.noun_phrases;
         diction.append([userName, cleanedTweet, polarity, nouns, party, state]);
         #might introduce party has 0 or 1 binary
         #something in here about a .split(", ") -> userName, party (R/D), and state
      finishedUsers.append(user);

   print("Iteration completed...")

   tweets = pd.DataFrame(diction, columns=['user', 'tweetText', 'simplePolarity', 'nouns', 'party', 'state']);
   return tweets;

def main():
   time.sleep(5);

   textFile = sys.argv[1];
   auth = tweepy.OAuthHandler("ZqarwsmGvqGU8IR7pmRUeG23j", "RfYaQf6l4hHIkynjvV5Yi17TzYjy2xBv9A0gwEjbFKfgxSrO3O");
   auth.set_access_token("4819588312-rXEoklKXE27hSQLnhERd8UBpJLp7FmVVk2CJles", "2imqeKDTNq6T1GeGCJUtL1yvpr6EJlOAYykAJxjhAIPvZ");

   api = tweepy.API(auth); # Connects to Twitter application using codes. 
   print("Authorizaion successful.")
   tweetsToPull = 5; # Number of tweets to pull for each individual user.
   parties = [];
   states = [];

   with open(textFile, 'r') as f:
      print("Attempting to populate user screen...")
      users = f.readlines()

   for user in range(len(users)):
      users[user] = users[user].strip();

   print("Data populated.")

   if 'realDonaldTrump' not in users:
      users.append('realDonaldTrump'); # List of all users to pull training data for.

   data = createFrame(api, tweetsToPull, users);
   print(data);
   
   now = datetime.now();
   dt_string = now.strftime("%d_%m_%Y_%H-%M");

   filename = 'data' + dt_string + '.csv'

   with open(filename, 'a') as f:
      data.to_csv(f, header=False)

   parties = {};
   states = {};

   for index, row in data.iterrows():
      if row['party'] in parties.keys():
         pass;
      else:
         parties[row['party']] = [];
      if row['state'] in states.keys():
         pass;
      else:
         states[row['state']] = [];
      for item in row['nouns']:
         if item not in parties[row['party']]:
            parties[row['party']].append(item);
         if item not in states[row['state']]
            #ensure there are no duplicates
            states[row['state']].append(item);
            #shoould i add a counter system
      #overall list for nouns, different than parties/state
      #reget the stuff oi lost from word

   print(parties);
   print(states);
         

if __name__ == '__main__':
   main();

#ok do machine learning stuff...
#use machine learning and some training data to determine whether  or not tweets fall into a certain cateofry... then run aspect-based on those assigned categoeis

'''
def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

diff(completed, finished);
'''

"""
stuff = api.user_timeline(screen_name = 'realDonaldTrump', count = 5, include_rts = True)
print(user.screen_name);
print(user.followers_count);
for friend in user.friends():
   print(friend.screen_name);

for item in stuff:
   print(item.text);
   
   #public_tweets = api.home_timeline()
#for tweet in public_tweets:
#    print(tweet.text)
#user = api.get_user('realDonaldTrump');

# above omitted for brevity
c = tweepy.Cursor(api.search,
                       q=search,
                       include_entities=True).items()
while True:
    try:
        tweet = c.next()
        # Insert into db
    except tweepy.TweepError:
        time.sleep(60 * 15)
        continue
    except StopIteration:
        break

"""
