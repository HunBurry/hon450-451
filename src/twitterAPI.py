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

parties = {}
states = {}
stateBreakdown = {}

def clean_tweet(tweet): 
   return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())  

def diff(first, second):
   second = set(second)
   return [item for item in first if item not in second];

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

def single_user_population(username):
    auth = tweepy.OAuthHandler("ZqarwsmGvqGU8IR7pmRUeG23j", "RfYaQf6l4hHIkynjvV5Yi17TzYjy2xBv9A0gwEjbFKfgxSrO3O");
    auth.set_access_token("4819588312-rXEoklKXE27hSQLnhERd8UBpJLp7FmVVk2CJles", "2imqeKDTNq6T1GeGCJUtL1yvpr6EJlOAYykAJxjhAIPvZ");

    api = tweepy.API(auth); # Connects to Twitter application using codes. 
    print("Authorizaion successful.")
    tweetsToPull = 5; # Number of tweets to pull for each individual user.
    tweets = [];
    cursor = tweepy.Cursor(api.user_timeline, screen_name=username, include_rts=True, tweet_mode='extended').items(tweetsToPull);
    for post in cursor:
        cleanedTweet = clean_tweet(post.full_text);
        if cleanedTweet[0:2] == 'RT':
            cleanedTweet = cleanedTweet[2:];
        tweets.append(cleanedTweet)
    
    tweets = pd.DataFrame(tweets, columns=['tweet']);
    return tweets;

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
      if state not in stateBreakdown.keys():
         stateBreakdown[state] = {party: 1};
      cursor = tweepy.Cursor(api.user_timeline, screen_name=userName, include_rts=True, tweet_mode='extended').items(tweetsToPull);
      for post in limit_handled(cursor, finishedUsers, users, diction):
         cleanedTweet = clean_tweet(post.full_text);
         if cleanedTweet[0:2] == 'RT':
            cleanedTweet = cleanedTweet[2:];
         blob = TextBlob(cleanedTweet);
         polarity = blob.sentiment.polarity;
         nouns = blob.noun_phrases;
         diction.append([userName, cleanedTweet, polarity, nouns, party, state]);
      finishedUsers.append(user);

   print("Iteration completed...");
   
   tweets = pd.DataFrame(diction, columns=['user', 'tweetText', 'simplePolarity', 'nouns', 'party', 'state']);
   return tweets;

def main():
   global parties, states;
   
   time.sleep(5);
   if len(sys.argv) <= 1:
      textFile = '../congress.txt'
   else:
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

   data = createFrame(api, tweetsToPull, users);
   print(data);
   
   now = datetime.now();
   dt_string = now.strftime("%d_%m_%Y_%H-%M");

   filename = 'data' + dt_string + '.csv'

   data.to_csv('./data/user_data/' + filename);
   return filename;

if __name__ == '__main__':
   main();