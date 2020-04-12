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

def naive_bayes_final(stateDict, word, state):
   '''
   p(state) -> numInState/numInCountry
   p(word|state) -> numOfTermInState/numOfTermTotal
   p(word) -> numOfTerm/numInCountry
   '''

   wordsInState = 0;
   for key in stateDict[state].keys():
      wordsInState = wordsInState + stateDict[state][key];

   if word not in stateDict[state].keys():
      return 0;
   numInState = stateDict[state][word];
   numOfTermTotal = 0;
   numTotal = 0; 
   for key in stateDict.keys():
      if word in stateDict[key].keys():
         numOfTermTotal = numOfTermTotal + stateDict[key][word];
      for subKey in stateDict[key].keys():
         numTotal = numTotal + stateDict[key][subKey];

   pWordState = numInState / wordsInState;
   pWord = numOfTermTotal / numTotal;
   pState = wordsInState / numTotal;

   print("wordsInState: " + str(wordsInState));
   print(pState);
   print("___________________")

   print("numInState: " + str(numInState));
   print("numOfTermTotal: " + str(numOfTermTotal));
   print(pWordState);
   print("___________________");

   print("numTotal: " + str(numTotal));
   print(pWord);

   finalProb = (pState * pWordState) / pWord;

   return finalProb;

def naive_bayes(dictionary, term, state, totalInState, totalInCountry):
   #count of term in state/ total words in state * (# of people from state / total ppl)
   # all that divided by number of term / total words 
   if term in dictionary[state].keys():
      numOfTermInCountry = 0;
      numOfWordsInCountry = 0;
      numOfTermOccurences = dictionary[state][term];
      numOfTotalStateOccurences = 0;
      for key in dictionary[state].keys():
         numOfTotalStateOccurences = numOfTotalStateOccurences + dictionary[state][key];
      for key in dictionary.keys():
         for subKey in dictionary[key].keys():
            if subKey == term:
               numOfTermInCountry = numOfTermInCountry + dictionary[key][subKey];
            numOfWordsInCountry = numOfWordsInCountry + dictionary[key][subKey];
      return (((numOfTermOccurences / numOfTotalStateOccurences)  * (totalInState / totalInCountry)) / (numOfTermInCountry / numOfWordsInCountry));
   else:
      return 0;
   #if numOfTermInCountry = 0; return 1;

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
      if state not in stateBreakdown.keys():
         stateBreakdown[state] = {party: 1};
      else:
         stateBreakdown[state][party] = stateBreakdown[state][party] + 1;
      cursor = tweepy.Cursor(api.user_timeline, screen_name=userName, include_rts=True, tweet_mode='extended').items(tweetsToPull);
      for post in limit_handled(cursor, finishedUsers, users, diction):
         cleanedTweet = clean_tweet(post.full_text);
         if cleanedTweet[0:2] == 'RT':
            cleanedTweet = cleanedTweet[2:];
         #if location is not incuded in tweeet, use the NB to find their suspeected state
         blob = TextBlob(cleanedTweet);
         polarity = blob.sentiment.polarity;
         nouns = blob.noun_phrases;
         diction.append([userName, cleanedTweet, polarity, nouns, party, state]);
      finishedUsers.append(user);

   print("Iteration completed...");
   
   tweets = pd.DataFrame(diction, columns=['user', 'tweetText', 'simplePolarity', 'nouns', 'party', 'state']);

   ##new data frame -> one for state, one for party
   ##also include number used, highestSent, lowestSent, averageSent;
   ##https://scikit-learn.org/stable/modules/naive_bayes.html
   return tweets;

def main():
   global parties, states;
   
   time.sleep(5);
   if len(sys.argv) <= 1:
      textFile = 'congress.txt'
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

   #if 'realDonaldTrump' not in users:
   #   users.append('realDonaldTrump'); # List of all users to pull training data for.

   data = createFrame(api, tweetsToPull, users);
   print(data);
   
   '''
   numTotal = 0;
   for key in stateBreakdown.keys():
      if "D" not in stateBreakdown[key].keys():
         stateBreakdown[key]["D"] = 0;
      else:
         numTotal = numTotal + stateBreakdown[key]['D'];
      if "R" not in stateBreakdown[key].keys():
         stateBreakdown[key]['R'] = 0;
      else:
         numTotal = numTotal + stateBreakdown[key]['R'];

         '''
       
   
   now = datetime.now();
   dt_string = now.strftime("%d_%m_%Y_%H-%M");

   filename = 'data' + dt_string + '.csv'

   with open(filename, 'a') as f:
      data.to_csv(f, header=False)

   highestSent = {};
   lowestSent = {};

   for index, row in data.iterrows():
      if row['party'] in parties.keys():
         pass;
      else:
         parties[row['party']] = {};
      if row['state'] in states.keys():
         pass;
      else:
         states[row['state']] = {};
   
      for item in row['nouns']:
         if item not in parties[row['party']].keys():
            parties[row['party']][item] = 1;
         else:
            parties[row['party']][item] = parties[row['party']][item] + 1;
         if item not in states[row['state']]:
            states[row['state']][item] = 1;
         else:
            states[row['state']][item] = states[row['state']][item] + 1;

         #record highest and compare back 
      
   listOfStates = ['Alaska', 'Alabama', 'Arkansas', 'Arizona', 'California', 'Colorado', 'Connecticut', 
   'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Iowa', 'Idaho', 'Illinois', 'Indiana', 'Kansas', 'Kentucky', 
   'Louisiana', 'Massachusetts', 'Maryland', 'Maine', 'Michigan', 'Minnesota', 'Missouri', 'Mississippi', 'Montana', 
   'North Carolina', 'North Dakota', 'Nebraska', 'New Hampshire', 'New Jersey', 'New Mexico', 'Nevf3ada', 'New York', 
   'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 
   'Texas', 'Utah', 'Virginia', 'Vermont', 'Washington', 'Wisconsin', 'West Virginia', 'Wyoming'];

   highest = 0;
   highestState = '';
   for state in listOfStates:
      curState = states[state];
      nbr = naive_bayes_final(states, "someWord", curState);
      #multiply NP by another NP to get multiple words, times pState
      if nbr > max:


   print(states);
   print(parties);
         

if __name__ == '__main__':
   main();

#ok do machine learning stuff...
#use machine learning and some training data to determine whether  or not tweets fall into a certain cateofry... then run aspect-based on those assigned categoeis