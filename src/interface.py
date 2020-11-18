import pickle
import xgboost
from sklearn.linear_model import LogisticRegression
import hybrid_SA
import twitterAPI
import tweepy
import sentimentCacher
import pandas as pd
import numpy as np
import os

def single_tweet_analysis(tweet):
    '''
    Parameters:
        tweet: 
            Type = String
            Tweet text to predict on.
    Takes a tweet and predicts the results based on saved models. 
    '''

    if len(os.listdir('../data/user_models')) == 3:
        loaded_vectorizer_model = pickle.load(open('../data/user_models/vectorizer_model', 'rb'))
        loaded_xgb_model = pickle.load(open('../data/user_models/xgboost_model', 'rb'))
        loaded_log_reg_model = pickle.load(open('../data/user_models/log_reg_model', 'rb'))
    else:
        loaded_vectorizer_model = pickle.load(open('../data/premade_model/vectorizer_model', 'rb'))
        loaded_xgb_model = pickle.load(open('../data/premade_model/xgboost_model', 'rb'))
        loaded_log_reg_model = pickle.load(open('../data/premade_model/log_reg_model', 'rb'))

    data = pd.DataFrame(['tweet': tweet]);
    bagOfWords = loaded_vectorizer_model.transform(data['tweet'])
    bagOfWords = bagOfWords.todense();
    sentiments = sentimentCacher.single_row_sents(data['tweet']).to_numpy();
    totalInformation = np.append(bagOfWords, sentiments, axis=1);
    xgb_prediction = loaded_xgb_model.predict(totalInformation);
    log_reg_predicition = loaded_log_reg_mdel.predict(totalInformation);

    print("The XGBoost model predicts that the tweet is " + xgb_prediction + ".")
    print("The XGBoost model predicts that the tweet is " + log_reg_prediction + ".")

def single_user_analysis(username):
    '''
    Parameters:
        username: 
            Type = String
            Twitter username to predict on.
    Takes a Twitter user and predicts the results based on saved models and the last five tweets from the user.
    '''


def main():
    print("Starting program...");
    while (true):
        input1 = input("Would you like to 1) Start the pipeline, or 2) run tweet anlysis on training data (1/2)? ")
        if input1 == '1':
            input2 = input("Are you sure you would like to do this? Pipeline processes may take upwards of 12 hours (Y/N). ")
            if lower(input2) == 'y':
                doSomethingAsyncHere();
                print("Pipeline completed, and models have been trained.")
    
        elif input1 == '2':
            input3 = input("Would you like to 1) run analysis on a single tweet/text, or 2) pull from a live feed (1/2)? ")
            if input3 == '1':
                input4 = input("Please enter the text of the tweet. ")
                single_tweet_analysis(input4);
            elif input3 == '2':

        else:
            print("Invalid response.")





if __name__ == '__main__':
    main();
