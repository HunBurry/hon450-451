from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import pandas as pd
import aspect_based_sentiment_analysis as absa
from xgboost import XGBClassifier
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import numpy as np;
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from os import path, remove;

def getTopics():
    '''
    Returns all topics that ABSA is currently being ran on. 
    '''
    topics = {
        'economics': ['consumption', 'commerce', 'economics', 'economic', 'trade', 'gdp', 'China', 'investment', 
            'stock market', 'stocks', 'stock', 'goods', 'financial', 'fiscal', 'economical', 'profitable',
            'economy', 'efficent', 'finance', 'monetary', 'management', 'economist', 'macroeconomics',
            'microeconomics', 'protectionism', 'resources', 'real value', 'nominal value', 'capital', 'markets'],
        'police': ['blm', 'defund', 'police', 'militiarization', 'police officer', 'sheriff', 'crime', 
            'fatal', 'shooting', 'abuse of power', 'line of duty', 'protect', 'protecting', 'patrol',
            'patrolling', 'law enforcement', 'riot', 'looting', 'arrest', 'racism', 'law', 'black lives matter'],
        'foreign_policy': ['imperialism', 'occupation', 'un', 'united nations', 'united', 'hrc', 
            'who', 'free trade', 'anarchy', 'nationalism', 'foreign', 'china', 'russia', 'cuba',
            'multinational', 'regional', 'trade', 'international', 'commerce', 'alien', 'refugee',
            'border', 'ambassador', 'israel', 'pakistan', 'terrorism'],
        "immigration": ['alien', 'wall', 'emigration', 'immigration', 'migration', 'illegal alien',
            'naturalization', 'visa', 'citizenship', 'refugee', 'welfare', 'family reunification', 'border',
            'immigrants', 'enforcement', 'seperation', 'asylum', 'sanctuary city', 'sancuary cities'],
        "president": ['trump', 'president', 'genius', 'smart', 'incompatent', 'idiot', 'leadership',
            'cabinet', 'election', 'biden', 'elect', 'harris', 'joe biden', 'donald trump', 'government',
            'corrupt', 'russia', 'head of state', 'presidency'],
        "military": ['military', 'armed forces', 'air force', 'coast guard', 'national guard', 'army', 
            'navy', 'marines', 'combat', 'forces', 'invasion', 'occupation', 'overseas', 'over seas', 
            'foreign policy', 'defense', 'intelligence', 'military intelligence', 'militaristic', 'militia',
            'peacekeeping', 'occupy', 'regiment', 'noncombatant', 'naval'],
        "abortion": ['abortion', 'birth control', 'contraceptives', 'condoms', 'abortion laws', 'feticide',
            'abortion clinic', 'pro-choice', 'pro choice', 'prochoice', 'abortion pill', 'trimester',
            'first trimester', 'planned parenthood'],
        'general': ['healthcare', 'ppe', 'personal protective equipment', 'covid', 'health care', 'health', 'coronavirus', 'covid-19']
    }
    return topics;


def beginCache(ideal):
    '''
    When given a number of lines to run on, runs ABSA on said number of needed rows. For a more in-depth
    description of ABSA functionality within the project, see xgboost_SA_attempt2.aspectifyv2. 
    '''
    topics = getTopics();

    aspectsArray = []

    if path.exists('sentiments.csv'):
        data = pd.read_csv('sentiments.csv')
        for key in topics.keys():
            for item in topics[key]:
                aspectsArray.append(item)
                if item not in data.columns:
                    data[item] = np.nan;
    else:
        data = pd.read_csv('../data/data12_04_2020_22-58.csv')
        data = data.drop(data.columns[[0, 1, 3, 4, 6]], axis=1)
        data = data.dropna();
        data.columns = ['tweet', 'party']
        data.reset_index(drop=True, inplace=True);
        for key in topics.keys():
            for item in topics[key]:
                aspectsArray.append(item);
                data[item] = np.nan;

    nlp = absa.load();
    for i in range(10):
        print('')

    counter = 0;

    for index, row in data.iterrows():
        if counter < ideal:
            needToRun = [];
            for aspect in aspectsArray:
                if np.isnan(pd.to_numeric(row[aspect], errors='coerce')):
                    needToRun.append(aspect);
            if len(needToRun) > 0:
                results = nlp(row['tweet'], aspects=needToRun);
                for term in needToRun:
                    if term in row['tweet'].lower():
                        if results[term].sentiment == absa.Sentiment.negative:
                            data.at[index, term] = 1
                        elif results[term].sentiment == absa.Sentiment.positive:
                            data.at[index, term] = 2
                    else:
                        data.at[index, term] = 0
                counter = counter + 1

    data.to_csv('sentiments.csv', index=None, mode='w')

if __name__ == "__main__":
    ideal = int(input("How many rows of data do you want to run this on? "));
    beginCache(ideal);