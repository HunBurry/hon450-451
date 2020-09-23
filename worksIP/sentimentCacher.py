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
        'first trimester', 'planned parenthood']
}

data = pd.read_csv('../data/data12_04_2020_22-58.csv')
data = data.drop(data.columns[[0, 1, 3, 4, 6]], axis=1)
data = data.dropna();
data.columns = ['tweet', 'party']
data.reset_index(drop=True, inplace=True)

aspectsArray = []
for key in topics.keys():
    for item in topics[key]:
        aspectsArray.append(item)
nlp = absa.load();
for i in range(10):
    print('')

print(aspectsArray)

numberOfLines = 0;
if path.exists("sentiments.txt"):
    numberOfLines = len(open("sentiments.txt").readlines())

for row in range(len(data) - numberOfLines):
    rowIQ = data['tweet'][row + numberOfLines];
    results = nlp(rowIQ, aspects=aspectsArray)
    myFile = open("sentiments.txt", "a")
    myString = '';
    for term in aspectsArray:
        if term in rowIQ:
            if results[term].sentiment == absa.Sentiment.negative:
                myString = myString + "1 ";
            elif results[term].sentiment == absa.Sentiment.positive:
                myString = myString + "2 ";
        else:
            myString = myString + "0 ";
    myFile.write(myString + '\n')
    myFile.close()