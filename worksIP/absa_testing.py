from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd 
import numpy as np 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from IPython.display import display
#import gensim;
#import spacy;
import aspect_based_sentiment_analysis as absa

topics = ['economics', 'police', 'foreign_policy', "president", "immigration", "military", "abortion"]

topicsDict = {
    'economics': ['consumption', 'commerce', 'economics', 'economic', 'trade', 'gdp', 'investment' +
        'stock market', 'stocks', 'stock', 'goods', 'financial', 'fiscal', 'economical', 'profitable' +
        'economy', 'efficent', 'finance', 'monetary', 'management', 'economist', 'macroeconomics' + 
        'microeconomics', 'protectionism', 'resources', 'real value', 'nominal value', 'capital', 'markets'],
    'police': ['blm', 'defund', 'police', 'militiarization', 'police officer', 'sheriff', 'crime' +
        'fatal', 'shooting', 'abuse of power', 'line of duty', 'protect', 'protecting', 'patrol',
        'patrolling', 'law enforcement', 'riot', 'looting', 'arrest', 'racism', 'law', 'black lives matter'],
    'foreign_policy': ['imperialism', 'occupation', 'un', 'united nations', 'united', 'hrc' +
        'who', 'free trade', 'anarchy', 'nationalism', 'foreign', 'china', 'russia', 'cuba' +
        'multinational', 'regional', 'trade', 'international', 'commerce', 'alien', 'refugee' + 
        'border', 'ambassador', 'israel', 'pakistan', 'terrorism'],
    "immigration": ['alien', 'wall', 'emigration', 'immigration', 'migration', 'illegal alien' + 
        'naturalization', 'visa', 'citizenship', 'refugee', 'welfare', 'family reunification', 'border' + 
        'immigrants', 'enforcement', 'seperation', 'asylum', 'sanctuary city', 'sancuary cities'],
    "president": ['trump', 'president', 'genius', 'smart', 'incompatent', 'idiot', 'leadership' +
        'cabinet', 'election', 'biden', 'elect', 'harris', 'joe biden', 'donald trump', 'government' +
        'corrupt', 'russia', 'head of state', 'presidency'],
    "military": ['military', 'armed forces', 'air force', 'coast guard', 'national guard', 'army' +
        'navy', 'marines', 'combat', 'forces', 'invasion', 'occupation', 'overseas', 'over seas' +
        'foreign policy', 'defense', 'intelligence', 'military intelligence', 'militaristic', 'militia' +
        'peacekeeping', 'occupy', 'regiment', 'noncombatant', 'naval'],
    "abortion": ['abortion', 'birth control', 'contraceptives', 'condoms', 'abortion laws', 'feticide' +
        'abortion clinic', 'pro-choice', 'pro choice', 'prochoice', 'abortion pill', 'trimester' +
        'first trimester', 'planned parenthood']
}

#change this to a dictionary

#recognizer = absa.probing.AttentionGradientProduct()
#nlp = absa.load('absa/classifier-rest-0.1', pattern_recognizer=recognizer)
def justScores():
    nlp = absa.load();

    tweet1 = "I think the President is doing an excellent job handling the ongoing issues with China. Despite the nation's desperate attempts to interfere with our economy, our military has protected important trade regions and ensured financial success for years to come."
    tweet2 = "Our President is a corrupt and terrible man. This is evident in how he has treated the leaders of numerous other nations, handled the nation's economy during the COVID situation, and protected corrupt police from justice." 

    results = nlp(tweet1, aspects=topics)
    mySents = []
    mySentsV2 = []
    for topic in topics:
        myChoice = topicsDict[topic]
        haveFound = False;
        for word in myChoice:
            if (word in tweet1) and not haveFound:
                if results[topic].sentiment == absa.Sentiment.negative:
                    mySents.append(1);
                elif results[topic].sentiment == absa.Sentiment.positive:
                    mySents.append(2);
                mySentsV2.append(np.round(results[topic].scores, decimals=3))
                #html = absa.probing.explain(results[topic])
                #display(html)
                haveFound = True;
                print('----------------------------------------------------------------------------------')
        if not haveFound:
            mySents.append(0)
            mySentsV2.append(0) #neutral

    print(mySents);
    print(mySentsV2);


def explainations():
    recognizer = absa.probing.AttentionGradientProduct()
    nlp = absa.load('absa/classifier-rest-0.1', pattern_recognizer=recognizer)
    text = ("I think the President is doing an excellent job handling the ongoing issues with China. Despite the nation's desperate "
    "attempts to interfere with our economy, our military has protected important trade regions and ensured financial success for years to come.")
            
    task = nlp(text, aspects=['economics', 'police', 'foreign_policy', "president", "immigration", "military", "abortion"])

    econ, poli, fp, pres, immi, milit, abor = task.batch
    econ_scores = np.round(econ.scores, decimals=3)

    print(f'Sentiment for "slack": {repr(econ.sentiment)}')
    print(f'Scores (neutral/negative/positive): {econ_scores}')

    html = absa.probing.explain(econ)
    display(html)

justScores();