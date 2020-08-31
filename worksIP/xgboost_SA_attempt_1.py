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

train = pd.read_csv('../data/data12_04_2020_22-58.csv')

train_original = train.copy()

test = pd.read_csv('../data/data12_02_2020_11-32.csv')

test_original = test.copy()

train = train.drop(train.columns[[1, 3, 4, 6]], axis=1)
test = test.drop(test.columns[[1, 3, 4, 6]], axis=1)
test = test.dropna();
train = train.dropna();

train.columns = ['id', 'tweet', 'party']
test.columns = ['id', 'tweet', 'party']

################################################################################################################################

combine = train.append(test, ignore_index=True, sort=True)

combine['parsed_tweet'] = combine['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

tokenized_tweet = combine['parsed_tweet'].apply(lambda x: x.split())
ps = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combine['parsed_tweet'] = tokenized_tweet

###############################################################################################################################

all_words_repub = ' '.join(text for text in combine['parsed_tweet'][combine['party'] == 'R'])
all_words_demo = ' '.join(text for text in combine['parsed_tweet'][combine['party'] == 'D'])

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bagOfWords = bow_vectorizer.fit_transform(combine['parsed_tweet'])
df_bow = pd.DataFrame(bagOfWords.todense())
train_bow = bagOfWords[0:int(round(bagOfWords.shape[0] * .7))]
train_bow.todense()

tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')
tfidf_matrix=tfidf.fit_transform(combine['parsed_tweet'])
df_tfidf = pd.DataFrame(tfidf_matrix.todense())
train_tfidf_matrix = tfidf_matrix[0:int(round(tfidf_matrix.shape[0] * .7))]
train_tfidf_matrix.todense()

x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(train_bow, train['party'], test_size=0.3, random_state=2)
x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix, train['party'], test_size=0.3, random_state=17)

################################################################################################################################

log_reg = LogisticRegression(random_state=5,solver='lbfgs')
log_reg.fit(x_train_bow, y_train_bow)

prediction_bow = log_reg.predict_proba(x_valid_bow)
prediction_int = prediction_bow[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
log_bow = f1_score(y_valid_bow, prediction_int)

log_reg.fit(x_train_tfidf, y_train_tfidf)
prediction_tfidf = log_reg.predict_proba(x_valid_tfidf)
prediction_int = prediction_tfidf[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
log_tfidf = f1_score(y_valid_tfidf, prediction_int)

###############################################################################################################################

model_bow = XGBClassifier(random_state=9,learning_rate=0.9)
model_bow.fit(x_train_bow, y_train_bow)
xgb = model_bow.predict_proba(x_valid_bow)

xgb=xgb[:,1]>=0.3
xgb_int=xgb.astype(np.int)
xgb_bow=f1_score(y_valid_bow,xgb_int)

model_tfidf = XGBClassifier(random_state=38,learning_rate=0.9)
model_tfidf.fit(x_train_tfidf, y_train_tfidf)
xgb_tfidf=xgb_tfidf[:,1]>=0.3
xgb_int_tfidf=xgb_tfidf.astype(np.int)
score=f1_score(y_valid_tfidf,xgb_int_tfidf)

###############################################################################################################################









###should i go back and get hashtags too, instead of removing them? should i add some more people to the testing database (i.e. celeberties)