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
warnings.filterwarnings("ignore")

def main(): 
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

    combine['parsed_tweet'] = tokenized_tweet;

    all_words_repub = ' '.join(text for text in combine['parsed_tweet'][combine['party'] == 'R'])
    all_words_demo = ' '.join(text for text in combine['parsed_tweet'][combine['party'] == 'D'])

    #extract_words = pos_tagging(combine)

    ###############################################################################################################################

    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, stop_words='english')
    bagOfWords = bow_vectorizer.fit_transform(combine['parsed_tweet'])

    tfidf = TfidfVectorizer(max_df=0.90, min_df=2, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combine['parsed_tweet'])

    x_train_bow = bagOfWords.todense();
    y_train_bow = combine['party']

    x_train_tfidf = tfidf_matrix.todense();
    y_train_tfidf = combine['party']

    ################################################################################################################################

    recentTrumpTweet = []
    initialTweet = "We had FAR more people (many millions) watching us at the RNC than did Sleepy Joe and the DNC, and yet an ad just ran saying the opposite. This is what we’re up against. Lies. But we will WIN!"
    parsed = ' '.join([w for w in initialTweet.split() if len(w)>2])
    parsed = parsed.split();
    recentTrumpTweet = [ps.stem(i) for i in parsed]
    recentTrumpTweet = ' '.join([word for word in recentTrumpTweet])
    print(recentTrumpTweet)
    bagOWords2 = bow_vectorizer.transform([recentTrumpTweet])

    #log_reg = LogisticRegression(random_state=5,solver='lbfgs')
    #log_reg.fit(x_train_bow, y_train_bow)

    #prediction_bow = log_reg.predict_proba(bagOWords2)
    #prediction_int = prediction_bow[:,1]>=0.3
    #prediction_int = prediction_int.astype(np.int)
    #log_bow = f1_score(y_valid_bow, prediction_int)

    #log_reg.fit(x_train_tfidf, y_train_tfidf)
    #prediction_tfidf = log_reg.predict_proba(bagOWords2)
    #prediction_int = prediction_tfidf[:,1]>=0.3
    #prediction_int = prediction_int.astype(np.int)
    #log_tfidf = f1_score(y_valid_tfidf, prediction_int)

    ###############################################################################################################################

    topics = ['economics', 'police', 'foreign_policy', "president" +
        "immigration", "military", "abortion"]

    topicsDict = {
        'economics': ['consumption', 'commerce', 'economics', 'economic', 'trade', 'gdp', 'China', 'investment' +
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
    nlp = absa.load();
    overall = [];
    print(type(combine))
    for row in range(len(x_train_bow)):
        print(combine['tweet'])
        results = nlp(combine['tweet'][row], aspects=topics)
        mySents = []
        mySentsV2 = []
        for topic in topics:
            myChoice = topicsDict[topic]
            haveFound = False;
            for word in myChoice:
                if (word in combine[row]['tweet']) and not haveFound:
                    if results[topic].sentiment == absa.Sentiment.negative:
                        mySents.append(1);
                    elif results[topic].sentiment == absa.Sentiment.positive:
                        mySents.append(2);
                    mySentsV2.append(np.round(results[topic].scores, decimals=3))
                    haveFound = True;
            if not haveFound:
                mySents.append(0)
                mySentsV2.append(0) #neutral
        overall.append(mySents);
    x_train_bow = np.append(x_train_bow, overall);

    model_bow = XGBClassifier(random_state=9,learning_rate=0.9)
    model_bow.fit(x_train_bow, y_train_bow)
    xgb = model_bow.predict_proba(bagOWords2)
    print(xgb)

    #xgb=xgb[:,1]>=0.3
    #xgb_int=xgb.astype(np.int)
    #xgb_bow=f1_score(y_valid_bow,xgb_int)

    #model_tfidf = XGBClassifier(random_state=38,learning_rate=0.9)
    #model_tfidf.fit(x_train_tfidf, y_train_tfidf)
    #tfidfres = model_tfidf.predict_proba(bagOWords2)
    #print(tfidfres)
    #xgb_tfidf=xgb_tfidf[:,1]>=0.3
    #xgb_int_tfidf=xgb_tfidf.astype(np.int)
    #score=f1_score(y_valid_tfidf,xgb_int_tfidf)

    ###############################################################################################################################

if __name__ == '__main__':
    main();


#get pos/neg numbers for each sentiment
    #add to darafgrame

#get kewywords
#associate with sentiment
#zero everything else otu





#make examples/test cases to verify working stuff
#make an example for every topic, maybe some with multiple as well
#fix the sbda to not append 0s each time
#make matrix initially, update matrix then push to append