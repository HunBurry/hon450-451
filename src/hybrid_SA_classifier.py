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
from os import path, remove, chdir, getcwd;
import glob
import sys;
warnings.filterwarnings("ignore")
import sentimentCacher
import pickle

###########################################################################

def tokenize(dataframe):
    '''
    Parameters
        dataframe
            Type = pd.DataFrame
            Contains all training data.
    Tokenizes and stems text to reduce variability of words.
    '''
    dataframe['parsed_text'] = dataframe['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

    tokenized_tweet = dataframe['parsed_text'].apply(lambda x: x.split())
    ps = SnowballStemmer("english")
    tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    dataframe['parsed_text'] = tokenized_tweet;
    
    return dataframe;

###########################################################################

def vectorize(dataframe):
    '''
    Parameters:
        dataframe
            Type = pd.DataFrame
            Contains all training data.
    Converts all text into a bag of words. 
    '''
    bow_vectorizer = CountVectorizer(stop_words='english')
    if 'parsed_text' in dataframe.columns:
        bagOfWords = bow_vectorizer.fit_transform(dataframe['parsed_text'])
    elif 'tweetText' in dataframe.columns:
        bagOfWords = bow_vectorizer.fit_transform(dataframe['tweetText'])
    else:
        bagOfWords = bow_vectorizer.fit_transform(dataframe['tweet'])

    x_train = bagOfWords.todense();
    y_train = dataframe['party'];

    filename = './data/user_models/vectorizer_model'
    pickle.dump(bow_vectorizer, open(filename, 'wb'))

    return x_train, y_train, bow_vectorizer;

###########################################################################

def tfidf(dataframe):
    '''
    Parameters:
        dataframe
            Type = pd.DataFrame
            Contains all training data.
    Converts all text into term frequency–inverse document frequency (TFIDF). 
    '''
    tfidf = TfidfVectorizer(stop_words='english')
    if 'parsed_text' in dataframe.columns:
        tfidf_matrix = tfidf.fit_transform(dataframe['parsed_text'])
    elif 'tweetText' in dataframe.columns:
        tfidf_matrix = tfidf.fit_transform(dataframe['tweetText'])
    else:
        tfidf_matrix = tfidf.fit_transform(dataframe['tweet'])

    x_train = tfidf_matrix.todense();
    y_train = dataframe['party'];

    return x_train, y_train, tfidf;

###########################################################################

def aspectify(topics, dataframe, writeToFile):
    '''
    Parameters:
        topics:
            Type = Dict
            Contains all topics and keywords.
        dataframe:
            type = pandas.DataFrame
            Contains all text for a given dataset.
        writeToFile: 
            Type = Boolean
            If true, write to file, no return statement. 
            If false, return the ABSA results in array form.
    Runs ABSA on all texts to find sentiments. Uses keywords to determine overall sentiment for class. 

    Depreciated.
    '''
    aspectsArray = topics.keys();
    nlp = absa.load();
    overall = []

    for row in range(len(dataframe)):
        rowIQ = dataframe['tweet'][row];
        results = nlp(rowIQ, aspects=aspectsArray)
        mySents = [];
        for key in aspectsArray:
            haveFound = False;
            for term in topics[key]:
                if term in rowIQ and not haveFound:
                    if results[key].sentiment == absa.Sentiment.negative:
                        mySents.append(1)
                    elif results[key].sentiment == absa.Sentiment.positive:
                        mySents.append(2);
                    haveFound = True;
            if not haveFound:
                mySents.append(0);
        overall.append(mySents)

    overall = np.asarray(overall);

    if writeToFile:
        np.savetxt('sentiments.txt', overall)
    return overall;

###########################################################################

def aspectifyV2(topics, dataframe, writeToFile):
    '''
    Parameters:
        topics:
            Type = Dict
            Contains all topics and keywords.
        dataframe:
            type = pandas.DataFrame
            Contains all text for a given dataset.
        writeToFile: 
            Type = Boolean
            If true, write to file, no return statement. 
            If false, return the ABSA results in array form.
    Runs ABSA on all texts to find sentiments. Uses keywords to determine sentiment per indivudal keyword. 
    '''
    aspectsArray = []
    for key in topics.keys():
        for item in topics[key]:
            aspectsArray.append(item)
    nlp = absa.load();
    overall = []

    for row in range(len(dataframe)):
        rowIQ = dataframe['tweet'][row];
        results = nlp(rowIQ, aspects=aspectsArray)
        mySents = [];
        for term in aspectsArray:
            if term in rowIQ.lower():
                if results[term].sentiment == absa.Sentiment.negative:
                    mySents.append(1)
                elif results[term].sentiment == absa.Sentiment.positive:
                    mySents.append(2);
            else:
                mySents.append(0);
        overall.append(mySents)

    overall = np.asarray(overall);

    if writeToFile:
        np.savetxt('sentiments.txt', overall)
    return overall;

###########################################################################

def createXGBClassifier(bag, dataset, y_train):
    '''
    Parameters:
        dataset:
            Type: np.Dense
            TFIDF/BoW Data
        sentiments:
            Type: np.Matrix
            All ABSA analysis for texts.
        y_train:
            Type: np.Array
            Array of all training data. 
    Returns trained XGB Classifier. 
    '''

    if 'tweet' in dataset.columns:
        dataset = dataset.drop(['tweet', 'party'], axis=1);
    elif 'tweetText' in dataset.columns:
        dataset = dataset.drop(['tweetText', 'party'], axis=1);
    else:
        dataset = dataset.drop(['parsed_tweet', 'party'], axis=1);
    
    sentiments = dataset.to_numpy();
    x_train = np.append(bag, sentiments, axis=1);
    
    xgb_model = XGBClassifier(random_state=9,learning_rate=0.3)
    xgb_model.fit(x_train, y_train)

    filename = './data/user_models/xgboost_model'
    pickle.dump(xgb_model, open(filename, 'wb'))
    return xgb_model;

###########################################################################

def createLogisticRegressor(bag, dataset, y_train):
    '''
    Parameters:
        dataset:
            Type: np.Dense
            TFIDF/BoW Data
        sentiments:
            Type: np.Matrix
            All ABSA analysis for texts.
        y_train:
            Type: np.Array
            Array of all training data. 
    Returns trained Logistic Regressor.
    '''

    if 'tweet' in dataset.columns:
        dataset = dataset.drop(['tweet', 'party'], axis=1);
    elif 'tweetText' in dataset.columns:
        dataset = dataset.drop(['tweetText', 'party'], axis=1);
    else:
        dataset = dataset.drop(['parsed_tweet', 'party'], axis=1);
    
    sentiments = dataset.to_numpy();
    x_train = np.append(bag, sentiments, axis=1);

    log_reg = LogisticRegression(random_state=5,solver='lbfgs')
    log_reg.fit(x_train, y_train)

    filename = './data/user_models/log_reg_model'
    pickle.dump(log_reg, open(filename, 'wb'))

    return log_reg;

def test_and_predictionsV2(classifier, topics, vectorizer, data):
    '''
    Parameters:
        classifier:
            Type: XGBClassifer or LogisticRegressor
            Classification system.
        topics: 
            Type = Dict
            Contains all topics and keywords.
        vectorizer:
            Type = TFIDF or Count Vectorizer (Bag of Wordss)
            Vectorization system for word frequency. 
        testingData:
            Type = String
            Filename (w/h path) to the testing data.
    Get multiple rows at once, parse them all, use that for prediction. 
    '''
    data = data.dropna();
    data = data.head(data.shape[0]);

    bagOfWords = vectorizer.transform(data['tweet']);
    bagOfWords = bagOfWords.todense();
    sentiments = data.drop(['tweet', 'party'], axis=1).to_numpy();
    totalInformation = np.append(bagOfWords, sentiments, axis=1);
    results = classifier.predict(totalInformation);

    counter = 0;
    correct = 0;
    incorrect = 0;
    corrects = {};
    incorrects = {};

    for index, row in data.iterrows():
        if results[counter] == row['party']:
            correct = correct + 1;
            if row['party'] in corrects.keys():
                corrects[row['party']] = corrects[row['party']] + 1;
            else:
                corrects[row['party']] = 1;
        else:
            incorrect = incorrect + 1;
            if row['party'] in incorrects.keys():
                incorrects[row['party']] = incorrects[row['party']] + 1;
            else:
                incorrects[row['party']] = 1;
        counter = counter + 1; 
    
    return correct, incorrect, corrects, incorrects;

###########################################################################

def main(filename, testingFilename):
    '''
    Parameters:
        filename:
            Type: String or None
            File name (not path name) to train models. If None, uses default data. 
        testingFilename: 
            Type = String or None
            File name (not path name) to test models. If None, uses default data. 
    Main function for hybrid sentiment analysis/machine learning predictions. Used to train and test models. 
    '''
    topics = sentimentCacher.getTopics();

    if filename is None:
        vectorizer = pickle.load(open('./data/premade_models/vectorizer_model', 'rb'))
        xgb_classifier = pickle.load(open('./data/premade_models/xgboost_model', 'rb'))
        logistic_regressor = pickle.load(open('./data/premade_models/log_reg_model', 'rb'))
    else:
        sentiments = pd.read_csv('./data/user_data/' + 'completedSentiments.csv');
        data = pd.read_csv('./data/user_data/' + filename);
        sentiments = sentiments.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1);
        data = data.dropna();
            
        bagOfWords, y_train, vectorizer = vectorize(data);
        #tfidf_matrix, y_train, vectorizer = tfidf(data);

        xgb_classifier = createXGBClassifier(bagOfWords, sentiments, y_train);
        #xgb_classifier = createXGBClassifier(tfidf_matrix, sentiments, y_train);
        logistic_regressor = createLogisticRegressor(bagOfWords, sentiments, y_train);
        #logistic_regressor = createLogisticRegressor(tfidf_matrix, sentiments, y_train);
    
    if not testingFilename:
        testingFilename = './data/project_data/testingData.csv';
    else:
        testingFilename = './data/user_data/' + testingFilename;

    testingData = pd.read_csv(testingFilename);

    correct, incorrect, corrects, incorrects = test_and_predictionsV2(xgb_classifier, topics, vectorizer, testingData);
    print("Out XGBoost got " + str(correct) + " out of " + str((correct + incorrect)) + " right. This equates to a " + str(correct/(correct+incorrect) * 100) + " accuracy level.")
    print("The regressor got a " + str(corrects['R']/(corrects['R']+incorrects['R']) * 100) + " percent accuracy level for Republicans, and a " + str(corrects['D']/(corrects['D']+incorrects['D']+incorrects[' D']) * 100) + " percent accuracy level for Democrats.")
    
    correct, incorrect, corrects, incorrects = test_and_predictionsV2(logistic_regressor, topics, vectorizer, testingData);
    print("Our LogisticRegressor got " + str(correct) + " out of " + str((correct + incorrect)) + " right. This equates to a " + str(correct/(correct+incorrect) * 100) + " accuracy level.")
    print("The regressor got a " + str(corrects['R']/(corrects['R']+incorrects['R']) * 100) + " percent accuracy level for Republicans, and a " + str(corrects['D']/(corrects['D']+incorrects['D']+incorrects[' D']) * 100) + " percent accuracy level for Democrats.")

if __name__ == "__main__":
    main(None, None);
