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
import sys;
warnings.filterwarnings("ignore")
import sentimentCacher
import pickle

########################################################################

def createTrain(topics, classificationA, classificationB, writeToText):
    '''
    Parameters:
        topics: 
            Type = Dict
            Contains all possible topics to run ABSA on.
        classificationA:
            Type = Array
            Type-A comments to reproduce.
        classificationB:
            Type = Array
            Type-B comments to reproduce.         
        writeToText
            Type = Boolean   
    Creates training data and saves it to a CSV file for future use. 
    '''
    data = [ item for item in [ (cmt, 'A') for cmt in classificationA] * 10 ] \
        + [ item for item in [ (cmt, 'B') for cmt in classificationB] * 10 ]

    data = pd.DataFrame(data, columns=['tweet', 'party'])

    if writeToText:
        data.to_csv('data.csv', header=None, index=None, sep=' ', mode='a')

    return data;

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
    else:
        bagOfWords = bow_vectorizer.fit_transform(dataframe['tweet'])

    x_train = bagOfWords.todense();
    y_train = dataframe['party'];

    filename = 'vectorizer_model'
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
    dataset = dataset.drop(['tweet', 'party'], axis=1);
    sentiments = dataset.to_numpy();
    x_train = np.append(bag, sentiments, axis=1);
    
    xgb_model = XGBClassifier(random_state=9,learning_rate=0.3)
    xgb_model.fit(x_train, y_train)

    filename = 'xgboost_model'
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
    dataset = dataset.drop(['tweet', 'party'], axis=1);
    sentiments = dataset.to_numpy();
    x_train = np.append(bag, sentiments, axis=1);

    log_reg = LogisticRegression(random_state=5,solver='lbfgs')
    log_reg.fit(x_train, y_train)

    filename = 'log_reg_model_getDateHere()'
    pickle.dump(log_reg, open(filename, 'wb'))

    return log_reg;

###########################################################################

def removeData():
    '''
    Removes data files.
    '''
    if path.exists("data.csv"):
        remove("data.csv")
    if path.exists("sentiments.csv"):
        remove("sentiments.csv")

###########################################################################

def test_and_predictionsV1(classifier, topics, vectorizer):
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
    Gets individual rows, parses individually, then computes prediction on each one. Overall prediction is most frequent prediction.
    '''
    data = pd.read_csv('../data/sentiments2.csv')
    data = data.dropna();
    #data.columns = ['tweet', 'party'] 
    #data = tokenize(data);

    resultsOverall = []
    for number in range(5):
        row = data.iloc[[number]];
        bagOfWords = vectorizer.transform([row['tweet']]);
        bagOfWords = bagOfWords.todense()
        sentiments = data.drop(['tweet', 'party'], axis=1).to_numpy();
        totalInformation = np.append(bagOfWords, sentiments, axis=1);
        results = classifier.predict(totalInformation);
        resultsOverall.append(results);
    prediction = max(set(resultsOverall), key = resultsOverall.count);

    return prediction;

###########################################################################

def test_and_predictionsV2(classifier, topics, vectorizer):
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
    Get multiple rows at once, parse them all, use that for prediction. 
    '''
    data = pd.read_csv('../data/sentiments2.csv')
    #data = data.drop(data.columns[[0, 1, 3, 4, 6]], axis=1)
    data = data.dropna();
    #data.columns = ['tweet', 'party'] 
    #data = tokenize(data);
    correct = 0;
    incorrect = 0;
    '''

    for index, row in data.iterrows():
        bagOfWords = vectorizer.transform([row['tweet']]);
        bagOfWords = bagOfWords.todense();
        sentiments = data.drop(['tweet', 'party'], axis=1).to_numpy();
        totalInformation = np.append(bagOfWords, sentiments, axis=1);
        results = classifier.predict(totalInformation);
        if results == data['party']:
            correct = correct + 1;
        else:
            incorrect = incorrect + 1;
    '''
        
    
    data = data.head(data.shape[0]);
    bagOfWords = vectorizer.transform(data['tweet']); #if tokenize, change to parsed_tweet
    bagOfWords = bagOfWords.todense();
    sentiments = data.drop(['tweet', 'party'], axis=1).to_numpy();
    #print(bagOfWords);
    #print(sentiments)
    totalInformation = np.append(bagOfWords, sentiments, axis=1);
    results = classifier.predict(totalInformation);
    #print(len(results))
    #print(data.shape[0])
    counter = 0;
    corrects = {};
    incorrects = {};
    for index, row in data.iterrows():
        #print(index)
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

def main():
    topics = sentimentCacher.getTopics();

    data = pd.read_csv('../data/combinedSentiments.csv')
    #print(data.columns)
    data = data.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    data = data.dropna();
    #data.columns = ['tweet', 'party']
    #data = tokenize(data);
    
    #if path.exists("sentiments.csv"):
    #    sentiments = pd.read_csv('sentiments.csv');
    #else:
    #    sentimentCacher.beginCache(100);
    #    print("Beggining cache process... Please reload when cache has been completed.")
    #    sys.exit();
        
    bagOfWords, y_train, vectorizer = vectorize(data);
    #tfidf_matrix, y_train, vectorizer = tfidf(data);

    xgb_classifier = createXGBClassifier(bagOfWords, data, y_train);
    #xgb_classifier = createXGBClassifier(tfidf_matrix, sentiments, y_train);
    logistic_regressor = createLogisticRegressor(bagOfWords, data, y_train);
    #logistic_regressor = createLogisticRegressor(tfidf_matrix, sentiments, y_train);
    
    #test_and_predictionsV1(xgb_classifier, topics, vectorizer);
    correct, incorrect, corrects, incorrects = test_and_predictionsV2(xgb_classifier, topics, vectorizer);
    print("Out XGBoost got " + str(correct) + " out of " + str((correct + incorrect)) + " right. This equates to a " + str(correct/(correct+incorrect) * 100) + " accuracy level.")
    print("The regressor got a " + str(corrects['R']/(corrects['R']+incorrects['R']) * 100) + " percent accuracy level for Republicans, and a " + str(corrects['D']/(corrects['D']+incorrects['D']+incorrects[' D']) * 100) + " percent accuracy level for Democrats.")
    #print(incorrects)
    correct, incorrect, corrects, incorrects = test_and_predictionsV2(logistic_regressor, topics, vectorizer);
    print("Our LogisticRegressor got " + str(correct) + " out of " + str((correct + incorrect)) + " right. This equates to a " + str(correct/(correct+incorrect) * 100) + " accuracy level.")
    print("The regressor got a " + str(corrects['R']/(corrects['R']+incorrects['R']) * 100) + " percent accuracy level for Republicans, and a " + str(corrects['D']/(corrects['D']+incorrects['D']+incorrects[' D']) * 100) + " percent accuracy level for Democrats.")
    #print(str(correct/(correct+incorrect)))
    #print(corrects);
    #print(incorrects)
    #res = test_and_predictionsV2(xgb_classifier, topics, vectorizer);
    #print(res)

if __name__ == "__main__":
    main();
