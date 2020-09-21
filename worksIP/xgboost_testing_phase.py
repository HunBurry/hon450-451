from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import pandas as pd
import aspect_based_sentiment_analysis as absa
from xgboost import XGBClassifier
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import numpy as np;
from os import path, remove;
warnings.filterwarnings("ignore")

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

    data = pd.DataFrame(data, columns=['text', 'type'])

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
    dataframe['parsed_text'] = dataframe['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

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
    bagOfWords = bow_vectorizer.fit_transform(dataframe['text'])

    x_train = bagOfWords.todense();
    y_train = dataframe['type'];

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
    tfidf_matrix = tfidf.fit_transform(dataframe['text'])

    x_train = tfidf_matrix.todense();
    y_train = dataframe['type'];

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
    '''
    aspectsArray = topics.keys();
    nlp = absa.load();
    overall = []

    for row in range(len(dataframe)):
        rowIQ = dataframe['text'][row];
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
        rowIQ = dataframe['text'][row];
        results = nlp(rowIQ, aspects=aspectsArray)
        mySents = [];
        for term in aspectsArray:
            if term in rowIQ:
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

def createClassifier(bagOfWords, sentiments, y_train):
    x_train = np.append(bagOfWords, sentiments, axis=1);
    
    model_bow = XGBClassifier(random_state=9,learning_rate=0.9)
    model_bow.fit(x_train, y_train)

    return model_bow;

###########################################################################

def removeData():
    '''
    Removes data files.
    '''
    if path.exists("data.csv"):
        remove("data.csv")
    if path.exists("sentiments.txt"):
        remove("sentiments.txt")

###########################################################################

def main():
    topics = { 'animals' : ["lions", "boas", "elephants", "dillos"] ,
        'cars' : ["tesla", "audi", "toyota", "bmw"] }

    Acomments = [ 'lions and boas are good',
        'elephants are bad',
        'dillos are terrible',
        'i love tesla' ]

    Bcomments = [ 'boas are nice', 
        'elephants are amazing', 
        'lions are cool',
        'dillos are great, but tesla stinks',
        'audi and toyota are my favorite' ]
    
    if path.exists("data.csv") and path.exists("sentiments.txt"):
        data = pd.read_csv('data.csv', sep=' ', header=None);
        data.columns = ['text', 'type'];
        sentiments = np.genfromtxt("sentiments.txt");
    else:
        data = createTrain(topics, Acomments, Bcomments, True);
        sentiments = aspectify(topics, data, True);
    bagOfWords, y_train, vectorizer = vectorize(data);

    xgb_classifier = createClassifier(bagOfWords, sentiments, y_train);

    ########################################################
    
    userInput = input("Give me a tweet to test (use enter or 'F' to close): ")
    while (userInput != 'F' and userInput != ""):

        bagOfWordsUser = vectorizer.transform([userInput])
        bagOfWordsUser = bagOfWordsUser.todense()
        sentimentsUser = np.asmatrix(aspectify(topics, pd.DataFrame({"text": [userInput]}), False));
        totalInformation = np.append(bagOfWordsUser, sentimentsUser, axis=1); #error here
        results = xgb_classifier.predict(totalInformation)

        print("Prediction: " + results);

        userInput = input("Give me a tweet to test (use enter or 'F' to close): ")

if __name__ == "__main__":
    main();