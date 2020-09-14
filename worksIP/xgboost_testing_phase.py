from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import pandas as pd
import aspect_based_sentiment_analysis as absa
from xgboost import XGBClassifier
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import numpy as np;
warnings.filterwarnings("ignore")

topics = { 'animals' : ["lions", "boas", "elephants", "dillos"] ,
'cars' : ["tesla", "audi", "toyota", "bmw"] }

Acomments = [ 'lions and boas are good',
'elephants are bad',
'dillos are terrible',
'i love tesla' ]

Bcomments = [ 'boas are nice', 
'dillos are great, but tesla stinks',
'audi and toyota are my favorite' ]

def testing():
    #############################################################

    data = [ item for item in [ (cmt, 'A') for cmt in Acomments ] * 10 ] \
    + [ item for item in [ (cmt, 'B') for cmt in Bcomments]]

    #############################################################

    train = pd.DataFrame(data, columns=['text', 'type'])

    train['parsed_text'] = train['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    #remove verbs? yay/nay?

    tokenized_tweet = train['parsed_text'].apply(lambda x: x.split())
    ps = SnowballStemmer("english")
    tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    train['parsed_text'] = tokenized_tweet;

    #############################################################

    bow_vectorizer = CountVectorizer(stop_words='english')
    bagOfWords = bow_vectorizer.fit_transform(train['parsed_text'])

    x_train = bagOfWords.todense();
    y_train = train['type']

    #############################################################

    aspectsArr = topics.keys()
    nlp = absa.load();
    for row in range(10):
        print("")
    overall = [];
    for row in range(len(x_train)):
        rowIQ = train['text'][row];
        results = nlp(rowIQ, aspects=aspectsArr)
        mySents = [];
        for key in aspectsArr:
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

    overall2 = np.asarray(overall);

    x_train = np.append(x_train, overall2, axis=1);

    model_bow = XGBClassifier(random_state=9,learning_rate=0.9)
    model_bow.fit(x_train, y_train)

    return [model_bow, bow_vectorizer]

def main():
    models = testing();
    xgb = models[0]
    vectorizer = models[1]

    userInput = input("Give me a tweet to test: ")

    tokenized_tweet = userInput.split();
    ps = SnowballStemmer("english")
    tokenized_tweet = [ps.stem(i) for i in tokenized_tweet]

    final = ' '.join(tokenized_tweet)
    bagOfWords = vectorizer.fit_transform([final])

    finalBoW = bagOfWords.todense();
    nlp = absa.load();
    
    aspectsArr = topics.keys()
    results = nlp(userInput, aspects=aspectsArr)
    mySents = [];
    for key in aspectsArr:
        haveFound = False;
        for term in topics[key]:
            if term in userInput and not haveFound:
                if results[key].sentiment == absa.Sentiment.negative:
                    mySents.append(1)
                elif results[key].sentiment == absa.Sentiment.positive:
                    mySents.append(2);
                haveFound = True;
        if not haveFound:
            mySents.append(0);

    print(finalBoW)
    print(mySents)

    mySents = np.asarray(mySents);

    print(mySents);
    
    newInfo = np.append(finalBoW, mySents, axis=1)

    prediction = xgb.predict(newInfo)
    print(prediction)

if __name__ == "__main__":
    main();

