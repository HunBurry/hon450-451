### TextBlob Built-In Naive Bayes Classification System

from textblob.classifiers import NaiveBayesClassifier;
import pandas as pd;
import time;
import statistics;


def createTestTrain(listOfFiles):
    holdData = [];

    for fileName in listOfFiles:
        df = pd.read_csv(fileName, skip_blank_lines=True)
        df.columns = ['index', 'twitterID', 'tweetText', 'polarity', 'nounPhrases', 'party', 'state'];
        df.dropna(axis=0);
        #print(df.head(50))
        for index, row in df.iterrows():
            holdData.append(tuple([row['tweetText'], row['party']]));

        df['train'] = df[['tweetText', 'party']].apply(tuple, axis=1)

        aTrain = df['train'].values.tolist()

        return aTrain;

    return holdData;

def chunk(data, mode, classificationS):
    length = len(data);
    curPos = 0;
    classifier = None;
    if classificationS is not None:
        classifier = classificationS;

    if mode == "train":
        while curPos <= length:
            if curPos == 0:
                d = data[0:50]
                for i in d:
                    print(i)
                classifer = NaiveBayesClassifier(data);
                curPos = 50;
            else:
                if curPos + 50 >= length:
                    classifer.update(data[curPos:length]);
                else:
                    classifier.update(data[curPos:curPos + 50])
                curPos = curPos + 50;
                time.sleep(2);
        return classifier;

    elif mode == 'test':
        listOfAccs = [];
        while curPos <= length:
            if curPos + 50 >= length:
                listOfAccs.append(classifier.accuracy(data[curPos:length]));
            else:
                listOfAccs.append(classifier.accuracy(data[curPos:curPos + 50]));
            curPos = curPos + 50;
            time.sleep(2);
        return listOfAccs;

def main():
    trainingFiles = ["data07_10_2020_22-08.csv"];
    trainingData = createTestTrain(trainingFiles);
    testingFiles = ["data31_08_2020_09-34.csv"];
    testingData = createTestTrain(testingFiles);

    classifier = chunk(trainingData, 'train', None);
    accuracies = chunk(testingData, 'test', classifier);

    #print(classifier.show_informative_features());
    print("Average Accuracy: " + str(statistics.mean(accuracies)));

if __name__ == "__main__":
    main();





'''
## possible ways to increase this -> my custom Naive Bayes, giving it more data, or combining it with ML and more data

"""print(cl.classify("The big Oil Deal with OPEC Plus is done. This will save hundreds of thousands of energy jobs in the United States. I would like to thank and congratulate President Putin of Russia and King Salman of Saudi Arabia. I just spoke to them from the Oval Office. Great deal for all!"))

# From Donald Trump, Should Predict 'R'

print(cl.classify("No Montanan should have to choose between paying medical bills and putting food on the table for their families during a public health crisis. I fought to expand SNAP benefits to ensure no one in our state goes hungry as we combat this crisis."))

# From Democractic Senator Tester, Should Predict 'D' 
"""

                          dotprod(weights, encode(fs,label))
prob(fs|label) = ---------------------------------------------------
                 sum(dotprod(weights, encode(fs,l)) for l in labels)


>>> def end_word_extractor(document):
...     tokens = document.split()
...     first_word, last_word = tokens[0], tokens[-1]
...     feats = {}
...     feats["first({0})".format(first_word)] = True
...     feats["last({0})".format(last_word)] = False
...     return feats
>>> features = end_word_extractor("I feel happy")
>>> assert features == {'last(happy)': False, 'first(I)': True}

'''