from textblob.classifiers import NaiveBayesClassifier;
import pandas as pd;
import time;
import pickle;
import statistics;


def createTestTrain(listOfFiles):
    '''
    Parameters:
        listOfFiles:
            Type: Array
            Array containing the names of all files to be used in the set.
    Converts a list of files into a single array for use within the 
    training/testing phases. 
    '''
    holdData = [];

    for fileName in listOfFiles:
        df = pd.read_csv(fileName, skip_blank_lines=True)
        df.columns = ['index', 'twitterID', 'tweetText', 'polarity', 'nounPhrases', 'party', 'state'];
        df = df.dropna(axis=0); 
        for index, row in df.iterrows():
            holdData.append(tuple([row['tweetText'], row['party']]));

        df['train'] = df[['tweetText', 'party']].apply(tuple, axis=1)

        aTrain = df['train']
        aTrain = aTrain.dropna()
        aTrain = aTrain.values.tolist()

        return aTrain;

    return holdData;

def chunk(data, mode, classificationS):
    '''
    Parameters:
        data:
            Type: Array
            Dataframe containing tweets and party information. 
        mode:
            Type: String "train" or String "test"
            Determines whether or not we are training our classifier or testing the accuracy of it. 
        classificationS:
            Type: NLTK Classifier or None
            Sets a classifier if one exists for testing purposes.
    Trains/tests a NLTK Naive Bayes Classifier (NBC) on arrays. Data must be loaded
    in slowly/overtime to prevent memory errors. 
    '''
    length = len(data);
    curPos = 0;
    classifier = None;
    if classificationS is not None:
        classifier = classificationS;

    if mode == "train":
        while curPos <= length:
            if curPos == 0:
                d = data[0:50]
                classifier = NaiveBayesClassifier(d);
                curPos = 50;
            else:
                if curPos + 50 >= length:
                    classifier.update(data[curPos:length]);
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

def main(trainingFiles, testingFiles):
    '''
    Creates testing and training dataframes before using these to train 
    a NBC from the TextBlob library. Prints out accuracy, and has some
    examples that can be ran if uncommented. 
    '''
    if trainingFiles is None:
        classifier = pickle.load(open('./data/premade_models/naive_bayes_model', 'rb'))
        print("Classifier successfully loaded...")
    else:
        print("Creating training data and model...")
        trainingData = createTestTrain(trainingFiles);
        classifier = chunk(trainingData, 'train', None);
        print("Classifier created...")

    if testingFiles is None:
        testingFiles = ["./data/project_data/naive_testing_data.csv"];
    print("Creating testing data...")
    testingData = createTestTrain(testingFiles);

    print("Data created, testing accuracy now...")
    accuracies = chunk(testingData, 'test', classifier);
    print("Process completed.")
    print("Average Accuracy: " + str(statistics.mean(accuracies)));

    if trainingFiles is None:
        filename = './data/user_models/naive_bayes_model'
        pickle.dump(classifier, open(filename, 'wb'))

    '''
    Examples Below. Undocument this to show tests.
    
    print(cl.classify("The big Oil Deal with OPEC Plus is done. This will save hundreds of thousands of energy jobs in the United States. I would like to thank and congratulate President Putin of Russia and King Salman of Saudi Arabia. I just spoke to them from the Oval Office. Great deal for all!"))
    #From Donald Trump, Should Predict 'R'

    print(cl.classify("No Montanan should have to choose between paying medical bills and putting food on the table for their families during a public health crisis. I fought to expand SNAP benefits to ensure no one in our state goes hungry as we combat this crisis."))
    #From Democractic Senator Tester, Should Predict 'D' 
    '''

if __name__ == "__main__":
    main(None, None);