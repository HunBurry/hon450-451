### TextBlob Built-In Naive Bayes Classification System

from textblob.classifiers import NaiveBayesClassifier

import csv
    

myList = [];
trainingData = [];
trainingFiles = ['data12_04_2020_22-58.csv', 'data12_02_2020_11-20.csv']

for fName in trainingFiles:
    with open(fName, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            if i % 2 == 0:
                myList.append(line[0])

for line in myList:
    info = line.split(',');
    tweet = info[2]
    polar = info[len(info) - 1]
    c = tuple([tweet, polar]);
    trainingData.append(c);

print(len(trainingData))

myData1 = trainingData[0:1000]
trainingData = trainingData[1000:len(trainingData)]

cl = NaiveBayesClassifier(myData1)

for i in range(0, len(trainingData), 50):
    chunk = trainingData[i:i + 50]
    print(i)
    cl.update(chunk)

print(cl.show_informative_features());

myList2 = []
test = []

with open("data05_02_2020_11-17.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        if i % 2 == 0:
            myList2.append(line[0])

for line in myList2:
    info = line.split(',');
    tweet = info[2]
    polar = info[len(info) - 1]
    c = tuple([tweet, polar]);
    test.append(c);


for i in range(0, len(trainingData), 100):
    chunk = trainingData[i:i + 100]
    print(cl.accuracy(chunk))

# must be split up to avoid memory errors

'''
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