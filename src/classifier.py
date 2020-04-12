### TextBlob Built-In Naive Bayes Classification System

from textblob.classifiers import NaiveBayesClassifier

import csv

myList = [];
trainingData = [];

with open("data12_02_2020_11-20.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        if i % 2 == 0:
            myList.append(line[0])

for line in myList:
    info = line.split(',');
    tweet = info[2]
    polar = info[len(info) - 2]
    c = tuple([tweet, polar]);
    trainingData.append(c);

cl = NaiveBayesClassifier(trainingData)

print(cl.classify("The big Oil Deal with OPEC Plus is done. This will save hundreds of thousands of energy jobs in the United States. I would like to thank and congratulate President Putin of Russia and King Salman of Saudi Arabia. I just spoke to them from the Oval Office. Great deal for all!"))

# From Donald Trump, Should Predict 'R'

print(cl.classify("No Montanan should have to choose between paying medical bills and putting food on the table for their families during a public health crisis. I fought to expand SNAP benefits to ensure no one in our state goes hungry as we combat this crisis."))

# From Democractic Senator Tester, Should Predict 'D' 


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
    polar = info[len(info) - 2]
    c = tuple([tweet, polar]);
    test.append(c);


h1 = test[0:round(len(test) * .5)]
h2 = test[round(len(test) * .5) + 1:len(test) - 1]

print(cl.accuracy(h1))
print(cl.accuracy(h2))

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