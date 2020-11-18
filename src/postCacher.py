import os;
import glob;
import numpy as np;
import pandas as pd;

def convert(fileName):
    '''
    Converts a data file and a sentiment file into one combined file.
    '''
    if os.path.exists(fileName):
        print(fileName)
        date = fileName[4:-1];
        data = pd.read_csv('../data/' + fileName);
        nan_value = float("NaN")
        print(data)
        data.replace("", nan_value, inplace=True)
        data = data.dropna();
        print(data.head(5));
        newFileName = '../data/completedProcessing' + date + ".csv";
        data.to_csv(newFileName)
    '''
    if os.path.exists(fileName):
        date = fileName[4:-1];
        sentimentData = "sentiments" + date;
        data = pd.read_csv('../data/' + fileName);
        data = data.drop(data.columns[[0, 1, 3, 4, 6]], axis=1)
        data = data.dropna();
        sentiments = pd.read_csv('../data/' + sentimentData);
        print(data.head(5));
        print(sentiments.head(5));
        if os.path.exists(toDelete):
            deletionList = np.genfromtxt(toDelete, delimiter=', ').tolist();
            sentiments.drop(deletionList);
            data.drop([i + 1 for i in deletionList]);
        newFileName = '../data/completedProcessing' + date + '.csv';
        pd.to_csv(newFileName)
    '''

def combine():
    filesToConcat = [];
    for file in os.listdir("../data"):
        if file.endswith(".csv") and 'completedProcessing' in file:
            filesToConcat.append(pd.read_csv("../data/" + file));
    overall = pd.concat(filesToConcat);
    print(overall);
    overall.to_csv('../data/combinedSentiments.csv');

def main():
    print("Converting sentiment/data files into single files...");
    print("Deleting forgien language based tweets...");
    for file in os.listdir("../data"):
        if file.endswith(".csv") and 'sentiments' in file:
            convert(file);
    print("Concatinating all sentiment files...");
    combine();
    print("combinedSentiments.csv file created/overwritten. Process completed.")

if __name__ == '__main__':
    main();