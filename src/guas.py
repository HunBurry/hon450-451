from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#run the location analysis and then the party analysis, or vice versa? 
X, y = load_iris(return_X_y=True)
print(X);
print("____________");
print(y);
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(y_pred);

def states_to_num(state):
    list_of_states = {'Alaska': 1, 'Alabama': 2, 'Arkansas': 3, 'Arizona': 4, 'California': 5, 'Colorado': 6, 
    'Connecticut': 7, 'Delaware': 8, 'Florida': 9, 'Georgia': 10, 'Hawaii': 11, 'Iowa': 12, 'Idaho': 13, 
    'Illinois': 14, 'Indiana': 15, 'Kansas': 16, 'Kentucky': 17, 'Louisiana': 18, 'Massachusetts': 19, 'Maryland': 20, 
    'Maine': 21, 'Michigan': 22, 'Minnesota': 23, 'Missouri': 24, 'Mississippi': 25, 'Montana': 26, 'North Carolina': 27, 
    'North Dakota': 28, 'Nebraska': 29, 'New Hampshire': 30, 'New Jersey': 31, 'New Mexico': 32, 'Nevf3ada': 33, 'New York': 34, 
    'Ohio': 35, 'Oklahoma': 36, 'Oregon': 37, 'Pennsylvania': 38, 'Rhode Island': 39, 'South Carolina': 40, 'South Dakota': 41, 
    'Tennessee': 42, 'Texas': 43, 'Utah': 44, 'Virginia': 45, 'Vermont': 46, 'Washington': 47, 'Wisconsin': 48, 'West Virginia': 49, 
    'Wyoming': 50};
    return list_of_states(state);


#run naive bayes for each state? 

def naiveBayesLocation(dataX, dataY):
    '''
    Data X = [highestSentiment, highestSentimentParty, lowestSentiment, lowestSentimentParty, 
    termsInParty, termsInCountry];
    '''
    #takes data, returns lcoation -> location must be classification 1-50
    #[hSent, hSentP, lSent, lSentP, termsIP, termsIC, termsIN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0);
    gnb = GaussianNB();
    y_pred = gnb.fit(X_train, y_train).predict(X_test);


def naiveBayesParty(dataX, dataY):
    '''
    Data X = [highestSentiment, highestSentimentParty, lowestSentiment, lowestSentimentParty,
    ]
    '''
    #takes data, returns party -> party 0 (demo), 1 (repo);
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0);
    gnb = GaussianNB();
    y_pred = gnb.fit(X_train, y_train).predict(X_test);



'''
x   x   x   x   class  

1.3 3.2 23  423 'RED'





HIGHEST SENT / PARTY, LOWEST SENT / PARTY, NUM IN STATE, AVERAGE SENT, NUM IN STATE, NUM IN PARTY, PARTY, STATE, NUM IN TOTAL -> PROBABILITY THAT WORD IS CHOSEN


two pronged approach - 

two layered naive bayes approach 

'''
