def naive_bayes_final(stateDict, word, state):
   '''
   p(state) -> numInState/numInCountry
   p(word|state) -> numOfTermInState/numOfTermTotal
   p(word) -> numOfTerm/numInCountry
   '''

   wordsInState = 0;
   for key in stateDict[state].keys():
      wordsInState = wordsInState + stateDict[state][key];

   if word not in stateDict[state].keys():
      return 0;
   numInState = stateDict[state][word];
   numOfTermTotal = 0;
   numTotal = 0; 
   for key in stateDict.keys():
      if word in stateDict[key].keys():
         numOfTermTotal = numOfTermTotal + stateDict[key][word];
      for subKey in stateDict[key].keys():
         numTotal = numTotal + stateDict[key][subKey];

   pWordState = numInState / wordsInState;
   pWord = numOfTermTotal / numTotal;
   pState = wordsInState / numTotal;

   print("wordsInState: " + str(wordsInState));
   print(pState);
   print("___________________")

   print("numInState: " + str(numInState));
   print("numOfTermTotal: " + str(numOfTermTotal));
   print(pWordState);
   print("___________________");

   print("numTotal: " + str(numTotal));
   print(pWord);

   finalProb = (pState * pWordState) / pWord;

   return finalProb;

def naive_bayes(dictionary, term, state, totalInState, totalInCountry):
   #count of term in state/ total words in state * (# of people from state / total ppl)
   # all that divided by number of term / total words 
   if term in dictionary[state].keys():
      numOfTermInCountry = 0;
      numOfWordsInCountry = 0;
      numOfTermOccurences = dictionary[state][term];
      numOfTotalStateOccurences = 0;
      for key in dictionary[state].keys():
         numOfTotalStateOccurences = numOfTotalStateOccurences + dictionary[state][key];
      for key in dictionary.keys():
         for subKey in dictionary[key].keys():
            if subKey == term:
               numOfTermInCountry = numOfTermInCountry + dictionary[key][subKey];
            numOfWordsInCountry = numOfWordsInCountry + dictionary[key][subKey];
      return (((numOfTermOccurences / numOfTotalStateOccurences)  * (totalInState / totalInCountry)) / (numOfTermInCountry / numOfWordsInCountry));
   else:
      return 0;
   #if numOfTermInCountry = 0; return 1;