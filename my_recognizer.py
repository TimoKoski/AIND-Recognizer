import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # Valuable insight from 
    # https://discussions.udacity.com/t/recognizer-implementation/234793
    # https://discussions.udacity.com/t/recognizer-implementation/234793/27
    
    test_sequences = list(test_set.get_all_Xlengths().values())
    for test_X, test_Xlength in test_sequences:
        prob = {}
        score = float('-inf')
        best_score = float('-inf')
        guess = ""
        for model_word, model in models.items():
            try:
                score = model.score(test_X, test_lengths)
                prob[model_word] = score
            except:
                prob[model_word] = float('-inf')
            if score > best_score:
                guess = model_word
                best_score = score    
            
        probabilities.append(prob)
        #print(probabilities)
        guesses.append(guess)
        
    return probabilities, guesses
    #raise NotImplementedError
