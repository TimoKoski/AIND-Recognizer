import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
#from test.test_bufio import lengths
#from numba.types import none


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores 
        # Additional ideas from 
        # https://discussions.udacity.com/t/how-to-start-coding-the-selectors/476905/7   
        # https://discussions.udacity.com/t/verify-results-cv-bic-dic/247347/13
        # https://discussions.udacity.com/t/bic-failing-for-some-words-and-not-other/329847     
        
        best_bic_score = float('inf') # initializing bic
        best_hmm_model = None # initializing model
        #best_n = 0
        for n in range(self.min_n_components, self.max_n_components + 1):   
            try:        
                hmm_model = GaussianHMM(n_components=n, n_iter=1000).fit(self.X, self.lengths)                
                # calculating log_l for the model 
                log_l = hmm_model.score(self.X, self.lengths)
                # determining the number of parameters
                param = (n * n) + (2 * n * len(self.X[0])) - 1
                # calculating the bic_score              
                bic_score = (-2 * log_l) + (param * np.log(len(self.X)))
                # storing values and updating if current bic_score is less than previous best
                if bic_score < best_bic_score:
                    best_bic_score = bic_score
                    best_hmm_model = hmm_model
                    #best_n = n
            except:
                pass
        #return self.base_model(best_n) 
        return best_hmm_model
        
        #raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        # Further ideas from
        # https://discussions.udacity.com/t/selectorbic-and-selectordic-errors/304157
        
        # initializing variables
        best_dic_score = float('-inf')
        best_hmm_model = None
        #best_n = 0
        for n in range(self.min_n_components, self.max_n_components + 1):   
            try:        
                hmm_model = GaussianHMM(n_components=n, n_iter=1000).fit(self.X, self.lengths)                
                # log_l of the current word
                log_l = hmm_model.score(self.X, self.lengths)
                # initializing the variable log_ls_others to accumulate log_l of other words in there
                log_ls_others = 0
                
                for word in self.hwords:
                    X, lengths = self.hwords[word]                    
                    log_ls_others += hmm_model.score(X, lengths)
                # calculating the average log_l for other other words (subtracting log_l of the current word)
                avg_log_l_others = ((log_ls_others)-log_l) / (len(self.hwords)-1)
                # calculating the dic_score
                dic_score = log_l - avg_log_l_others
                # storing values, and updating if current dic_score higher than previous best
                if dic_score > best_dic_score:
                    best_dic_score = dic_score
                    best_hmm_model = hmm_model
                    #best_n = n
            except:
                pass
        #return self.base_model(best_n) 
        return best_hmm_model
        #raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # Further ideas from
        # https://discussions.udacity.com/t/implement-selectorcv/247078/6
        # https://discussions.udacity.com/t/issue-with-selectorcv/299868/5
        # https://discussions.udacity.com/t/selectorcv-fails-with-indexerror-list-index-out-of-range/397820
        # https://discussions.udacity.com/t/implement-selectorcv/247078/23
        
        split_method = KFold(n_splits=3) # KFold(n_splits=min(3, len(self.sequences)))
        best_avg_log_l = float('-inf')
        previous_score = float('-inf')
        best_n = 3 # initializing 
        hmm_model = None
        for n in range(self.min_n_components, self.max_n_components + 1):   
            try:
                folds = 0
                total_log_l = 0
                # handling case len(self.lenghts) < 2 on its own, as suggested by forum mods       
                if len(self.sequences) < 2:
                    hmm_model = GaussianHMM(n_components=n, n_iter=1000).fit(self.X, self.lengths)
                    score = hmm_model.score(self.X, self.lengths)
                    if previous_score < score:
                        best_n =  n

                else: # end of case len(self.lenghts) == 1
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        folds += 1 # add to folds count
                        # Initializing train and test sets
                        x_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        x_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)                    
                        # Train the model with training set 
                        hmm_model = GaussianHMM(n_components=n, n_iter=1000).fit(x_train, lengths_train)                
                        # Test the model with test set
                        log_l = hmm_model.score(x_test, lengths_test)
                        # add to the total score for the folds
                        total_log_l += log_l
                    # calculate the average after the loop    
                    avg_log_l = total_log_l / folds
                    # comparison to current best    
                    if best_avg_log_l < avg_log_l:
                        best_avg_log_l = avg_log_l
                        #best_hmm_model = hmm_model
                        best_n = n
            except:
                pass
        return self.base_model(best_n) # return FULL model [not any of those trained with folded sets]
        #return GaussianHMM(n_components=best_n, n_iter=1000).fit(self.X, self.lengths)
        #raise NotImplementedError
