import pandas as pd
import numpy as np
import spacy
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

nlp = spacy.load('en_core_web_sm')
stop_words = stopwords.words("english")

# create class for custom transformer
class CharCounter(BaseEstimator, TransformerMixin):

    # takes in a 2d array X for the feature data and a 1d array y for the target labels
    def fit(self, X, y=None):
        return self
    # transform method also takes a 2d array X
    def transform(self, X):

        n_char = X.str.len()

        X_values = n_char

        return pd.DataFrame(np.array(X_values))

# create class for custom transformer
class CaseCounter(BaseEstimator, TransformerMixin):

    # takes in a 2d array X for the feature data and a 1d array y for the target labels
    def fit(self, X, y=None):
        return self
    # transform method also takes a 2d array X
    def transform(self, X):

        n_title_case = pd.Series(X).apply(lambda x: sum(1 for c in x if c.isupper())).values
        n_char = X.str.len()

        X_values = n_title_case / n_char

        return pd.DataFrame(np.array(X_values))

# create class for custom transformer
class StopWordCounter(BaseEstimator, TransformerMixin):

    # takes in a 2d array X for the feature data and a 1d array y for the target labels
    def fit(self, X, y=None):
        return self

    # transform method also takes a 2d array X
    def transform(self, X):

        '''
        INPUT: string
        OUPUT: number of stopwords found (int)
        '''

        X_values = []

        for text in X:
            stop_words_found = []

            for i in text.split():
                if i in stop_words:
                    stop_words_found.append(1)

            X_values.append(sum(stop_words_found))

        return pd.DataFrame(np.array(X_values))

# create class for custom transformer
class WordPronCounter(BaseEstimator, TransformerMixin):

    # takes in a 2d array X for the feature data and a 1d array y for the target labels
    def fit(self, X, y=None):
        return self

    # transform method also takes a 2d array X
    def transform(self, X):

        X_values = []

        for text in X:
            pronouns = []

            for token in nlp(text):
                if token.pos_ == 'PRON':
                    pronouns.append(token)

            X_values.append(len(pronouns))

        return pd.DataFrame(np.array(X_values))

# create class for custom transformer
class WordNounCounter(BaseEstimator, TransformerMixin):

    # takes in a 2d array X for the feature data and a 1d array y for the target labels
    def fit(self, X, y=None):
        return self

    # transform method also takes a 2d array X
    def transform(self, X):

        X_values = []

        for text in X:
            pronouns = []

            for token in nlp(text):
                if token.pos_ == 'NOUN':
                    pronouns.append(token)

            X_values.append(len(pronouns))

        return pd.DataFrame(np.array(X_values))

# create class for custom transformer
class WordAdjCounter(BaseEstimator, TransformerMixin):

    # takes in a 2d array X for the feature data and a 1d array y for the target labels
    def fit(self, X, y=None):
        return self

    # transform method also takes a 2d array X
    def transform(self, X):

        X_values = []

        for text in X:
            pronouns = []

            for token in nlp(text):
                if token.pos_ == 'ADJ':
                    pronouns.append(token)

            X_values.append(len(pronouns))

        return pd.DataFrame(np.array(X_values))
