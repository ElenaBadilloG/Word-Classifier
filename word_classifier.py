"""
[summary]

:raises ValueError: [description]
:return: [description]
:rtype: [type]
"""
import sys
import time
import numpy as np
import pandas as pd
import csv
import sklearn
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,\
precision_score, recall_score
import string
from sklearn.model_selection import KFold
import requests
import re
#import bs4
#import pickle
from textraction_pipeline_OOP import case_param as cpar
from textraction_pipeline_OOP import wcf_params as wpar
from textraction_pipeline_OOP import utils_names_pred as utils
import pkg_resources

###### HELPER FUNCTIONS #############################################################

class WordClassifier():
    '''
    Instantiate a WordClassifier with two n-gram size values (int)
    and default attributes
    '''
    def __init__(self, N1, N2):

        #self.id = next(self.counter)
        if any(n < 2 or type(n) != int for n in [N1, N2]):
            raise ValueError('N must be set as an integer equal or higher than 2')

        self._N_name = N1
        self._N_gen= N2
        self._names = ''
        self._words_df = None
        self._name_featset = None
        self._gen_featset = None
        self._name_df = None
        self._gen_df = None
        self._name_model = None
        self._name_vect = None
        self._gen_model = None
        self._gen_vect = None


    def set_words_df(self, filename):
        """
        Construct a Spanish names dataset from a name text
        file
        
        :param filename
        :type filename: str
        :return: Spanish names dataset
        :rtype: pandas DataFrame
        """
        resource_package = __name__
        resource_path = "/".join((wpar.resource_folder, filename))
        resource_stream = pkg_resources.resource_stream(resource_package,
                                                        resource_path)
        dfnm = pd.read_csv(resource_stream) # Spanish names dataset
        self._words_df = dfnm.applymap(utils.pre_process)
        return self._words_df

    def build_names_featset(self, txt, deflt=0):
        """
        Decompose words into its n-grams and build a dataset with 
        the ngrams as features and their appropiate label (0= non-name, 1 = name)

        Methodology ref: http://theory.stanford.edu/~dfreeman/papers/namespam.pdf
        
        :param txt
        :type txt: str
        :param deflt: label value, defaults to 0
        :param deflt: int
        :return: dataset
        :rtype: pandas DataFrame
        """
        txt = utils.pre_process(txt).lower()
        featset = []
        for i, w in enumerate(txt.split()):
            lab = deflt
            if len(w) > 2 and w not in cpar.user_def_stop_words \
                                and utils.build_ng(w, self._N_name):
                ng = utils.build_ng(w, self._N_name).lower()
                featset.append(('^'+w+'$'.lower(), ng, lab))
        self._name_featset = pd.DataFrame(featset, columns=['word', 'ngrams',\
                                                                    'label'])
        return self._name_featset

    def build_word_train_df(self, use_local, filename):
        """
        Build a balanced training set of name vs. non-name Spanish words
        
        :param use_local: use or not local file
        :type use_local: bool
        :param filename: name of local file
        :type filename: str
        :return: trained dataset
        :rtype: pandas DataFrame
        """
        wds = utils.get_words(use_local)
        dfnm = self.set_words_df(filename)
        names = list(dfnm['Nombres'].unique())
        namset = []

        for n in names:
            namset.append(n.split()[0])
            if len(n.split()) == 2:
                namset.append(n.split()[1])
        apell = list(dfnm['Apellido Paterno'].unique()) + \
                     list(dfnm['Apellido Materno'].unique())
        allnames = ' '.join(n for n in namset + apell if type(n)==str)

        df = self.build_names_featset(wds)
        dfnames = self.build_names_featset(allnames, 1)
        df = df.sample(frac=0.05, random_state=42) # just 5% of +80,000, for balanced training set
        frames = [df.copy(), dfnames.copy()]
        df3 = pd.concat(frames)
        self._name_df = df3.sample(frac=1, random_state=42)
        return self._name_df

    def build_gen_train_df(self, filename):
        """
        Build a balanced training set of female vs. male Spanish names
        
        :param filename: name of local file
        :type filename: str
        :return: trained dataset
        :rtype: pandas DataFrame
        """
        resource_package = __name__
        resource_path = "/".join((wpar.resource_folder, filename))
        resource_stream = pkg_resources.resource_stream(resource_package,
                                                       resource_path)

        df = pd.read_csv(resource_stream) # Spanish names dataset
        df['Nombres'] = df['Nombres'].apply(utils.pre_process)
        df['ngrams'] = df['Nombres'].apply(utils.build_ng, args=(self._N_gen,))
        df['label'] = df['Género'].apply(lambda x: 1 if x=='M' else 0)
        self._gen_df = df
        return self._gen_df

    ###### TRAIN MODEL #########################################################

    def train_model(self, df, alphas, splits, t_size, prnt):
        alpha, vectorizer, X, Y = utils.test_models(df, alphas, splits, t_size, prnt)
        """
        Fit the best Naive Bayes model along varying alpha values to a training set
        
        :param df: Training set
        :type df: pandas DataFrame
        :param alphas: List of different lambda values to test model
        :type alphas: list
        :param splits: Number of splits to make with the data to test models
        :type splits: int
        :param t_size: fraction of the data to be left as test set (0, 1)
        :type t_size: float 
        :param prnt: print best model's performance metrics (accuracy, precision, F1 and ROC AUC scores)
        :type prnt: bool
        :return: trained NaiveBayesClassifier, loaded CountVectorizer
        :rtype: tup
        """
        # TEST DIFFERENT ALPHAS AND CHOOSE THE BEST
        nb = utils.fit_NBmodel(X, Y, alpha)

        return nb, vectorizer

    def get_name_model(self, alphas, splits, t_size, prnt):
        """
        Get the NB model for name vs. non-name word classifier
        
        :param alphas: List of different lambda values to test model
        :type alphas: list
        :param splits: Number of splits to make with the data to test models
        :type splits: int
        :param t_size: fraction of the data to be left as test set (0, 1)
        :type t_size: float 
        :param prnt: print best model's performance metrics (accuracy, precision, F1 and ROC AUC scores)
        :type prnt: bool
        :return: trained NaiveBayesClassifier, loaded Vectorizer object
        :rtype: tup
        """
        self._name_model, self._name_vect = self.train_model(self._name_df,
                    alphas, splits, t_size, prnt)
        return self._name_model, self._name_vect

    def get_gen_model(self,  alphas, splits, t_size, prnt):
        """
        Get the NB model for male vs. female name classifier
        
        :param alphas: List of different lambda values to test model
        :type alphas: list
        :param splits: Number of splits to make with the data to test models
        :type splits: int
        :param t_size: fraction of the data to be left as test set (0, 1)
        :type t_size: float 
        :param prnt: print best model's performance metrics (accuracy, precision, F1 and ROC AUC scores)
        :type prnt: bool
        :return: Trained NaiveBayesClassifier, loaded Vectorizer object
        :rtype: tup
        """
        self._gen_model, self._gen_vect = self.train_model(self._gen_df,
                 alphas, splits, t_size, prnt)
        return self._gen_model, self._gen_vect

    def load_classifier(self):
        """
        Compute and train the name and gender NB classifiers
        """
        self._name_df = self.build_word_train_df(filename = wpar.name_csv,
             use_local = True)
        self._gen_df = self.build_gen_train_df(filename=wpar.gen_csv)

        self._name_model, self._name_vect = self.get_name_model(wpar.alphas,
                    splits = 3, t_size = 0.2, prnt=True)
        
        self._gen_model, self._gen_vect = self.get_gen_model(wpar.alphas,
                    splits = 3, t_size = 0.2, prnt=True)
       

###### CLASSIFY NEW WORD #########################################################

    def is_person(self, new_word, thresh=0.7):
        """
        Method to classify a given word as name or non-name
        
        :param new_word: Word to be classified
        :type new_word: str
        :param thresh: Probability threshold to be cpnsidered a name, defaults to 0.78
        :param thresh: float
        :return: label (True=name)
        :rtype: bool
        """
        try:
            dfnew = self.build_names_featset(new_word, self._N_name)
            word_cnt_test =  self._name_vect.transform(dfnew['ngrams'])
            Xtst = word_cnt_test.toarray()
            #print(self._name_model.predict_proba(Xtst))
            if self._name_model.predict_proba(Xtst)[0][1] >= thresh:
                return True
            else:
                return False
        except ValueError as e:
            pass

    def gender(self, new_word, thresh=0.5):
        """
        Method to classify a given name as male or female
        
        :param new_word: Name to be classified
        :type new_word: str
        :param thresh: Probability threshold to be considered a male name, defaults to 0.5
        :param thresh: float, optional
        :return: label ('M' = male, 'F' = female)
        :rtype: str
        """
        if len(new_word) >= 3:
            try:
                new_ng = utils.build_ng(new_word, self._N_gen)
                new_row = [('^'+new_word+'$'.lower(), new_ng)]
                dfnew = pd.DataFrame(new_row, columns=['word', 'ngrams'])
                word_cnt_test =  self._gen_vect.transform(dfnew['ngrams'])
                Xtst = word_cnt_test.toarray()
                if self._gen_model.predict_proba(Xtst)[0][1] >= thresh:
                    return 'M'
                else:
                    return 'F'
            except ValueError as e:
                print(e)
            pass