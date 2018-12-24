"""[summary]

:return: [description]
:rtype: [type]
"""
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import sklearn
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as accuracy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,\
precision_score, recall_score
import pylab as pl
import string
from sklearn.model_selection import KFold
import requests
import re
import bs4
from textraction_pipeline_OOP import case_param as cpar
from textraction_pipeline_OOP import wcf_params as wpar
import pickle
import pkg_resources

###### HELPER FUNCTIONS #################################################################################################################################################

def get_words(use_local = False, use_internal = True, url=wpar.words_url, 
              local_file = wpar.words_list_local):
    """
    Get a text with +80,000 words in Spanish
    
    :param use_local: use local text file, defaults to False
    :param use_local: bool
    :param use_internal: [description], defaults to True
    :param use_internal: bool, optional
    :param url: url to scrape text from, defaults to wpar.words_url
    :param url: str
    :param local_file: path for local text file, defaults to wpar.words_list_local
    :param local_file: str
    :return: sequence of Spanish words 
    :rtype: str
    """
    if not use_local:

        # READING TEXT FROM ONLINE RESOURCE
        r = requests.get(url, allow_redirects=True)
        soup = bs4.BeautifulSoup(r.text, 'html.parser')
        divs = soup.findAll("td", {"class": "blob-code blob-code-inner js-file-line"})
        wds = ' '.join(div.text for div in divs)

        # SAVING TO A LOCAL FILE FOR SUBSEQUENT USE
        op_file = open(local_file, "w", encoding = "utf-8")
        op_file.write(wds)
        op_file.close()

        return wds
    else:
        if use_internal:
            resource_package = __name__
            resource_path = "/".join((wpar.resource_folder, 
                                      wpar.words_list_internal))
            resource_string = pkg_resources.resource_string(resource_package, 
                                                            resource_path).decode("utf-8")            
            return resource_string
        else:
            file = open(local_file, "r", encoding = "utf-8")
            txt = file.read()
            file.close()

            return txt

def pre_process(x):
    """
    Clean a text
    
    :param x: text
    :type x: str
    :return: cleaned text
    :rtype: str
    """
    chars =  [x for x in list(string.punctuation)+['«', '»', '©', '■', '®',\
     '€', '°', '’']]
    x = str(x)
    for w in x:
        if w in chars:
            x = x.replace(w, "")
        if w in cpar.tildic:
            x = x.replace(w, cpar.tildic[w])
    return x

def build_ng(w, n):
    """
    Obtain the ngrams of a given word having added ^ and $ to signal beginning
                and end, respectively, of the word as an additional feature
                                Ex:n=3, 'tablet' => '^Os Osc sca car' )
    
    :param w: word
    :type w: [str
    :param n: ngram size to decompose word
    :type n: int 
    :return: ngrams of the word
    :rtype: str
    """
    if len(w) >=3:
        wd =  '^'+w+'$'
        nlst = [wd[i:i+n] for i in range(len(wd)-n+1)]
        return ' '.join(n for n in nlst)
    else:
        return None

def spa_names_data(filename, cols_to_drop):
    """
    Construct and pre-process Spanish names dataset
    
    :param filename: name of the file containing the names dataset
    :type filename: str
    :param cols_to_drop: columns to drop from name dataset
    :type cols_to_drop: list
    :return: processed dataset
    :rtype: pandas DataFrame
    """
    dfnm = pd.read_csv(filename) # Spanish names dataset
    dfnm =  dfnm.drop(axis=1,  columns=cols_to_drop)

    dfnm['Apellido Paterno'] = dfnm['Apellido Paterno'].apply(pre_process)
    dfnm['Apellido Materno'] = dfnm['Apellido Materno'].apply(pre_process)
    dfnm['Nombres'] = dfnm['Nombres'].apply(pre_process)
    return dfnm

def get_vocabulary(doc):
    """
    Obtain a set of the words in a text

    :param doc
    :type doc: str
    :return: set of words
    :rtype: set
    """
    words = set()
    for d in doc:
        if d:
            d = d.split()
            words = words.union(set(d))
    return words

def split(df,test_size=0.3):
    """
    Randomly split a dataset into train and test datasets
    
    :param df: dataset to be split
    :type df: pandas DataFrame
    :param test_size: fraction of the dataset to be left as test set,
                                                    defaults to 0.3
    :param test_size: float
    :return: training & testing feature dataset, training & testing label dataset
    :rtype: tup
    """
    X = make_word_cnt_feats(df['ngrams'])
    Y = df['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42,\
        test_size=test_size)
    return X_train, X_test, Y_train, Y_test

def make_word_cnt_feats(docs, vocabulary=None):
    """
    Construct a matrix of word counts in a collection of text documents
    
    :param docs: iterable(ie: list or pandas.Series) of strings
    :type docs: [type]
    :param vocabulary: set of words(strings) that appear in docs, defaults to None
    :param vocabulary: set
    :return: matrix of the word count features
    :rtype: numpy array 
    """
    if vocabulary is None:
        vocabulary = get_vocabulary(docs)
    else:
        vocabulary = vocabulary.union(get_vocabulary(docs))
    words = sorted(vocabulary)
    word_dict = {w: idx for idx, w in enumerate(words)}
    
    # feature matrix will be of size n x (size of vocabulary + 1)
    # add 1 to account for words that dont appear in the vocabulary
    vocab_size = len(words)
    word_cnt_features = np.zeros((len(docs), len(words) + 1))
    
    for idx in range(len(docs)):
        doc = str(docs.iloc[idx])
        if doc:
            words = doc.split()
            for w in words:
                word_idx = word_dict.get(w, vocab_size)
                word_cnt_features[idx, word_idx] += 1
            
    return word_cnt_features

###### TRAIN MODEL ################################################################

def find_best_NB(df, alphas, splits, test_size=0.3, N=3):
    """
    Find the best performer among NB mddels with different alpha parameters
    
    :param df: dataset
    :type df: pandas DataFrame
    :param alphas: List of different lambda values to test models
    :type alphas: list
    :param splits: Number of splits to make with the data to test models
    :type splits: int
    :param test_size: Fraction of the data to be left as test set (0, 1),
                                                         defaults to 0.3
    :param test_size: float
    :return: best-performing: alpha, roc-auc score, and results dataset
    :rtype: tup
    """
    best_alpha = None
    best_score = 0
    results = {} # store your cross validation results
    X_train, X_test, Y_train, Y_test = split(df,test_size=0.3)
    kf = KFold(n_splits=splits)
    kf.split(X_train)
    for (train_i, test_i) in kf.split(X_train):
        x_split_train, x_split_test = X_train[train_i], X_train[test_i]
        y_split_train, y_split_test = Y_train.iloc[train_i], Y_train.iloc[test_i]
        for a in alphas:
            nb = MultinomialNB(alpha=a)
            nb.fit(x_split_train, y_split_train)
            preds = nb.predict(x_split_test)
            results[a] = results.get(a, 0) + roc_auc_score(preds,\
             y_split_test) / splits
    for k, v in results.items():
        if v > best_score:
            best_alpha = k
            best_score = v
    return best_alpha, best_score, results

def fit_NBmodel(X, Y, alpha):
    """
    Fit a Multinomial Naive Bayes classifier to a labelled dataset
    
    :param X: feature dataset
    :type X: pandas DataFrame
    :param Y: label dataset
    :type Y: pandas DataFrame
    :param alpha: alpha parameter to be used for model
    :type alpha: int
    :return: trained Naive Bayes classifier
    :rtype: MultinomialNB
    """
    nb = MultinomialNB(alpha=alpha)
    nb.fit(X, Y)
    return nb

def test_models(df, alphas=[0.1, 1, 5, 10], splits = 3, t_size = 0.2, prnt=False):
    """
    Test various NB classifiers (varying alpha and randomly splitting dataset)
    and find the alpha corresponding to the one with best performance
    
    :param df: dataset
    :type df: pandas DataFrame
    :param alphas: list of alpha values to try, defaults to [0.1, 1, 5, 10]
    :param alphas: list
    :param splits:  Number of splits to make with the data to test models, default=3
    :param splits: int
    :param t_size: Fraction of the data to be left as test set (0, 1), default=0.2
    :param t_size: float
    :param prnt: print best model's performance metrics (accuracy, precision,
                                        F1 and ROC AUC scores, default=False)
    :param prnt: bool
    :return:  best-performing alpha, loaded CountVectorizer,
                         feature dataset and label dataset
    :rtype: tup
    """
    best_alpha = find_best_NB(df, alphas, splits, test_size=t_size, N=splits)[0]
    count_vectorizer = CountVectorizer()
    word_cnt_feats = count_vectorizer.fit_transform(df['ngrams'])
    X = word_cnt_feats.toarray()
    Y = (df['label'] == 1) # gen booleans
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=t_size,\
     random_state=42)
    nb = fit_NBmodel(X_train, Y_train, best_alpha)
    preds = nb.predict(X_test)
    if prnt:
        logg = 'MODEL PERFORMANCE for {}:'.format(df) +\
             '\n Accuracy: {:.3f}'.format(accuracy_score(preds, Y_test)) +\
             '\n Precision: {:.3f}'.format(precision_score(preds, Y_test)) +\
             '\n F1 Score: {:.3f}'.format(f1_score(preds, Y_test)) +\
             '\n ROC AUC Score: {:.3f}'.format(roc_auc_score(preds, Y_test))
    return best_alpha, count_vectorizer, X, Y
