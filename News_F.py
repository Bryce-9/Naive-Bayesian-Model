#-*- encoding: utf-8 -*-
import pickle
#import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
#import  warnings
#from gensim.models import Word2Vec
#from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import GridSearchCV
#import pandas as pd
#import jieba



with open('tfidf_feature.model','rb') as file:
    tfidf_feature = pickle.load(file)
    X = tfidf_feature['featureMatrix']
    y = tfidf_feature['label']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)


clf=MultinomialNB(alpha=1.0e-10).fit(train_X,train_y)
doc_class_predicted=clf.predict(test_X)
print(clf.score(test_X, test_y))
print(metrics.classification_report(test_y,doc_class_predicted))


clf1=BernoulliNB().fit(train_X,train_y)
doc_class_predicted1=clf1.predict_proba(test_X)
print(clf1.score(test_X, test_y))
print(metrics.classification_report(test_y,doc_class_predicted1))


