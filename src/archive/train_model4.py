# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:58:33 2018

@author: SS
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
#matplotlib inline


# ------------ data pipeline -----------------

# =============================================================================
# def clean_data(text_field):
#     text_field = re.sub('[^a-zA-Z0-9]', ' ',text_field.lower())
#     return text_field
#
# #importing the dataset
# file_name="labelled.csv"
# data = pd.read_csv(file_name)
#
# data['is_cardiologist'] = data.specialty.str.contains("Card")
#
# data['corpus'] = data.procedure.apply(clean_data)
#
#
# data.to_csv("corpus.csv")
#
# =============================================================================

# importing the processed dataset
file_name="corpus.csv"
data = pd.read_csv(file_name)


# split into train test
from sklearn.model_selection import train_test_split
y = data['is_cardiologist'].values
X_train, X_test, y_train, y_test = train_test_split(data.corpus, y,test_size=0.25, random_state=0)


# Bag of words model
print ("TFIDF vectorizer..." )

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(ngram_range=(1, 3), stop_words= "english")
print(cv)

X_train_cv = cv.fit_transform(X_train).toarray()
X_test_cv = cv.transform(X_test).toarray()


# Feature Scaling
# =============================================================================
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train_scaled = sc.fit_transform(X_train_cv)
# X_test_scaled = sc.transform(X_test_cv)
# =============================================================================

# Fitting Naive Bayes import GaussianNB
print("MultinomialNB...")
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_cv, y_train)

#predicting the test results
y_pred = model.predict(X_test_cv)
y_pred_proba = model.predict_proba(X_test_cv)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# classification report
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
print("Accuracy Score Train:", accuracy_score(y_train, model.predict(X_train_cv), normalize = True))
print("Accuracy Score Test:", accuracy_score(y_test, y_pred, normalize = True))

# ROC curve
# =============================================================================
from sklearn.metrics import roc_curve
fpr, tpr, ths = roc_curve(y_test, y_pred_proba[:,1])
# plt.plot(fpr,tpr)
# plt.show()

from sklearn.metrics import auc
roc_auc = auc(fpr,tpr)
print("Area under the curve : ", roc_auc)

# save model
save_model_filename = "model.joblib"
print("Saving Model to", "model.joblib" )
from sklearn.externals import joblib
joblib.dump(model, save_model_filename)

print("==== Done ====")

# load model
# from sklearn.externals import joblib
#pulled_model = joblib.load("model.joblib")
#y_pulled_pred = pulled_model.predict(X_test)
#y_pulled_pred_proba = pulled_model.predict_proba(X_test)


