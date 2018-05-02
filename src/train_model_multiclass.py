#################################
#
# train_model_multiclass.py
#
#

#importing the libraries
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
#%matplotlib inline


def mapper(category):
    """
    input:
    category string
    top5  pandas series

    output:
    returns 'Other' if category is not in the top5

    """
    return category if category in top5 else 'Other'


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

############################3
sample_size = 0
if len(sys.argv) > 1 and int(sys.argv[1]) > 0 :
        sample_size = int(sys.argv[1])
        print("Sample Size ", sample_size)

        data = pd.read_csv(file_name)[:sample_size]
else:
        print("Reading Whole dataset..")
        data = pd.read_csv(file_name)
        print("Data size ", data.shape)





# creating target column from speciality
data['target'] = data['specialty']

# keeping top5 catgories. combining rest categories as Other
top5 = data['target'].value_counts()[:5]
data['target'] = data.target.apply(mapper)


# Label Encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

X = data.corpus
y = le.fit_transform(data.target)

# split into train test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=0)


# Bag of words model
print("*"*40)
print ("TFIDF vectorizer..." )
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 100, stop_words= "english")
print(cv)

X_train = cv.fit_transform(X_train_raw).toarray()
X_test = cv.transform(X_test_raw).toarray()



# Feature Scaling
# =============================================================================
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train_scaled = sc.fit_transform(X_train_cv)
# X_test_scaled = sc.transform(X_test_cv)
# =============================================================================

# Fitting MultinomialNB
print("*"*40)
print("MultinomialNB...")
#from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

print(model)

#predicting the test results
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

#Making the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
#Confusion Matrix
print(pd.DataFrame(cm))
print(classification_report(y_test, y_pred))
print("Accuracy Score Train:", accuracy_score(y_train, model.predict(X_train), normalize = True))
print("Accuracy Score Test:", accuracy_score(y_test, y_pred, normalize = True))


# Fitting GradoientBoosting
print("*"*40)
print("GradientBoosting Classifier...")
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
model.fit(X_train, y_train)

print(model)

#predicting the test results
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

#Making the confusion Matrix
cm_gb = confusion_matrix(y_test, y_pred)
#Confusion Matrix
print(pd.DataFrame(cm_gb))
print(classification_report(y_test, y_pred))
print("Accuracy Score Train:", accuracy_score(y_train, model.predict(X_train), normalize = True))
print("Accuracy Score Test:", accuracy_score(y_test, y_pred, normalize = True))



# Feature Importances
# =============================================================================
# print("********* RF Feature Importances ************")
# # Plot the feature importances of the forest
# fi = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4),
#                                 model.feature_importances_ ),
#                                 cv.get_feature_names()),
#                                 reverse=True),
#                         columns=['Importance', 'Feature'] )
#
# # plot chart top 10 features
# top_n = 10
# ax = fi.head(top_n).plot.bar(x='Feature', y='Importance')
# ax.set_xlabel("Features")
# ax.set_ylabel("Importance")
# plt.show()
#
# =============================================================================

# save model
save_model_filename = "docreach_model_multiclass.pkl"
print(f"Saving Model to {save_model_filename}...")

from sklearn.externals import joblib
joblib.dump(model, save_model_filename)



print("************ End *************")

