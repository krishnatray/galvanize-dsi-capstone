#################################
#
# train_model_multiclass.py
#
#

#importing the libraries
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.naive_bayes import MultinomialNB as MNB
from xgboost import XGBClassifier as XGB

#from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.preprocessing import StandardScaler
#from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import sys

import warnings
from sklearn.externals import joblib
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

file_name="./data/corpus.csv"
model_dir ="./model"
data_dir ="./data"

############################
sample_size = 0
if len(sys.argv) > 1 and int(sys.argv[1]) > 0 :
        sample_size = int(sys.argv[1])
        print("Sample Size ", sample_size)

        data = pd.read_csv(file_name).sample( n = sample_size)
else:
        print("Reading Whole dataset..")
        data = pd.read_csv(file_name)

print("Data size:", data.shape)

# creating target column from speciality
data['target'] = data['specialty']


# keeping top_n catgories. combining rest categories as Other
top5 = data['target'].value_counts()[:5]
data['target'] = data.target.apply(mapper)

print("Class counts:")
print(pd.DataFrame(data['target'].value_counts()))


# Label Encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

X = data.corpus
y = le.fit_transform(data.target)

## Save label encoder to disk
le_file_name = f"{model_dir}/le_multiclass.pkl"
print(f"Saving label encoder to {le_file_name}")
joblib.dump(le, le_file_name)


# split into train test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=0)


# Bag of words model
print("*"*40)
#from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 1500, stop_words= "english")
print(cv)

X_train = cv.fit_transform(X_train_raw).toarray()
X_test = cv.transform(X_test_raw).toarray()

## Save cv to disk
file_name = f"{model_dir}/tfidf_multiclass.pkl"
print(f"Saving tfidf count vector to {file_name}")
joblib.dump(cv, file_name)


# Feature Scaling
# =============================================================================
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train_scaled = sc.fit_transform(X_train_cv)
# X_test_scaled = sc.transform(X_test_cv)
# =============================================================================

classifiers = { 'MultinomialNB' : MNB(),
               'RandomForest': RF(n_jobs=-1),
               'GradientBoosting': GBC(),
               'xgb': XGB()}

best_score = 0
for model_name, model in classifiers.items():
    # Fitting MultinomialNB
    print("="*60)
    y_score = model.fit(X_train, y_train)
    print(model)
    #predicting the test results
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    #Making the confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    #Confusion Matrix
    print(pd.DataFrame(cm))
    print(classification_report(y_test, y_pred))
    accuracy_train = accuracy_score(y_train, model.predict(X_train), normalize = True)
    accuracy_test = accuracy_score(y_test, y_pred, normalize = True)
    print(f"Accuracy Score Train: {round(accuracy_train,2)} Test:{round(accuracy_test,2)}" )

    model_score = f1_score(y_test, y_pred, average='micro')

    if model_score > best_score:
        best_score = model_score
        best_model = model
        best_model_name = model_name

#     file_name = f"{model_dir}/{model_name}_multiclass.pkl"
#     print(f"Saving GBC Model to {file_name}...")
#     joblib.dump(model, file_name)

print("=================== Selected Model ===================")
print("")
print(f"Best Model is: {best_model_name} F1 Score: {round(best_score,2)}")


y_score = best_model.fit(X_train, y_train)
print(best_model)
#predicting the test results
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

#Making the confusion Matrix
cm = confusion_matrix(y_test, y_pred)

#Confusion Matrix
print(pd.DataFrame(cm))
print(classification_report(y_test, y_pred))
accuracy_train = accuracy_score(y_train, best_model.predict(X_train), normalize = True)
accuracy_test = accuracy_score(y_test, y_pred, normalize = True)
print(" ")
print(f"Accuracy Score Train: {round(accuracy_train,2)} Test:{round(accuracy_test,2)}" )


file_name = f"{model_dir}/multiclass.pkl"
print(f"Saving {best_model_name} to {file_name}")
joblib.dump(model, file_name)


# Feature Importances
print("********* Feature Importances ************")
# Plot the feature importances of the forest
fi = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4),
                                model.feature_importances_ ),
                                cv.get_feature_names()),
                                reverse=True),
                        columns=['Importance', 'Feature'] )

# Save top features to CSV
fi.to_csv(f"{data_dir}/top_features_multiclass.csv")

# plot chart top 10 features
top_n = 15
print(f"Top {top_n} features")
print(fi.head(top_n))

ax = fi.head(top_n).plot.bar(x='Feature', y='Importance')
ax.set_xlabel("Features")
ax.set_ylabel("Importance")
#plt.show()

print("************ End *************")

