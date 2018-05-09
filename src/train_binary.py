#################################
#
# train_binary.py
#
#
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
#from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import MultinomialNB, GaussianNB
#from sklearn.neighbors import KNeighborsClassifier as KNN
from xgboost import XGBClassifier as XGB

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.preprocessing import StandardScaler
#from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import sys


def get_train_test(file_name):
    """Get standardized training and testing data sets from csv at given
    filepath.

    Parameters
    ----------
    filepath : str - path to find corpus.csv

    Returns
    -------
    X_train : ndarray - 2D
    X_test  : ndarray - 2D
    y_train : ndarray - 1D
    y_test  : ndarray - 1D
    """
# =============================================================================
#     raw_df = pd.read_csv(file_name)
#     corpus_df = make_clean_corpus_df(raw_df)
#     y = corpus_df.pop('corpus').values
#     X = corpus_df.values
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
# =============================================================================
    #importing the processed dataset
    #file_name="./data/corpus.csv"

    if len(sys.argv) > 1 and int(sys.argv[1]) > 0 :
        sample_size = int(sys.argv[1])
        print("Taking first", sample_size, "rows from dataset..")
        data = pd.read_csv(file_name).sample( n = sample_size)
    else:
        print("Reading whole dataset..")
        data = pd.read_csv(file_name)


    # split into train test
    print("Class counts:")
    print(pd.DataFrame(data['is_cardiologist'].value_counts()))
    y = data['is_cardiologist'].values
    X_train, X_test, y_train, y_test = train_test_split(data.corpus, y,test_size=0.25, random_state=0)

    return X_train, X_test, y_train, y_test


def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])


def profit_curve(cost_benefit, predicted_probs, labels):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and prediced probabilities of data points and thier true labels.

    Parameters
    ----------
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1

    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D
    """

    print("----------------------------------------")
    n_obs = float(len(labels))
    # Make sure that 1 is going to be one of our thresholds
    maybe_one = [] if 1 in predicted_probs else [1]
    thresholds = maybe_one + sorted(predicted_probs, reverse=True)
    profits = []
    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        confusion_matrix = standard_confusion_matrix(labels, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit) / n_obs
        profits.append(threshold_profit)
    return np.array(profits), np.array(thresholds)


def get_model_profits(model, cost_benefit, X_train, X_test, y_train, y_test):
    """Fits passed model on training data and calculates profit from cost-benefit
    matrix at each probability threshold.

    Parameters
    ----------
    model           : sklearn model - need to implement fit and predict
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    X_train         : ndarray - 2D
    X_test          : ndarray - 2D
    y_train         : ndarray - 1D
    y_test          : ndarray - 1D

    Returns
    -------
    model_profits : model, profits, thresholds
    """
    model.fit(X_train, y_train)
    predicted_probs = model.predict_proba(X_test)[:, 1]
    profits, thresholds = profit_curve(cost_benefit, predicted_probs, y_test)

    return profits, thresholds


def plot_model_profits(model_profits, save_path=None):
    """Plotting function to compare profit curves of different models.

    Parameters
    ----------
    model_profits : list((model, profits, thresholds))
    save_path     : str, file path to save the plot to. If provided plot will be
                         saved and not shown.
    """
    for model, profits, _ in model_profits:
        percentages = np.linspace(0, 100, profits.shape[0])
        plt.plot(percentages, profits, label=model.__class__.__name__)

    plt.title("Profit Curves")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='best')

    plt.savefig("profit_curves.png")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def find_best_threshold(model_profits):
    """Find model-threshold combo that yields highest profit.

    Parameters
    ----------
    model_profits : list((model, profits, thresholds))

    Returns
    -------
    max_model     : str
    max_threshold : float
    max_profit    : float
    """
    max_model = None
    max_threshold = None
    max_profit = None
    summary_list = []
    for model, profits, thresholds in model_profits:
        max_index = np.argmax(profits)
        #print("Model:", model)
        summary_list.append([model.__class__.__name__, profits[max_index], thresholds[max_index] ])
        if not max_model or profits[max_index] > max_profit:
            max_model = model
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit, summary_list






if __name__ == '__main__':
    print("***************** Training Model *****************")
    corpus_filepath = './data/corpus.csv'
    print("Input File:", corpus_filepath)

    """
    cost_benefit  : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    """
    cost_benefit = np.array([[100, -10], [0, 0]])


    X_train_raw, X_test_raw, y_train, y_test = get_train_test(corpus_filepath)


    print("Training & Test", X_train_raw.shape, X_test_raw.shape, y_train.shape, y_test.shape)

    # Bag of words model
    cv = TfidfVectorizer(max_features = 1000, stop_words= "english")
    print(cv)

    X_train = cv.fit_transform(X_train_raw).toarray()
    X_test = cv.transform(X_test_raw).toarray()


    #models = [RF(n_jobs=-1), LR(n_jobs=-1), GBC(), SVC(probability=True)]
    models = [MultinomialNB(), GaussianNB(), RF(n_jobs=-1), GBC(), XGB(n_jobs=-1)]


    model_profits = []
    for model in models:
        print(model.__class__.__name__)
        profits, thresholds = get_model_profits(model, cost_benefit,
                                                X_train, X_test,
                                                y_train, y_test)
        model_profits.append((model, profits, thresholds))

    plot_model_profits(model_profits, "./presentation/proft_curve.png")
    #plot_model_profits(model_profits)

    max_model, max_thresh, max_profit, summary_list = find_best_threshold(model_profits)
    max_labeled_positives = max_model.predict_proba(X_test) >= max_thresh
    proportion_positives = max_labeled_positives.mean()

    print("***************** Summary Report *****************")

    print(pd.DataFrame(summary_list, columns=['Classifier','Profit', 'Threshold' ]))

    print("--------------------------------------------------")
    reporting_string = ('Best model:\t\t{}\n'
                        'Best threshold:\t\t{:.2f}\n'
                        'Resulting profit:\t{}\n'
                        'Proportion positives:\t{:.2f}')

    print(reporting_string.format(max_model.__class__.__name__, max_thresh,
                                  max_profit, proportion_positives))
    print("--------------------------------------------------")

    model = max_model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    #Making the confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # classification report
    from sklearn.metrics import classification_report, accuracy_score
    print(classification_report(y_test, y_pred))
    print("Accuracy Score Train:", accuracy_score(y_train, model.predict(X_train), normalize = True))
    print("Accuracy Score Test:", accuracy_score(y_test, y_pred, normalize = True))

    # ROC curve
    # =============================================================================

    fpr, tpr, ths = roc_curve(y_test, y_pred_proba[:,1])
    # plt.plot(fpr,tpr)
    # plt.show()


    roc_auc = auc(fpr,tpr)
    print("Area under the curve : ", roc_auc)


    # Feature Importances
    print("***************** Feature Importances *****************")
    # Plot the feature importances of the forest
    fi = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4),
                                    model.feature_importances_ ),
                                    cv.get_feature_names()),
                                    reverse=True),
                            columns=['Importance', 'Feature'] )

    # Save top features to CSV
    fi.to_csv("./data/top_features_binary.csv")


    # plot chart top n features
    top_n = 15

    print(f"Top {top_n} features")
    print(fi.head(top_n))

    ax = fi.head(top_n).plot.bar(x='Feature', y='Importance')
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    #plt.show()


    # save model
    model_filename = "./model/binary_model.pkl"
    print(f"Saving Model to {model_filename}" )

    joblib.dump(max_model, model_filename)


    print("***************** End *****************")
