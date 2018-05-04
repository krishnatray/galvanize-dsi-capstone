#################################
#
# train_model_profit_curve.py
#
#
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix, roc_curve, auc
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


def get_train_test(filepath):
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
#     raw_df = pd.read_csv(filepath)
#     corpus_df = make_clean_corpus_df(raw_df)
#     y = corpus_df.pop('corpus').values
#     X = corpus_df.values
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
# =============================================================================
    #importing the processed dataset
    file_name="corpus.csv"
    data = pd.read_csv(file_name)

    if len(sys.argv) > 1 and int(sys.argv[1]) > 0 :
        sample_size = int(sys.argv[1])

        print("Taking first", sample_size, "rows from dataset..")

        # split into train test
        print("Class counts:")
        print(pd.DataFrame(data['is_cardiologist'][:sample_size].value_counts()))
        y = data['is_cardiologist'].values[:sample_size]
        X_train, X_test, y_train, y_test = train_test_split(data.corpus[:sample_size], y,test_size=0.25, random_state=0)
    else:
        # split into train test
        print("Class counts:")
        print(pd.DataFrame(data['is_cardiologist'].value_counts()))
        y = data['is_cardiologist'].values
        X_train, X_test, y_train, y_test = train_test_split(data.corpus, y,test_size=0.25, random_state=0)


    return X_train, X_test, y_train, y_test


# =============================================================================
# def make_clean_corpus_df(df):
#     """Cleans DataFrame loaded from corpus.csv.
#
#     Parameters
#     ----------
#     df : DataFrame
#
#     Returns
#     -------
#     DataFrame : Columns lowercased, spaces replaced with underscores,
#                 Text data replaced with binarized ints.
#     """
#     df.columns = [col.lower().replace(' ', '_') for col in df.columns]
#     df['corpus'] = (df['corpus'] == 'True.').astype(int)
#     yes_no_cols = ["int'l_plan", "vmail_plan"]
#     df[yes_no_cols] = (df[yes_no_cols] == "yes").astype(int)
#     return df.drop(['state', 'area_code', 'phone'], axis=1)
# =============================================================================


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
    print("Fitting Model...")
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
# =============================================================================
#     if save_path:
#         plt.savefig(save_path)
#     else:
#         plt.show()
#
# =============================================================================

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
    for model, profits, thresholds in model_profits:
        max_index = np.argmax(profits)
        print("Model:", model, "Max Profit:", profits[max_index])
        if not max_model or profits[max_index] > max_profit:
            max_model = model
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit


def profit_curve_main(filepath, cost_benefit):
    """Main function to test profit curve code.

    Parameters
    ----------
    filepath     : str - path to find corpus.csv
    cost_benefit  : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    """
    print("Calling get_train_test()...")
    X_train_raw, X_test_raw, y_train, y_test = get_train_test(filepath)
    print("Returned from get_train_test()", X_train_raw.shape, X_test_raw.shape, y_test.shape, y_train.shape)

    # Bag of words model
    print ("TFIDF vectorizer..." )
    cv = TfidfVectorizer(max_features = 100, stop_words= "english")
    print(cv)

    X_train = cv.fit_transform(X_train_raw).toarray()
    X_test = cv.transform(X_test_raw).toarray()


    #models = [RF(n_jobs=-1), LR(n_jobs=-1), GBC(), SVC(probability=True)]
    models = [MultinomialNB(), GaussianNB(), RF(n_jobs=-1), GBC(), KNN()]

    print("Models ", models)

    model_profits = []
    for model in models:
        print(model)
        profits, thresholds = get_model_profits(model, cost_benefit,
                                                X_train, X_test,
                                                y_train, y_test)
        model_profits.append((model, profits, thresholds))
    #plot_model_profits(model_profits, "proft_curve.png")
    plot_model_profits(model_profits)

    max_model, max_thresh, max_profit = find_best_threshold(model_profits)
    max_labeled_positives = max_model.predict_proba(X_test) >= max_thresh
    proportion_positives = max_labeled_positives.mean()
    reporting_string = ('Best model:\t\t{}\n'
                        'Best threshold:\t\t{:.2f}\n'
                        'Resulting profit:\t{}\n'
                        'Proportion positives:\t{:.2f}')
    print(reporting_string.format(max_model.__class__.__name__, max_thresh,
                                  max_profit, proportion_positives))

    model = max_model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    #Making the confusion Matrix
    from sklearn.metrics import confusion_matrix
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
    print("********* RF Feature Importances ************")
    # Plot the feature importances of the forest
    fi = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4),
                                    model.feature_importances_ ),
                                    cv.get_feature_names()),
                                    reverse=True),
                            columns=['Importance', 'Feature'] )

    # plot chart top 10 features
    top_n = 10
    ax = fi.head(top_n).plot.bar(x='Feature', y='Importance')
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    #plt.show()


    # save model
    save_model_filename = "model.joblib"
    print("Saving Model to", "model.joblib" )
    from sklearn.externals import joblib
    joblib.dump(max_model, save_model_filename)

    return max_model, max_thresh, max_profit




def div_count_pos_neg(X, y):
    """Helper function to divide X & y into positive and negative classes
    and counts up the number in each.

    Parameters
    ----------
    X : ndarray - 2D
    y : ndarray - 1D

    Returns
    -------
    negative_count : Int
    positive_count : Int
    X_positives    : ndarray - 2D
    X_negatives    : ndarray - 2D
    y_positives    : ndarray - 1D
    y_negatives    : ndarray - 1D
    """
    negatives, positives = y == 0, y == 1
    negative_count, positive_count = np.sum(negatives), np.sum(positives)
    X_positives, y_positives = X[positives], y[positives]
    X_negatives, y_negatives = X[negatives], y[negatives]
    return negative_count, positive_count, X_positives, \
           X_negatives, y_positives, y_negatives


def undersample(X, y, tp):
    """Randomly discards negative observations from X & y to achieve the
    target proportion of positive to negative observations.

    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - range [0.5, 1], target proportion of positive class observations

    Returns
    -------
    X_undersampled : ndarray - 2D
    y_undersampled : ndarray - 1D
    """
    if tp < np.mean(y):
        return X, y
    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    negative_sample_rate = (pos_count * (1 - tp)) / (neg_count * tp)
    negative_keepers = np.random.choice(a=[False, True], size=neg_count,
                                        p=[1 - negative_sample_rate,
                                           negative_sample_rate])
    X_negative_undersampled = X_neg[negative_keepers]
    y_negative_undersampled = y_neg[negative_keepers]
    X_undersampled = np.vstack((X_negative_undersampled, X_pos))
    y_undersampled = np.concatenate((y_negative_undersampled, y_pos))

    return X_undersampled, y_undersampled


def oversample(X, y, tp):
    """Randomly choose positive observations from X & y, with replacement
    to achieve the target proportion of positive to negative observations.

    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - range [0, 1], target proportion of positive class observations

    Returns
    -------
    X_undersampled : ndarray - 2D
    y_undersampled : ndarray - 1D
    """
    if tp < np.mean(y):
        return X, y
    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    positive_range = np.arange(pos_count)
    positive_size = (tp * neg_count) / (1 - tp)
    positive_idxs = np.random.choice(a=positive_range,
                                     size=int(positive_size),
                                     replace=True)
    X_positive_oversampled = X_pos[positive_idxs]
    y_positive_oversampled = y_pos[positive_idxs]
    X_oversampled = np.vstack((X_positive_oversampled, X_neg))
    y_oversampled = np.concatenate((y_positive_oversampled, y_neg))

    return X_oversampled, y_oversampled


def smote(X, y, tp, k=None):
    """Generates new observations from the positive (minority) class.
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf

    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - [0, 1], target proportion of positive class observations

    Returns
    -------
    X_smoted : ndarray - 2D
    y_smoted : ndarray - 1D
    """
    if tp < np.mean(y):
        return X, y
    if k is None:
        k = int(len(X) ** 0.5)

    neg_count, pos_count, X_pos, X_neg, y_pos, y_neg = div_count_pos_neg(X, y)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_pos, y_pos)
    neighbors = knn.kneighbors(return_distance=False)

    positive_size = (tp * neg_count) / (1 - tp)
    smote_num = int(positive_size - pos_count)

    rand_idxs = np.random.randint(0, pos_count, size=smote_num)
    rand_nghb_idxs = np.random.randint(0, k, size=smote_num)
    rand_pcts = np.random.random((smote_num, X.shape[1]))
    smotes = []
    for r_idx, r_nghb_idx, r_pct in zip(rand_idxs, rand_nghb_idxs, rand_pcts):
        rand_pos, rand_pos_neighbors = X_pos[r_idx], neighbors[r_idx]
        rand_pos_neighbor = X_pos[rand_pos_neighbors[r_nghb_idx]]
        rand_dir = rand_pos_neighbor - rand_pos
        rand_change = rand_dir * r_pct
        smoted_point = rand_pos + rand_change
        smotes.append(smoted_point)

    X_smoted = np.vstack((X, np.array(smotes)))
    y_smoted = np.concatenate((y, np.ones((smote_num,))))
    return X_smoted, y_smoted


def sampling_main(model, filepath, cost_benefit, range_params=(0.35, 0.65, 0.05)):
    """Helper function to test smoting effectiveness.

    Parameters
    ----------
    model        : sklearn model - needs to implement fit and predict
    filepath     : str - path to find corpus.csv
    cost_benefit : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    range_params : tuple - floats to pass to np.arange to
    """
    def print_ratio_profit(ratio, profit):
        """Helper function to pretty print the sampleing ratio
        and profit from it.
        Parameters
        ----------
        ratio  : float - [0, 1]
        profit : float
        """
        print('\t{:.2f} ->\t{:.5f}'.format(ratio, profit))


    X_train_raw, X_test_raw, y_train, y_test = get_train_test(filepath)

    # Bag of words model
    cv = TfidfVectorizer(max_features = 1000, stop_words= "english")
    print(cv)

    X_train = cv.fit_transform(X_train_raw).toarray()
    X_test = cv.transform(X_test_raw).toarray()



    model.fit(X_train, y_train)
    print(model)
    y_predict = model.predict(X_test)
    confusion_mat = standard_confusion_matrix(y_test, y_predict)
    profit = np.sum(confusion_mat * cost_benefit) / len(y_test)
    original_ratio = np.mean(y_train)

    print('Profit from original ratio:')
    print_ratio_profit(original_ratio, profit)
    for sampling_techinque, name in zip([undersample, oversample, smote],
                                ['undersampling', 'oversampling', 'smoting']):
        print('Profit when {} to ratio of:'.format(name))
        for ratio in np.arange(*range_params):
            X_sampled, y_sampled = sampling_techinque(X_train, y_train, ratio)
            model.fit(X_sampled, y_sampled)
            y_predict = model.predict(X_test)
            confusion_mat = standard_confusion_matrix(y_test, y_predict)
            profit = np.sum(confusion_mat * cost_benefit) / float(len(y_test))
            print_ratio_profit(ratio, profit)


if __name__ == '__main__':
    print("*********** Training Model ************")
    corpus_filepath = './corpus.csv'
    print("Input File:", corpus_filepath)

    cost_benefit = np.array([[100, -10], [0, 0]])
    max_model, max_thresh, max_profit = profit_curve_main(corpus_filepath, cost_benefit)
    sampling_main(max_model, corpus_filepath, cost_benefit)

    print("*********** End ************")
