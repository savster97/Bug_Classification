import pandas as pd
import numpy as np
import nltk

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from nltk.corpus import stopwords


def grid_search(df, x_train, y_train, x_predict, parameters, pipeline, fileName):
    results_df = df

    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
    # Bug Report data here for training
    grid_search_tune.fit(x_train, y_train)

    print()
    print("Best parameters set:")
    print(grid_search_tune.best_estimator_.steps)
    print()

    # Predicting classification of Jira Issues
    print("Predicting Jira Issues Bug Types")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(x_predict)
    results_df['bugType'] = predictions

    print("Exporting results of " + fileName)
    results_df.to_csv(fileName)

def tfidf_naivebayes_classifier(df, x_train, y_train, x_predict, stop_words):
    fileName = "../Classified_Jira_Issues/TF-IDF_NaiveBayes_Classifier.csv"
    pipeline_nb = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
    ])
    parameters_nb = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__estimator__alpha': (1e-2, 1e-3)
    }
    grid_search(df, x_train, y_train, x_predict, parameters_nb, pipeline_nb, fileName)


def tfidf_logreg_classifier(df, x_train, y_train, x_predict, stop_words):
    fileName = "../Classified_Jira_Issues/TF-IDF_LogReg_Classifier.csv"
    pipeline_logreg = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
    ])
    parameters_logreg = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }
    grid_search(df, x_train, y_train, x_predict, parameters_logreg, pipeline_logreg, fileName)


def tfidf_svmlinear_classifier(df, x_train, y_train, x_predict, stop_words):
    fileName = "../Classified_Jira_Issues/TF-IDF_SVMLinear_Classifier.csv"
    pipeline_svmlin = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1))])
    parameters_svmlin = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None]}

    grid_search(df, x_train, y_train, x_predict, parameters_svmlin, pipeline_svmlin, fileName)


def tfidf_randomforest_classifier(df, x_train, y_train, x_predict, stop_words):
    fileName = "../Classified_Jira_Issues/TF-IDF_RF_Classifier.csv"
    pipeline_rf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(RandomForestClassifier()))])
    parameters_rf = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        "clf__estimator__max_depth": [100]}

    grid_search(df, x_train, y_train, x_predict, parameters_rf, pipeline_rf, fileName)


## Load NLTK's English stop-words list
# Global Variables
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))
print(STOP_WORDS)

# all the list of bug classes to be used by the classification report
# bug_types = ["info release issue", "add issue"]
BUG_TYPES = ["Configuration", "Network", "Program Anomaly", "Database Issue", "GUI",
             "Permission/Depreciation", "Security", "Performance", "Test"]

## Load training data
# Columns: ['Bug-ID ', 'Project ', 'Classification', 'Summary', 'Link']
print("Loading training data")
train_data_df = pd.read_excel("../Bug_Report.xlsx")
# Convert the 13 categories to the 9 reported in journal
train_data_df["Classification"] = train_data_df["Classification"].replace({"configurations property issue": "Configuration",
                                                               "connection issue ": "Network",
                                                               "connection issue": "Network",
                                                               "server issue": "Network",
                                                               "syntax issue": "Program Anomaly",
                                                               "error/syntax issue": "Program Anomaly",
                                                               "computation issue": "Program Anomaly",
                                                               "info release issue": "Program Anomaly",
                                                               "add issue": "Program Anomaly",
                                                               "add isuue": 'Program Anomaly',
                                                               "db issue": "Database Issue",
                                                               "interface issue": "GUI",
                                                               "permission issue/deprecate": "Permission/Depreciation",
                                                               "security issue": "Security",
                                                               "memory issue": "Performance",
                                                               "performance issue": "Performance",
                                                               "test issue": "Test"})

# Extracting the training data
train_data_x = np.array(train_data_df["Summary"].dropna())
train_data_y = np.array(train_data_df["Classification"].dropna())

## Load data for prediction
print("Loading Jira Issues")
# Columns: ['projectID', 'creationDate', 'type', 'priority', 'description', 'summary']
predict_data_df = pd.read_csv("../BugfromJira.csv")
# Extracting the new data for classification
predict_data_x = np.array(predict_data_df["summary"])

## TF-IDF and Naive Bayes
print("TF-IDF + Naive Bayes")
tfidf_naivebayes_classifier(predict_data_df, train_data_x, train_data_y, predict_data_x, STOP_WORDS)

## TF-IDF and Logistic Regression
print("TF-IDF + Logistic Regression")
tfidf_logreg_classifier(predict_data_df, train_data_x, train_data_y, predict_data_x, STOP_WORDS)

## TF-IDF and SVM Linear
print("TF-IDF + SVM Linear")
tfidf_svmlinear_classifier(predict_data_df, train_data_x, train_data_y, predict_data_x, STOP_WORDS)

## TF-IDF and Random Forest
print("TF-IDF + Random Forest")
tfidf_randomforest_classifier(predict_data_df, train_data_x, train_data_y, predict_data_x, STOP_WORDS)
