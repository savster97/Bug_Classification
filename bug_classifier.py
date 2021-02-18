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


def grid_search(x_train, y_train, x_test, y_test, bug_types, parameters, pipeline):
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
    grid_search_tune.fit(x_train, y_train)

    print()
    print("Best parameters set:")
    print(grid_search_tune.best_estimator_.steps)
    print()

    # measuring performance on test set
    print("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(x_test)

    print(classification_report(y_test, predictions, target_names=bug_types))

# Load NLTK's English stop-words list
# Global Variables
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))
print(STOP_WORDS)

# load pre-processed data
print("Loading already processed training data")
# Columns: ['Bug-ID ', 'Project ', 'Classification', 'Summary', 'Link']
data_df = pd.read_excel("../Bug_Report.xlsx")
# all the list of bug classes to be used by the classification report
bug_types = ["configurations property issue", "connection issue", "server issue",
            "syntax issue", "error/syntax issue", "db issue", "info release issue",
            "interface issue", "add issue", "permission issue/deprecate",
             "security issue", "computation issue", "memory/perfomance issue"]

data_x = np.array(data_df["Summary"].dropna())
data_y = np.array(data_df["Classification"].dropna())

## TF-IDF and Naive Bayes
print("TF-IDF + Naive Bayes")
# split the data, leave 1/3 out for testing
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=1/3, random_state=85)
# MultinomialNB: Multi-Class OneVsRestClassifier
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=STOP_WORDS)),
    ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
])
parameters_nb = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__estimator__alpha': (1e-2, 1e-3)
}
grid_search(x_train, y_train, x_test, y_test, bug_types, parameters_nb, pipeline_nb)

## TF-IDF and Logistic Regression
print("TF-IDF + Logistic Regression")
pipeline_logreg = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=STOP_WORDS)),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
])
parameters_logreg = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}
grid_search(x_train, y_train, x_test, y_test, bug_types, parameters_logreg, pipeline_logreg)

# TF-IDF and SVM Linear
print("TF-IDF + SVM Linear")

pipeline_svmlin = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=STOP_WORDS)),
    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1))])
parameters_svmlin = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None]}

grid_search(x_train, y_train, x_test, y_test, bug_types, parameters_svmlin, pipeline_svmlin)

# TF-IDF and Random Forest
print("TF-IDF + Random Forest")

pipeline_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=STOP_WORDS)),
    ('clf', OneVsRestClassifier(RandomForestClassifier()))])
parameters_rf = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__max_depth": [100]}

grid_search(x_train, y_train, x_test, y_test, bug_types, parameters_rf, pipeline_rf)


