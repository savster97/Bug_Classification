import pandas as pd
import numpy as np
import nltk
import pickle

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

from nltk import sent_tokenize
from nltk import pos_tag
from nltk import map_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

def grid_search(x_train, y_train, x_test, y_test, bug_types, parameters, pipeline):
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=5, n_jobs=3, verbose=10)
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


def tfidf_naivebayes_classifier(x_train, y_train, x_test, y_test, bug_types, stop_words):
    pipeline_nb = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
    ])
    parameters_nb = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__estimator__alpha': (1e-2, 1e-3)
    }
    grid_search(x_train, y_train, x_test, y_test, bug_types, parameters_nb, pipeline_nb)


def tfidf_logreg_classifier(x_train, y_train, x_test, y_test, bug_types, stop_words):
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
    grid_search(x_train, y_train, x_test, y_test, bug_types, parameters_logreg, pipeline_logreg)


def tfidf_svmlinear_classifier(x_train, y_train, x_test, y_test, bug_types, stop_words):
    pipeline_svmlin = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1))])
    parameters_svmlin = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None]}

    grid_search(x_train, y_train, x_test, y_test, bug_types, parameters_svmlin, pipeline_svmlin)


def tfidf_randomforest_classifier(x_train, y_train, x_test, y_test, bug_types, stop_words):
    pipeline_rf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(RandomForestClassifier()))])
    parameters_rf = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        "clf__estimator__max_depth": [100]}

    grid_search(x_train, y_train, x_test, y_test, bug_types, parameters_rf, pipeline_rf)


def tag_pos(x):
    sentences = sent_tokenize(x)
    sents = []
    for s in sentences:
        text = word_tokenize(s)
        pos_tagged = pos_tag(text)
        simplified_tags = [
            (word, map_tag('en-ptb', 'universal', tag)) for word, tag in pos_tagged]
        sents.append(simplified_tags)
    return sents


def post_tag_documents(df):
    x_data = []
    y_data = []
    total = len(df["Classification"])
    summaries = df["Summary"]
    print(len(summaries))
    types = df["Classification"]
    print(len(types))
    for i in range(1, len(summaries)):
        print(summaries[i])
        sents = tag_pos(summaries[i])
        print(sents)
        x_data.append(sents)
        y_data.append(types[i])
        print(types[i])
        i += 1
        if i % 5000 == 0:
            print(i, "/", total)

    # export Part-of-Speech tagging file
    with open('../pos_tagged_data.dat', 'wb') as f:
        pickle.dump((x_data, y_data), f)

    return x_data, y_data


def pos_tag_data(df):
    print("Converting data to Part-of-Speech tagging...")
    x_data, y_data = post_tag_documents(df)

    return x_data, y_data



# Load NLTK's English stop-words list
# Global Variables
STOP_WORDS = set(stopwords.words('english'))
print(STOP_WORDS)

# load pre-processed data
print("Loading already processed training data")
# Columns: ['Bug-ID ', 'Project ', 'Classification', 'Summary', 'Link']
data_df = pd.read_excel("../Bug_Report.xlsx").dropna()
# Number of classes
class_groups = data_df.groupby(["Classification"])
print(len(class_groups))
print(class_groups.groups.keys())
# Convert the 13 categories to the 9 reported in journal
data_df["Classification"] = data_df["Classification"].replace({"configurations property issue": "Configuration",
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
class_groups = data_df.groupby(["Classification"])
print(len(class_groups))
print(class_groups.groups.keys())
# all the list of bug classes to be used by the classification report
# bug_types = ["info release issue", "add issue"]
bug_types = ["Configuration", "Network", "Program Anomaly", "Database Issue", "GUI",
             "Permission/Depreciation", "Security", "Performance", "Test"]

# Extracting the data
data_x = np.array(data_df["Summary"].dropna())
data_y = np.array(data_df["Classification"].dropna())

# split the data, leave 1/3 out for testing
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=85)

## TF-IDF and Naive Bayes
print("TF-IDF + Naive Bayes")
tfidf_naivebayes_classifier(x_train, y_train, x_test, y_test, bug_types, STOP_WORDS)

## TF-IDF and Logistic Regression
print("TF-IDF + Logistic Regression")
tfidf_logreg_classifier(x_train, y_train, x_test, y_test, bug_types, STOP_WORDS)

## TF-IDF and SVM Linear
print("TF-IDF + SVM Linear")
tfidf_svmlinear_classifier(x_train, y_train, x_test, y_test, bug_types, STOP_WORDS)

## TF-IDF and Random Forest
print("TF-IDF + Random Forest")
tfidf_randomforest_classifier(x_train, y_train, x_test, y_test, bug_types, STOP_WORDS)

## Pre-processing data with Part-of-Speech tagging
x_data_postag, y_data_postag = pos_tag_data(data_df)

## Word2Vec and Naive Bayes
