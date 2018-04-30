import pickle
from pickles_algos import sc

randForestClassifier = pickle.load(open("./pickles/random_forest.pickle", 'rb'))
naiveBayesClassifier = pickle.load(open("./pickles/naive_bayes.pickle", 'rb'))
knnClassifier = pickle.load(open("./pickles/knn.pickle", 'rb'))
decisionTreeClassifier = pickle.load(open("./pickles/decision_tree.pickle", 'rb'))


# Random Forest Classification
def randforest(test_input):

    X_test = sc.transform([test_input])
    y_pred = randForestClassifier.predict(X_test)

    if y_pred[0] == 0:
        y_pred = False
    else:
        y_pred = True

    return y_pred


# Naive Bayes
def naivebayes(test_input):

    X_test = sc.transform([test_input])
    y_pred = naiveBayesClassifier.predict(X_test)
    if y_pred[0] == 0:
        y_pred = False
    else:
        y_pred = True

    return y_pred


def k_nearest_neighbours(test_input):

    X_test = sc.transform([test_input])

    # Predicting the Test set results
    y_pred = knnClassifier.predict(X_test)
    if y_pred[0] == 0:
        y_pred = False
    else:
        y_pred = True

    return y_pred


def decision_tree(test_input):

    X_test = sc.transform([test_input])
    # Predicting the Test set results
    y_pred = decisionTreeClassifier.predict(X_test)
    if y_pred[0] == 0:
        y_pred = False
    else:
        y_pred = True

    return y_pred
