import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


# todo add time
def train_logistic_regression(X, y, repeat=3):
    # transfrom to one-hot vector
    y = y.reshape(-1, 1)
    y = OneHotEncoder(categories='auto', sparse=False).fit_transform(y).astype(np.bool)

    # normalize x
    X = normalize(X, norm='l2')

    scores = []
    for _ in range(repeat):
        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

        # grid search with one-vs-rest classifiers
        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=1, cv=cv, verbose=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = prob_to_one_hot(y_pred)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        scores.append(test_acc)

    return scores
