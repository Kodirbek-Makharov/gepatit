# my diss
# from cmath import nan
# from pyexpat.errors import XML_ERROR_NOT_SUSPENDED
import numpy as np

# ======== RS asosida latent alomatlarni hosil qilish  ========
def LatentByStability(X, myu, stabilities, k=5, stability_border=0.5):  # k ta alomatlardan latent hosil qilinadi
    _st = np.argsort(stabilities)[::-1][: len(stabilities)]
    # newColCount = int(X.shape[1] / k + 1)

    getted = 0
    for q in stabilities:
        if q >= stability_border:
            getted = getted + 1

    if getted % k == 0:
        newColCount = int(getted / k)
    else:
        newColCount = int(getted / k + 1)
    # print(newColCount)

    q = -1
    newX = np.zeros(shape=(X.shape[0], newColCount))

    for q in range(newColCount):
        for j in range(k * q, k * q + k):
            if j == X.shape[1]:
                break
            # if stabilities[j] == 0.5:
            # print(stabilities[_st[j]])
            if stabilities[_st[j]] < stability_border:
                continue
            for i in range(X.shape[0]):
                if np.isnan(X[i, _st[j]]) == False:
                    newX[i, q] = newX[i, q] + myu[int(X[i, _st[j]]) - 1, _st[j]]
        # print(q, cr1.WesMiqdoriyFeature(newX[:, q], y))
    return newX


def LatentByWes(X, w, myu, k=5):  # k ta alomatlardan latent hosil qilinadi
    _st = np.argsort(w)[::-1][: len(w)]
    newColCount = int(X.shape[1] / k + 1)
    q = -1
    newX = np.zeros(shape=(X.shape[0], newColCount))

    for q in range(newColCount):
        for j in range(k * q, k * q + k):
            if j == X.shape[1]:
                break
            for i in range(X.shape[0]):
                if np.isnan(X[i, _st[j]]) == False:
                    newX[i, q] = newX[i, q] + myu[int(X[i, _st[j]]) - 1, _st[j]]
        # print(q, cr1.WesMiqdoriyFeature(newX[:, q], y))
    return newX


# ======== experiment 3 (Klassifikatsiya metodlarini ishlatish) ========
def ClassificationAccuracyWorldMethods(X, y, s=0.3, r=42):
    results = []
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB

    if s == 0:
        Xtrain = X
        Xtest = X
        train_labels = y
        test_labels = y
    else:
        Xtrain, Xtest, train_labels, test_labels = train_test_split(X, y, test_size=s, random_state=r)
    # NaiveBayes
    gnb = GaussianNB()
    model = gnb.fit(Xtrain, train_labels)
    predictive_labels = gnb.predict(Xtest)
    # print("{0}\t{1:0.3f}".format("NaiveBayes", accuracy_score(test_labels, predictive_labels)))
    results.append(accuracy_score(test_labels, predictive_labels))
    from sklearn.neighbors import KNeighborsClassifier

    # KNN
    gnb = KNeighborsClassifier(n_neighbors=11)

    model = gnb.fit(Xtrain, train_labels)
    predictive_labels = gnb.predict(Xtest)
    # print("{0}\t{1:0.3f}".format("KNN", accuracy_score(test_labels, predictive_labels)))
    results.append(accuracy_score(test_labels, predictive_labels))

    # SVM
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    gnb = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    model = gnb.fit(Xtrain, train_labels)
    predictive_labels = gnb.predict(Xtest)
    # print("{0}\t{1:0.3f}".format("SVM", accuracy_score(test_labels, predictive_labels)))
    results.append(accuracy_score(test_labels, predictive_labels))

    # RandomForest
    from sklearn.ensemble import RandomForestClassifier

    gnb = RandomForestClassifier(max_depth=2, random_state=0)
    model = gnb.fit(Xtrain, train_labels)
    predictive_labels = gnb.predict(Xtest)
    # print("{0}\t{1:0.3f}".format("RandomForest", accuracy_score(test_labels, predictive_labels)))
    results.append(accuracy_score(test_labels, predictive_labels))

    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    """regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_1.fit(Xtrain, train_labels)
    predictive_labels = regr_1.predict(Xtest)
    print(test_labels)
    print(predictive_labels)
    results.append(accuracy_score(test_labels, predictive_labels))"""
    gnb = DecisionTreeClassifier()
    model = gnb.fit(Xtrain, train_labels)
    predictive_labels = gnb.predict(Xtest)
    # print("{0}\t{1:0.3f}".format("RandomForest", accuracy_score(test_labels, predictive_labels)))
    results.append(accuracy_score(test_labels, predictive_labels))

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score

    gnb = LinearDiscriminantAnalysis()
    model = gnb.fit(Xtrain, train_labels)
    predictive_labels = gnb.predict(Xtest)
    results.append(accuracy_score(test_labels, predictive_labels))

    return results


def ClassificationAccuracyWorldMethodsNewObject(X, y, x):
    results = []
    probas = []

    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB

    Xtrain = X
    Xtest = [x]
    train_labels = y
    # NaiveBayes
    gnb = GaussianNB()
    model = gnb.fit(Xtrain.values, train_labels)
    predictive_labels = gnb.predict(Xtest)
    predictive_probas = gnb.predict_proba(Xtest)
    results.append(predictive_labels)
    probas.append(predictive_probas)

    # KNN
    from sklearn.neighbors import KNeighborsClassifier
    gnb = KNeighborsClassifier(n_neighbors=11)

    model = gnb.fit(Xtrain.values, train_labels)
    predictive_labels = gnb.predict(Xtest)
    predictive_probas = gnb.predict_proba(Xtest)
    results.append(predictive_labels)
    probas.append(predictive_probas)

    # SVM
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    gnb = make_pipeline(StandardScaler(), SVC(gamma="auto", probability=True, random_state=0))
    model = gnb.fit(Xtrain.values, train_labels)
    predictive_labels = gnb.predict(Xtest)
    predictive_probas = gnb.predict_proba(Xtest)
    results.append(predictive_labels)
    probas.append(predictive_probas)

    # RandomForest
    from sklearn.ensemble import RandomForestClassifier

    gnb = RandomForestClassifier(max_depth=2, random_state=0)
    model = gnb.fit(Xtrain.values, train_labels)
    predictive_labels = gnb.predict(Xtest)
    predictive_probas = gnb.predict_proba(Xtest)
    results.append(predictive_labels)
    probas.append(predictive_probas)

    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    """regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_1.fit(Xtrain, train_labels)
    predictive_labels = regr_1.predict(Xtest)
    print(test_labels)
    print(predictive_labels)
    results.append(accuracy_score(test_labels, predictive_labels))"""
    gnb = DecisionTreeClassifier(random_state=0)
    model = gnb.fit(Xtrain.values, train_labels)
    predictive_labels = gnb.predict(Xtest)
    predictive_probas = gnb.predict_proba(Xtest)
    results.append(predictive_labels)
    probas.append(predictive_probas)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score

    gnb = LinearDiscriminantAnalysis()
    model = gnb.fit(Xtrain.values, train_labels)
    predictive_labels = gnb.predict(Xtest)
    predictive_probas = gnb.predict_proba(Xtest)
    results.append(predictive_labels)
    probas.append(predictive_probas)

    return results, probas
