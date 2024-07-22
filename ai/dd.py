import numpy as np

import matplotlib.pyplot as plt


def AcceptableInterval(x, y, method="mean"):

    value_x = None
    value_y = None

    if method == "mean":
        value_x = x.mean()
        value_y = y.mean()
    else:
        value_x = np.median(x)
        value_y = np.median(y)

    R = x / value_x - y / value_y

    return R, value_x, value_y


def dd_for_dataset(X, headers):

    card = np.array([i + 1 for i in range(X.shape[0])])
    eps = [0.5, 0.2, 0.1]
    q = set()
    k = -1

    for i in range(X.shape[1] - 1):
        for j in range(i + 1, X.shape[1]):
            # k += 1
            # k = 0
            # print(headers[i], headers[j])
            x = X[:, i]
            y = X[:, j]
            R, value_x, value_y = AcceptableInterval(x, y)
            # print(R)
            # print(*card[R.argsort()])

            # marker = ["v", "x", "+"]
            # size = 7

            # eps = [(abs(R.min()) + abs(R.max())) * 0.05]

            # black = np.logical_and(eps[k] + R.min() <= R, R <= R.max() - eps[k])
            # red = np.logical_not(black)

            # print(sorted(card[red]))

            # q.update(card[red])

            # plt.xlabel("Объекты")
            # plt.ylabel("Интервал")
            # plt.xlabel("Object numbers")
            # plt.ylabel("Interval")
            # plt.xlabel("Obyektlar nomeri")
            # plt.ylabel("Interval")

            # plt.plot(card[black], R[black], marker[0], ms=size, markerfacecolor="None", alpha=1, markeredgecolor="black", markeredgewidth=1.5)
            # plt.plot(card[red], R[red], marker[1], ms=size, markerfacecolor="None", alpha=1, markeredgecolor="black", markeredgewidth=1.5)
            # print(R.min(), R.max(), eps)
            # x1, y1 = [-5, 570], [eps + R.min(), eps + R.min()]
            # x2, y2 = [-5, 570], [R.max() - eps, R.max() - eps]
            # plt.plot(x1, y1, x2, y2, color="black")

            # plt.title("{} - {}".format(headers[i], headers[j]))

            # plt.show()

    # print(sorted(q))
    # print(len(q))


def main():
    headers = ["Height", "Weight", "Age"]
    headers = ["Bo'y", "Vazn", "Yosh"]

    X_train = np.loadtxt(r"d:\UzMU\PhD\python\dirty-data-2\data_csv.txt", delimiter=";")
    X_test = np.loadtxt(r"d:\UzMU\PhD\python\dirty-data-2\data1.txt", delimiter=";")
    # X_train_test = np.vstack((X_train, X_test))
    X_train_test = X_train

    X_train_test = X_train_test[:, [0, 1, 2]]

    # X_train_test = X_train_test[X_train_test[:, 3] == 1]
    # X_train_test = X_train_test[:, [0, 1, 2]]

    card = np.array([i + 1 for i in range(X_train_test.shape[0])])
    eps = [0.5, 0.2, 0.1]

    # print(*headers, sep="|\t")
    # print(*X_train_test.max(axis=0), sep="|\t")
    # print(*X_train_test.min(axis=0), sep="|\t")

    q = set()

    k = -1

    for i in range(X_train_test.shape[1] - 1):
        for j in range(i + 1, X_train_test.shape[1]):
            # k += 1
            k = 0
            print(headers[i], headers[j])
            x = X_train_test[:, i]
            y = X_train_test[:, j]
            R, value_x, value_y = AcceptableInterval(x, y)
            # print(R)
            # print(*card[R.argsort()])

            marker = ["v", "x", "+"]
            size = 7

            eps = [(abs(R.min()) + abs(R.max())) * 0.05]

            black = np.logical_and(eps[k] + R.min() <= R, R <= R.max() - eps[k])
            red = np.logical_not(black)

            # print(sorted(card[red]))

            q.update(card[red])

            plt.xlabel("Объекты")
            plt.ylabel("Интервал")
            plt.xlabel("Object numbers")
            plt.ylabel("Interval")
            plt.xlabel("Obyektlar nomeri")
            plt.ylabel("Interval")

            plt.plot(card[black], R[black], marker[0], ms=size, markerfacecolor="None", alpha=1, markeredgecolor="black", markeredgewidth=1.5)
            plt.plot(card[red], R[red], marker[1], ms=size, markerfacecolor="None", alpha=1, markeredgecolor="black", markeredgewidth=1.5)
            print(R.min(), R.max(), eps)
            x1, y1 = [-5, 570], [eps + R.min(), eps + R.min()]
            x2, y2 = [-5, 570], [R.max() - eps, R.max() - eps]
            plt.plot(x1, y1, x2, y2, color="black")

            plt.title("{} - {}".format(headers[i], headers[j]))

            plt.show()
            # plt.savefig(r"D:\Nuu\AI\Researchs\Tuberculosis\Documents\Reports\images of acceptability by interval\({}-{}).eps".format(headers[i], headers[j]), format="eps")
            # plt.savefig(r"D:\Nuu\AI\Researchs\Tuberculosis\Documents\Reports\images of acceptability by interval\({}-{})-en".format(headers[i], headers[j]))

    print(sorted(q))
    print(len(q))

