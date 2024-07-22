import traceback
import numpy as np


def criteria1_with_nominal(X, y, types=None):
    m = X.shape[0]
    n = X.shape[1]
    if types is None:
        types = [1] * n
    informativnost = []

    _, ln = np.unique(y, return_counts=True)
    k1 = ln[0]
    k2 = sum(ln) - k1
    ln = [k1, k2]

    for q in range(n):
        # print(q)
        x = X[:, q]
        if types[q] == 1:
            inf = WesNominalFeture(x, y)
            informativnost += [inf]
            continue
        # if the feature is scale
        # indexes of sorted x
        arg_sort = np.argsort(x)

        # number of objects in each interval and class
        u = np.array([[0, 0], [0, 0]])
        # the result of the function
        max_val = 0
        # The number index of optimal-border number, by default the minimum value of the feature
        opt_index = 0
        m1 = ln[0] * (ln[0] - 1) + ln[1] * (ln[1] - 1)
        m2 = 2 * ln[0] * ln[1]
        for i in range(x.shape[0] - 1):
            u[0, int(y[arg_sort[i]]) - 1] += 1
            # Calculate len of object's by class in second interval
            u[1, 0] = ln[0] - u[0, 0]
            u[1, 1] = ln[1] - u[0, 1]

            # if the current object and the next object are not equal
            if x[arg_sort[i]] != x[arg_sort[i + 1]]:
                # the first sum
                sum1 = 0
                # the second sum
                sum2 = 0
                # method
                for j in range(0, 2):
                    for l in range(0, 2):
                        sum1 += u[j][l] * (u[j][l] - 1)
                        sum2 += u[j][l] * (ln[1 - l] - u[j][1 - l])

                current_max = (sum1 / m1) * (sum2 / m2)

                # Check current max than more max value
                if current_max > max_val:
                    max_val = current_max
                    opt_index = i
        informativnost += [max_val]
        # return max_val, (x[arg_sort[opt_index]] + x[arg_sort[opt_index + 1]]) / 2

    return informativnost


def criteria1(X, y):
    m = X.shape[0]
    n = X.shape[1]

    informativnost = []

    _, ln = np.unique(y, return_counts=True)
    k1 = ln[0]
    k2 = sum(ln) - k1
    ln = [k1, k2]

    print("===========cr1==============")
    for q in range(n):
        # print(q)
        x = X[:, q]

        # if the feature is scale
        # indexes of sorted x
        arg_sort = np.argsort(x)

        # number of objects in each interval and class
        u = np.array([[0, 0], [0, 0]])
        # the result of the function
        max_val = 0
        # The number index of optimal-border number, by default the minimum value of the feature
        opt_index = 0
        m1 = ln[0] * (ln[0] - 1) + ln[1] * (ln[1] - 1)
        m2 = 2 * ln[0] * ln[1]
        for i in range(x.shape[0] - 1):
            u[0, int(y[arg_sort[i]]) - 1] += 1
            # Calculate len of object's by class in second interval
            u[1, 0] = ln[0] - u[0, 0]
            u[1, 1] = ln[1] - u[0, 1]

            # if the current object and the next object are not equal
            if x[arg_sort[i]] != x[arg_sort[i + 1]]:
                # the first sum
                sum1 = 0
                # the second sum
                sum2 = 0
                # method
                for j in range(0, 2):
                    for l in range(0, 2):
                        sum1 += u[j][l] * (u[j][l] - 1)
                        sum2 += u[j][l] * (ln[1 - l] - u[j][1 - l])

                current_max = (sum1 / m1) * (sum2 / m2)

                # Check current max than more max value
                if current_max > max_val:
                    max_val = current_max
                    opt_index = i
        informativnost += [max_val]
        # return max_val, (x[arg_sort[opt_index]] + x[arg_sort[opt_index + 1]]) / 2

    return informativnost


def WesNominalFeture(x, y):
    uniq_class, ln = np.unique(y, return_counts=True)
    uniq_grad = np.unique(x)
    g = np.empty(shape=(uniq_grad.shape[0], ln.shape[0]))
    l = np.empty(shape=(ln.shape[0]))
    for i in range(uniq_grad.shape[0]):
        for j in range(uniq_class.shape[0]):
            g[i][j] = len(x[np.logical_and(x == uniq_grad[i], y == uniq_class[j])])

    for i in range(uniq_class.shape[0]):
        l[i] = len(np.unique(x[y == uniq_class[i]]))

    dominator = 1
    for item in ln:
        dominator *= item

    s = 0
    for item in g:
        p = 1
        for i in item:
            p *= i
        s += p

    lymada = 1 - s / dominator

    d = [0, 0]
    if uniq_grad.shape[0] > 2:
        d[0] = (ln[0] - l[0] + 1) * (ln[0] - l[0])
        d[1] = (ln[1] - l[1] + 1) * (ln[1] - l[1])
    else:
        d[0] = ln[0] * (ln[0] - 1)
        d[1] = ln[1] * (ln[1] - 1)

    betta = 0
    if d[0] + d[1] > 0:
        for i in range(uniq_grad.shape[0]):
            for j in range(uniq_class.shape[0]):
                betta += g[i][j] * (g[i][j] - 1)
        betta /= d[0] + d[1]

    return betta * lymada


def WesMiqdoriyFeature(x, y):
    """
    Miqdoriy alomat vaznini hisoblash
    Arguments:
        x: alomat vektori
        y: sinf vektori
    Returns:
        alomat vazni, chegara qiymati
    """
    _, ln = np.unique(y, return_counts=True)
    # k1 = ln[0]
    # k2 = sum(ln) - k1
    # ln = [k1, k2]

    arg_sort = np.argsort(x)

    # number of objects in each interval and class
    u = np.array([[0, 0], [0, 0]])
    # the result of the function
    max_val = 0
    # The number index of optimal-border number, by default the minimum value of the feature
    opt_index = 0
    m1 = ln[0] * (ln[0] - 1) + ln[1] * (ln[1] - 1)
    m2 = 2 * ln[0] * ln[1]
    for i in range(x.shape[0] - 1):
        u[0, int(y[arg_sort[i]]) - 1] += 1
        # Calculate len of object's by class in second interval
        u[1, 0] = ln[0] - u[0, 0]
        u[1, 1] = ln[1] - u[0, 1]

        # if the current object and the next object are not equal
        if x[arg_sort[i]] != x[arg_sort[i + 1]]:
            # the first sum
            sum1 = 0
            # the second sum
            sum2 = 0
            # method
            for j in range(0, 2):
                for l in range(0, 2):
                    sum1 += u[j][l] * (u[j][l] - 1)
                    sum2 += u[j][l] * (ln[1 - l] - u[j][1 - l])

            current_max = (sum1 / m1) * (sum2 / m2)

            # Check current max than more max value
            if current_max > max_val:
                max_val = current_max
                opt_index = i
    return max_val, x[arg_sort[opt_index]]


def criteria1_nominal(X, y):
    m = X.shape[0]
    n = X.shape[1]

    informativnost = []

    _, ln = np.unique(y, return_counts=True)
    k1 = ln[0]
    k2 = sum(ln) - k1
    ln = [k1, k2]

    print("===========cr1==============")
    for q in range(n):
        x = X[:, q]
        """uniques = np.unique(x)
        p = uniques.shape[0]
        ldr1 = np.unique(np.logical_and())
        ldr2 = uniques.shape[0]
        gdr1 = []
        gdr2 = []
        for l in uniques:
            gdr1.append(np.count_nonzero(np.logical_and(x == l, y == 1)))
            gdr2.append(np.count_nonzero(np.logical_and(x == l, y == 2)))
        sum_mul = np.sum(np.multiply(gdr1, gdr2))
        lambdaR = 1 - (sum_mul) / (k1 * k2)
        """
        inf = WesNominalFeture(x, y)
        informativnost += [inf]
        # return max_val, (x[arg_sort[opt_index]] + x[arg_sort[opt_index + 1]]) / 2

    return informativnost


def WesNominalFetureWithNaN(x, y):
    uniq_class, ln = np.unique(y[~np.isnan(x)], return_counts=True)

    if len(uniq_class) == 1:
        ln = np.hstack([ln, 0])
        if uniq_class[0] == 2:
            ln[1] = ln[0]
            ln[0] = 0

    if len(uniq_class) == 0:
        ln = np.hstack([ln, 0])
        ln = np.hstack([ln, 0])
        return 0, 0, ln

    uniq_grad = np.unique(x[~np.isnan(x)])
    g = np.empty(shape=(uniq_grad.shape[0], ln.shape[0]))
    l = np.empty(shape=(ln.shape[0]))
    for i in range(uniq_grad.shape[0]):
        for j in range(uniq_class.shape[0]):
            g[i][j] = len(x[np.logical_and(x == uniq_grad[i], y == uniq_class[j])])

    if len(uniq_class) == 1:
        if uniq_class[0] == 1:
            return 1, g, ln
        else:
            return 0, g, ln

    for i in range(uniq_class.shape[0]):
        l[i] = len(np.unique(x[y == uniq_class[i]]))

    dominator = 1
    for item in ln:
        dominator *= item

    s = 0
    for item in g:
        p = 1
        for i in item:
            p *= i
        s += p

    lyambda = 1 - s / dominator

    d = [0, 0]
    if uniq_grad.shape[0] > 2:
        d[0] = (ln[0] - l[0] + 1) * (ln[0] - l[0])
        d[1] = (ln[1] - l[1] + 1) * (ln[1] - l[1])
    else:
        d[0] = ln[0] * (ln[0] - 1)
        d[1] = ln[1] * (ln[1] - 1)

    betta = 0
    if d[0] + d[1] > 0:
        for i in range(uniq_grad.shape[0]):
            for j in range(uniq_class.shape[0]):
                betta += g[i][j] * (g[i][j] - 1)
        betta /= d[0] + d[1]
    return betta * lyambda, g, ln


def RSByBinarFeature(X, y, stabilities):
    wes_binar = []
    g_binar = []
    ln_binar = []
    for a in range(X.shape[1]):
        w, g, ln = WesNominalFetureWithNaN(X[:, a], y)
        # print(a, w)
        wes_binar.append(w)
        g_binar.append(g)
        ln_binar.append(ln)

    # RS hisoblash
    RS_binar = np.empty(shape=X.shape[0])
    myu = np.empty(shape=(2, X.shape[1]))
    for i in range(X.shape[0]):
        rs = 0
        for j in range(X.shape[1]):
            if stabilities[j] == 0.5:
                continue
            g = g_binar[j]
            ln = ln_binar[j]
            w = wes_binar[j]
            if np.isnan(X[i, j]) == False:
                if w == 0:
                    myu[0, j] = 0
                    myu[1, j] = 0
                elif w == 1:
                    myu[0, j] = 1
                    myu[1, j] = 1
                else:
                    myu[0, j] = w * (g[0][0] / ln[0] - g[0][1] / ln[1])
                    if len(g) != 1:
                        myu[1, j] = w * (g[1][0] / ln[0] - g[1][1] / ln[1])
                rs = rs + myu[int(X[i, j]) - 1, j]
                # if i == 15:
                #    print(w, g, ln, myu[0, j], myu[1, j], int(X[i, j]) - 1)
        RS_binar[i] = rs
    return RS_binar, myu  # , wes_binar


def RSByNominalFeature(X, y):
    wes_binar = []
    g_binar = []
    ln_binar = []
    for a in range(X.shape[1]):
        betta, lyamda, g, ln = WesNominalFetureWithNaNExperimentFarmi(X[:, a], y)
        wes_binar.append(betta * lyamda)
        g_binar.append(g)
        ln_binar.append(ln)

    RS = np.empty(shape=X.shape[0])
    myu = np.empty(shape=(4, X.shape[1]))
    for i in range(X.shape[0]):
        rs = 0
        for j in range(X.shape[1]):
            g = g_binar[j]
            ln = ln_binar[j]
            w = wes_binar[j]
            # print(g, ln)
            if np.isnan(X[i, j]) == False:
                if w == 0:
                    myu[0, j] = 0
                    myu[1, j] = 0
                    myu[2, j] = 0
                    myu[3, j] = 0
                elif w == 1:
                    myu[0, j] = 1
                    myu[1, j] = 1
                    myu[2, j] = 1
                    myu[3, j] = 1
                else:
                    myu[int(X[i, j]) - 1, j] = w * (g[int(X[i, j]) - 1][0] / ln[0] - g[int(X[i, j]) - 1][1] / ln[1])
                    # if len(g) == 2:
                    #     myu[1, j] = w * (g[1][0] / ln[0] - g[1][1] / ln[1])
                    # if len(g) == 3:
                    #     myu[2, j] = w * (g[2][0] / ln[0] - g[2][1] / ln[1])
                    # if len(g) == 4:
                    #     myu[3, j] = w * (g[3][0] / ln[0] - g[3][1] / ln[1])
                # print(j, int(X[i, j]), w, myu[int(X[i, j]) - 1, j], g[int(X[i, j]) - 1][0], g[int(X[i, j]) - 1][1], ln[0], ln[1])
                rs = rs + myu[int(X[i, j]) - 1, j]
                # if i == 15:
                #    print(w, g, ln, myu[0, j], myu[1, j], int(X[i, j]) - 1)
        # print("rs=", rs)
        # break
        RS[i] = rs
    return RS, myu, wes_binar
