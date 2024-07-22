# 1 nominal, 0 miqdoriy
import warnings
import traceback
import numpy as np

# full python, slow performance
def DivideIntervals(x, y, ln, return_intervals=False):
    a = 0
    # b = y.shape[0]
    b = ln[0] + ln[1]
    # index of sorted x
    _x = np.argsort(x)
    intervals = []

    def inner(a, b):
        rng_opt = None
        w_opt = 0
        for i in range(a, b):
            if i == 0 or i > 0 and x[_x[i]] != x[_x[i - 1]]:
                j = i
                d = [0, 0]
                while j < b and not np.isnan(x[_x[j]]):
                    d[int(y[_x[j]]) - 1] += 1
                    if j == b - 1 or j < b - 1 and x[_x[j]] != x[_x[j + 1]]:
                        nyu = [d[0] / ln[0], d[1] / ln[1]]
                        w_current = abs(nyu[0] - nyu[1])
                        if w_current >= w_opt:
                            f = nyu[0] / (nyu[0] + nyu[1])
                            rng_opt = [i, j, f, x[_x[i]], x[_x[j]]]
                            w_opt = w_current

                    j += 1
        if rng_opt:
            intervals.append(rng_opt)
            # Go to left side
            print(a, b, rng_opt[0], rng_opt[1])
            if rng_opt[0] > a:
                inner(a, rng_opt[0])
            # Go to right side
            if rng_opt[1] < b:
                inner(rng_opt[1] + 1, b)

    inner(a, b)

    group_index = np.full(shape=(y.shape[0]), fill_value=np.nan)
    group_estimation = np.full(shape=(y.shape[0]), fill_value=np.nan)
    for i in range(len(intervals)):
        for item in range(intervals[i][0], intervals[i][1] + 1):
            if np.isnan(x[_x[item]]):
                group_index[_x[item]] = np.nan
                group_estimation[_x[item]] = np.nan
                x[_x[item]] = np.nan
            else:
                group_index[_x[item]] = i
                group_estimation[_x[item]] = intervals[i][2]
                x[_x[item]] = intervals[i][2]
    if return_intervals:
        return group_index, group_estimation, intervals
    return group_index, group_estimation


def DivideIntervalsNominalFeture(x, y, ln):
    try:
        uniq_grad, uniq_count = np.unique(x, return_counts=True)
        g = np.empty(shape=(uniq_grad.shape[0], 2))
        intervals = []
        for i in range(uniq_grad.shape[0]):
            if ~np.isnan(uniq_grad[i]):
                cond = x == uniq_grad[i]
                g[i, 0] = np.count_nonzero(np.logical_and(cond, y == 1) == True)
                g[i, 1] = uniq_count[i] - g[i, 0]

                if ln[0] == 0:
                    nyu1 = 0
                else:
                    nyu1 = g[i, 0] / ln[0]
                if ln[1] == 0:
                    nyu2 = 0
                else:
                    nyu2 = g[i, 1] / ln[1]

                x[cond] = nyu1 / (nyu1 + nyu2)
                # print(uniq_grad[i], g[i, 0], g[i, 1], x[cond])

                intervals.append([uniq_grad[i], nyu1 / (nyu1 + nyu2)])
            else:
                break
    except:
        traceback.print_exc()
    return intervals


def Stability(X, y, types):
    stabilities = np.zeros(shape=(X.shape[1]))
    X_binar = np.empty(shape=(X.shape[0], 0), dtype="int64")
    intervals = []
    for i in range(X.shape[1]):
        x = X[:, i].copy()
        _, ln = np.unique(y[~np.isnan(x)], return_counts=True)
        # print("----------------------------------------------------------")
        # if i == 3:
            # print(i, _, ln, x)
        # print("----------------------------------------------------------")
        
        if len(_) > 0 and len(_) < 2:
            ln = np.hstack([ln, 0])
            if _[0] == 2:
                ln[1] = ln[0]
                ln[0] = 0
        if types[i] == 1:
            interval = DivideIntervalsNominalFeture(x, y, ln)
        else:
            group_index, group_estimation, interval = DivideIntervals(x, y, ln, True)
        intervals.append(interval)

        # print("----------------------------------------------------------")
        # if i == 3:
        #     print(i, interval)
        # print("----------------------------------------------------------")

        s = 0
        for j in range(x.shape[0]):
            if np.isnan(x[j]) == False:
                if x[j] > 0.5:
                    s += x[j]
                else:
                    s += 1 - x[j]
        stabilities[i] = s / ln.sum()

        warnings.filterwarnings("ignore")

        # binar datani hosil qilish
        x[x >= 0.5] = 2
        x[x < 0.5] = 1
        xb = np.array([x]).T
        X_binar = np.append(X_binar, xb, axis=1)

    return stabilities, intervals, X_binar


def NonlinearTransformation(X, y, intervals):
    X_binar = X.copy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(X[i, j]) == False:
                for interval in intervals[j]:
                    if X_binar[i, j] >= interval[3] and X_binar[i, j] <= interval[4] and interval[2] > 0.5:
                        X_binar[i, j] = 1
                        break
    return X_binar

