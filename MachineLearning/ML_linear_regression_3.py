from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation =='neg':
            val -=step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def bestFitSlopeAndIntercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b


def squaredError(ysOrig, ysLine):
    return sum((ysLine-ysOrig)**2)


def coefficient_of_determination(ysOrig, ysLine):
    y_mean_line = [mean(ysOrig) for y in ysOrig]
    squared_error_regr = squaredError(ysOrig, ysLine)
    squared_error_y_mean = squaredError(ysOrig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


xs, ys = create_dataset(40, 10, 2, correlation=False)

m, b = bestFitSlopeAndIntercept(xs, ys)

regressionLine = [(m * x) + b for x in xs]

predict_x = 8
predict_y = (m * predict_x) + b

r_squared = coefficient_of_determination(ys, regressionLine)

print(r_squared)
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y)
plt.plot(xs, regressionLine)
plt.show()
