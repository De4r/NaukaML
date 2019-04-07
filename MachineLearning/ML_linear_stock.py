import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')
quandl.ApiConfig.api_key = 'pRbQXy4e6crm13NdV_KG'
dataFrame = quandl.get('WIKI/GOOGL')


dataFrame = dataFrame[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
dataFrame['HL_PCT'] = (dataFrame['Adj. High'] - dataFrame["Adj. Close"]) / dataFrame['Adj. Close'] * 100.0
dataFrame['PCT_change'] = (dataFrame['Adj. Close'] - dataFrame["Adj. Open"]) / dataFrame['Adj. Open'] * 100.0

dataFrame = dataFrame[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

foreCastCol = 'Adj. Close'

dataFrame.fillna(-99999, inplace=True)

foreCastOut = int(math.ceil(0.1*len(dataFrame)))
print(foreCastOut)
dataFrame['label'] = dataFrame[foreCastCol].shift(-foreCastOut)

X = np.array(dataFrame.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-foreCastOut:]
X = X[:-foreCastOut]


dataFrame.dropna(inplace=True)
y = np.array(dataFrame['label'])


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# clf = LinearRegression(n_jobs=-1) #svm.SVR()#
# clf.fit(X_train, y_train)
# with open('linearregresion.pickle','wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('linearregresion.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

foreCastSet = clf.predict(X_lately)
print(foreCastSet, accuracy, foreCastOut)

dataFrame['Forecast'] = np.nan

lastDate = dataFrame.iloc[-1].name
lastUnix = lastDate.timestamp()
oneDay = 86400
nextUnix = lastUnix + oneDay

for i in foreCastSet:
    nextDate = datetime.datetime.fromtimestamp(nextUnix)
    nextUnix += oneDay
    dataFrame.loc[nextDate] = [np.nan for __ in range(len(dataFrame.columns)-1)] + [i]

print(dataFrame.head())
print(dataFrame.tail())

dataFrame['Adj. Close'].plot()
dataFrame['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Data')
plt.ylabel('Price')
plt.show()