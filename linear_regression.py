import pandas as pd
import quandl, math, datetime
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import pickle


style.use('ggplot')

# Credit https://www.youtube.com/watch?v=lN5jesocJjk&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=3

# Get data set
df = quandl.get('WIKI/GOOGL')

# Select columns
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

# Create two useful features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# Set dataframe to desired features
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# Prediction! This var is set to 35
forecast_out = int(math.ceil(0.01*len(df)))
print("Predicting {} days in the future".format(forecast_out))

# Shift closing price X amount of days in the future to todays date...This creates a dataframe where the future price was guessed 100% correct!
df['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)

# Set X to all features - Just remove the label
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

df.dropna(inplace=True)

#X = X[-forecast_out
# y is equal to only the label
y = np.array(df['label'])
# Magic
X = preprocessing.scale(X)

# Use 20% for testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Best algorithm
clf = LinearRegression(n_jobs=-1)

#clf = svm.SVR(kernel='poly')
# Bad
#clf = svm.SVR(kernel='poly')

# Magic / Training
clf.fit(X_train, y_train)
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)

#print("The algorithm was correct {} % of the time".format(accuracy))

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

# nan forecast
df['Forecast'] = np.nan

# Setting up time for chart
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # For new data...nan all useless columns except for forecast then set value
    # Forecast is the last column
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

print(df.head())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()