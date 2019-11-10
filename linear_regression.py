import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression


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

# Prediction!
forecast_out = int(math.ceil(0.01*len(df)))
print("Predicting {} days in the future".format(forecast_out))

# Shift closing price X amount of days in the future to todays date...This creates a dataframe where the future price was guessed 100% correct!
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# Set X to all features - Just remove the label
X = np.array(df.drop(['label'],1))
# y is equal to only the label
y = np.array(df['label'])
# Magic
X = preprocessing.scale(X)
#y = np.array(df['label'])

# Use 20% for testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Best algorithm
clf = LinearRegression()

#clf = svm.SVR(kernel='poly')
# Bad
#clf = svm.SVR(kernel='poly')

# Magic
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)

print("The alogirthm was correct {} % of the time".format(accuracy))
