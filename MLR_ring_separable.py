import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# A2-ring-separable data
#read and parse the training set .csv features file 
df_train = pd.read_csv('Data/A2-ring/A2-ring-separable.txt', delimiter = '\t', header=None)

# construct an array of arrays of two-input features of the train set
input_train_columns = df_train.columns[0 : 2]
train_features = df_train[input_train_columns].values

# construct an array of size of output class labels target values
outputcolumn = df_train.columns[2]
train_labels = df_train[outputcolumn].values


#read and parse the test set .csv features file
df_test = pd.read_csv('Data/A2-ring/A2-ring-test.txt', delimiter = '\t', header=None)

# construct an array of arrays of two-input features of the train set
input_test_columns = df_test.columns[0 : 2]
test_features = df_test[input_test_columns].values

# construct an array of size of output class labels target values
outputcolumn = df_test.columns[2]
test_labels = df_test[outputcolumn].values

reg = LinearRegression().fit(train_features, train_labels)
score = reg.score(train_features, train_labels)
print(score)

predicted_labels_test = reg.predict(test_features)
predicted_labels_test = np.abs(np.round(predicted_labels_test, decimals=0))
print(predicted_labels_test)