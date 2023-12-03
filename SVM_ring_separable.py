import pandas as pd
from libsvm.svmutil import *
import numpy as np
import scatter_plot

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


# train the model using the training set
model = svm_train(train_labels, train_features, '-t 2 -g 200 -c 100')

# Predict labels using test set
predicted_labels, accuracy, decision_values = svm_predict(test_labels, test_features, model)

predicted_labels = np.array(predicted_labels)
x = test_features[:,0]
y = test_features[:,1]
output = predicted_labels
sizes = np.full((1, output.shape[0]), 1)
scatter_plot.scatter_plot(x, y, output, sizes, 'Scatter Plot of predicted labels of ring test separable using SVM')


# # scatter plot of test true labels of separable class 0
# class_0_condition = test_labels == 1
# selected_values = test_labels[class_0_condition]
# indices = np.where(class_0_condition)
# x = test_features[indices][:,0]
# y = test_features[indices][:,1]
# output = test_labels[indices]
# sizes = np.full((1, output.shape[0]), 1)
# scatter_plot.scatter_plot(x, y, output, sizes, 'Scatter Plot of ring test true labels of separable class 0')

# # scatter plot of test predicted labels of separable class 0
# predicted_labels = np.array(predicted_labels)
# class_0_condition = predicted_labels == 1
# selected_values = predicted_labels[class_0_condition]
# indices = np.where(class_0_condition)
# x = test_features[indices][:,0]
# y = test_features[indices][:,1]
# output = predicted_labels[indices]
# sizes = np.full((1, output.shape[0]), 1)
# scatter_plot.scatter_plot(x, y, output, sizes, 'Scatter Plot of ring test predicted labels of separable class 0')

