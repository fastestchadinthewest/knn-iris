#!/usr/bin/env python3
import numpy as np
from collections import Counter

# load data
def load_data():
    data = np.genfromtxt('./bezdekIris.data', delimiter=',', dtype=None, encoding=None)

    #shuffle data
    np.random.shuffle(data)

    #separate X and y features
    X = np.array([[row[i] for i in range(4)] for row in data], dtype=float)
    y = np.array([row[4] for row in data])

    #separate into 80/20 split
    split = int(0.8 * len(data))

    #create train data
    train_X, train_y = X[:split], y[:split]

    #create test data
    test_X, test_y = X[split:], y[split:]

    #reshape y labels to make them "column vectors"
    # train_y, test_y = train_y[:, np.newaxis], test_y[:, np.newaxis]

    return train_X, train_y, test_X, test_y



# gets nearest neighbors
# example_set: set of examples (features only)
# query: a single x example
# k: the number of nearest neighbors to get

def get_k_nearest_neighbors(example_set, query, k):

    #create an array of the distances each example is from the query
    distances = np.linalg.norm(example_set - query, axis = 1)

    #find the indexes of the nearest neighbors to validate
    indexes_of_nearest = np.argsort(distances)[:k]

    return indexes_of_nearest

# using the nearest examples, look at their y values and classify current query
def classify_point(indexes_of_nearest, y_labels):
    
    #create an array of just the k nearest neighbors
    nearest = y_labels[indexes_of_nearest]

    #count each label and return the most common one
    label_counter = Counter(nearest)
    # print("label: ", label_counter.most_common(1)[0][0])
    return label_counter.most_common(1)[0][0]


# Calculate accuracy using actual y-labels
def validate(train_X, test_X, train_y, test_y, k):
    #validate <- classify <- get_nearest_neighbors

    # number of correct classifications
    count = 0
    
    for i in range(len(test_X)):

        # find nearest indexes in train set to individual query from test set
        indexes_of_nearest = get_k_nearest_neighbors(train_X, test_X[i], k)
        
        # create prediction label for query from train_x
        prediction_label = classify_point(indexes_of_nearest, train_y)
        
        # Increment total, and if label is right, add to count.
        if prediction_label == test_y[i]:
            count = count + 1

    return count / len((test_X))


# K FOLD CROSS VALIDATION
def cross_validation(X, y, num_folds, k):

    # make subarrays
    sub_arrs_X = np.split(X, num_folds, axis=0)
    sub_arrs_y = np.split(y, num_folds, axis=0)
    
    accuracy = []

    for i in range(num_folds):
        
        #use i as test set
        test_X = sub_arrs_X[i]
        test_y = sub_arrs_y[i]

        # stack all the other sets as the training sets
        new_train_X = np.vstack([sub_arrs_X[j] for j in range(num_folds) if j != i])        
        new_train_y = np.concatenate([sub_arrs_y[j] for j in range(num_folds) if j != i])
        
        acc = validate(new_train_X, test_X, new_train_y, test_y, k)
        accuracy.append(acc)
    
    return accuracy
    


# MAIN

# declare variables
train_X, train_y, test_X, test_y = load_data()
k = 5
num_folds = 4

#run K-fold cross validation on training data
accuracy = cross_validation(train_X, train_y, num_folds, k)
print(num_folds, "-fold cross validation accuracy: ", accuracy)

# run actual classification on test data

score = validate(train_X, test_X, train_y, test_y, k)
print("Score on test set: ", score)
