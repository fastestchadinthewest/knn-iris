import numpy as np
from collections import Counter

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

def classify_point(indexes_of_nearest, y_labels):
    
    #create an array of just the k nearest neighbors
    nearest = y_labels[indexes_of_nearest]

    #count each label and return the most common one
    label_counter = Counter(nearest)
    # print("label: ", label_counter.most_common(1)[0][0])
    return label_counter.most_common(1)[0][0]

# needs prediction label and actual y for each example
def validate(X, y, k):
    #validate <- classify <- get_nearest_neighbors
    total = 0
    count = 0
    
    for i in range(len(X)):
        indexes_of_nearest = get_k_nearest_neighbors(X, X[i], k)
        prediction_label = classify_point(indexes_of_nearest, y)

        # debug
        #print(prediction_label)

        if prediction_label == y[i]:
            count = count + 1
        total = total + 1 
    # print("count:", count, ". Total: ", total)
    return count / total


# main
# declare variables
train_X, train_y, test_X, test_y = load_data()
k = 5
accuracy = validate(train_X, train_y, k)
print(accuracy)



   