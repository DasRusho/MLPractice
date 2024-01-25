import numpy as np
import random
import matplotlib.pyplot as plt
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from KNN import KNN

'''
def my_train_test_split(data_x, data_y, test_proportion = 0.3, random_seed = 12): # custom train/test split function
    data_len = data_x.shape[0]
    test_data_size = int(data_len*test_proportion)
    random.seed(random_seed) # for reproducibility
    test_indices = random.sample([i for i in range(data_len)],test_data_size)
    X_train, X_test, y_train, y_test = [],[],[],[]
    for idx in range(data_len):
        if idx in test_indices:
            X_test.append(data_x[idx])
            y_test.append(data_y[idx])
        else:
            X_train.append(data_x[idx])
            y_train.append(data_y[idx])
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
'''

iris_data = datasets.load_iris()
X, y = iris_data.data, iris_data.target
#X_train, X_test, y_train, y_test = my_train_test_split(X, y, 0.2, 12345)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)
print(y_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)