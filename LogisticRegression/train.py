import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

classification_dataset = datasets.load_breast_cancer()
X, y = classification_dataset.data, classification_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression(n_epochs=1000, learning_rate=0.0001)
clf.fit(X_train, y_train)
pred_y = clf.predict(X_test)
acc = np.mean([y_test == pred_y])
print(f'Test accuracy : {acc:.3f}')