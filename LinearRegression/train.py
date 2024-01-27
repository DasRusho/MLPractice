import numpy as np
from DataSet import LinearRegressionDataSet
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

# Create DataSet
data = LinearRegressionDataSet(n_features=5, n_samples=1000, weights=[7,3,9,12,47], bias=3)
X, y = data.createDataSet()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

# Fit LR
reg = LinearRegression(learning_rate=0.0001,n_epochs=100) # using higher learning rate can result in divergence
reg.fit(X_train,y_train)

# Predict
y_pred = reg.predict(X_test)
rmse = (np.mean((y_pred-y_test)**2))**0.5
print(f'Test RMSE = {rmse : 0.3f}')
print(reg.weights) # compare with wights in LinearRegressionDataSet (line 7)