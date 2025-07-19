import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Your custom small X and y values
diabetes_X = np.array([[1], [2], [3]])  # shape: (3,1)
diabetes_y_train = np.array([3, 2, 4])
diabetes_y_test = np.array([3, 2, 4])

# Use same data for training and testing
diabetes_X_train = diabetes_X
diabetes_X_test = diabetes_X

# Create linear regression model
model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)

# Predict using model
diabetes_y_predicated = model.predict(diabetes_X_test)

# Mean squared error
print("Mean squared error is", mean_squared_error(diabetes_y_test, diabetes_y_predicated))
print("Weights", model.coef_)
print("Intercept", model.intercept_)

# Plotting (optional)
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_predicated, color="blue", linewidth=2)
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
