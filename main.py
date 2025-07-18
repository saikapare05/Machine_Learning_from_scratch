import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

#'data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

diabetes = datasets.load_diabetes()
#diabetes_X = diabetes.data[:,np.newaxis,6]
diabetes_X = diabetes.data
#print(diabetes_X)
#for feature
diabetes_X_train = diabetes_X[1:30]
diabetes_X_test = diabetes_X[:20]
#for label
diabetes_y_train = diabetes.target[1:30]
diabetes_y_test = diabetes.target[:20]
model = linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_y_train)
diabetes_y_predicated= model.predict(diabetes_X_test)
#mean squred error
print("Mean squared error is ", mean_squared_error(diabetes_y_test,diabetes_y_predicated))
print("Weights",model.coef_)
print("Intercept",model.intercept_)



#plt.scatter(diabetes_X_test,diabetes_y_test)
#for line
#plt.plot(diabetes_X_test,diabetes_y_predicated)
#plt.show()
'''Mean squared error is  1162.2498456437354
Weights [-137.02119948  130.33629511   97.9008855    73.95005618  385.64198049
 -519.76460948 -266.82932135 -181.5922573  1214.17181648 -599.12469015]
Intercept 138.18133391773137
'''