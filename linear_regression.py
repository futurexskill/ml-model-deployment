import pandas as pd

"""## Read the dataset"""

dataset = pd.read_csv('studentmarks.csv')

dataset.describe()


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2,random_state=0)

"""## Build a Linear Regression model"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


rsquared = regressor.score(X_test, y_test)
rsquared

### Y = a + b X
#Get the constant or intercept  value of the line

intercept = regressor.intercept_
intercept

### Get the slope or coefficient of the line##

coefficient = regressor.coef_
coefficient

### Y = a + b X
### Mark = 0.531  + 9.93 Hours
## Predict for the test set


####Predict using the Regressor Model


y_new_pred_1 = regressor.predict([[3]])
print(y_new_pred_1)

####Predict using the formula

y_new_pred_2 = coefficient * 3 + intercept
print(y_new_pred_1)