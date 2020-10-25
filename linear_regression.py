import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('houseprice.csv')

dataset.describe()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#plt.xlabel('Distance to City center')
#plt.ylabel('House price USD')
#plt.title('House price analysis')
#plt.scatter(X, y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2,random_state=48)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

rsquared = regressor.score(X_test, y_test)
rsquared

intercept = regressor.intercept_
intercept

coefficient = regressor.coef_
coefficient

### Y = a + b X
### Price = 610710.0319872361 - 72635.89282856 Distance

y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_test, y_pred, color = 'green')
plt.title('House price analysis')
plt.xlabel('Distance to City center')
plt.ylabel('House price USD')
plt.show()


y_new_pred_1 = regressor.predict([[2.5]])
print(y_new_pred_1)


y_new_pred_2 = coefficient * 2.5 + intercept
print(y_new_pred_1)




