import numpy as np
import pandas as pd

training_data = pd.read_csv('storepurchasedata.csv')

training_data.describe()

X = training_data.iloc[:, :-1].values
y = training_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""## Build a Classification model
### We are using KNN Classifier in this example
*n_neighbors = 5 -* Number of neighbors
*metric = 'minkowski', p = 2* - For Eucledian distance calculation
"""

from sklearn.neighbors import KNeighborsClassifier
# minkowski is for ecledian distance
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

# Model training
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:,1]

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))


from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))


new_prediction = classifier.predict(sc.transform(np.array([[40,20000]])))

new_prediction_proba = classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]

new_pred = classifier.predict(sc.transform(np.array([[42,50000]])))

new_pred_proba = classifier.predict_proba(sc.transform(np.array([[42,50000]])))[:,1]



# Picking the Model and Standard Scaler

import pickle

model_file = "classifier.pickle"

pickle.dump(classifier, open(model_file,'wb'))

scaler_file = "sc.pickle"

pickle.dump(sc, open(scaler_file,'wb'))
















