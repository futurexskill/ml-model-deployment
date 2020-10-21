import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

if __name__ == "__main__":

    mlflow.set_experiment(experiment_name="mlflow demo")
    
    training_data = pd.read_csv('storepurchasedata.csv')
    print("loaded training data")

    training_data.describe()
    
    X = training_data.iloc[:, :-1].values
    y = training_data.iloc[:,-1].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.70,random_state=0)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    """## Build a Classification model
    ### We are using KNN Classifier in this example
    *n_neighbors = 5 -* Number of neighbors
    *metric = 'minkowski', p = 2* - For Eucledian distance calculation
    """
    print("Completed Feature Scaling")
    
    from sklearn.neighbors import KNeighborsClassifier
    # minkowski is for ecledian distance
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    
    # Model training
    classifier.fit(X_train, y_train)
    print("Model trained")
   
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:,1]
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    from sklearn.metrics import accuracy_score
    
    model_accuracy = accuracy_score(y_test,y_pred)
    
    print(model_accuracy)
    
    mlflow.log_metric("accuracy", model_accuracy)
    mlflow.sklearn.log_model(classifier, "model")
    
    