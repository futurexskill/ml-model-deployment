from flask import Flask, request
import pickle

import numpy as np
local_classifier = pickle.load(open('classifier.pickle','rb'))
local_scaler = pickle.load(open('sc.pickle','rb'))

app = Flask(__name__)

@app.route('/model',methods=['POST'])
def hello_world():
    request_data = request.get_json(force=True)
    age = request_data['age']
    salary = request_data['salary']
    print(age)
    print(salary)
    pred_proba = local_classifier.predict_proba(local_scaler.transform(np.array([[age,salary]])))[:,1]
    return "The prediction from GCP API is {}".format(pred_proba)

if __name__ == "__main__":
    app.run(host= '0.0.0.0',port=8005, debug=True)
    
    
    
