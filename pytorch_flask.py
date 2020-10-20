from flask import Flask, request
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import pandas as pd
import numpy as np

input_size=2
output_size=2
hidden_size=10

local_scaler = pickle.load(open('sc.pickle','rb'))

class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.fc1 = torch.nn.Linear(input_size, hidden_size)
       self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
       self.fc3 = torch.nn.Linear(hidden_size, output_size)


   def forward(self, X):
       X = torch.sigmoid((self.fc1(X)))
       X = torch.sigmoid((self.fc2(X)))
       X = self.fc3(X)

       return F.log_softmax(X,dim=1)

new_predictor2 = Net()


app = Flask(__name__)

@app.route('/model',methods=['POST'])
def hello_world():
    request_data = request.get_json(force=True)
    age = request_data['age']
    salary = request_data['salary']
    print(age)
    print(salary)
    
    prediction = new_predictor2(torch.from_numpy(local_scaler.transform(np.array([[age,salary]]))).float())[:,0]


    return "The prediction from GCP API is {}".format(prediction)

if __name__ == "__main__":
    app.run(host= '0.0.0.0',port=8005, debug=True)
    
	
	