import json
import requests

url = 'http://127.0.0.1:1244/invocations'
headers = {'Content-Type' : 'application/json'}

request_data = json.dumps({"columns":["age", "salary"],"data":[[42, 50000]]})
response = requests.post(url,request_data,headers=headers)
print (response.text)



