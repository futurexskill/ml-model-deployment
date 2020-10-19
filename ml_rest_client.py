import json
import requests

url = 'http://34.68.168.133:8005/model'

request_data = json.dumps({'age':40,'salary':20000})
response = requests.post(url,request_data)
print (response.text)



