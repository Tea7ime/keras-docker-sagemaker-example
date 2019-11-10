# You can execute this script against a launched endpoint. If you need to test against, cloud try postman instead.

import numpy as np
import requests
import json

def post_data(url, json):
    print('Publishing POST message to %s' % url)
    headers = {'Content-Type': 'application/json'}
    try:
        resp = requests.post(url, data=json, headers=headers)
        print(f"Request Body: {resp.request.body}")
        print(f" Response Message: ... Code: {resp.status_code} {resp.content.decode('utf8')}")
    except requests.RequestException as e:
        print('RequestException Error: %s' % (e))
    else:
        print('Response from POST: %s' % resp)


dataset_path = './data/pima-indians-diabetes.csv'
dataset = np.loadtxt(dataset_path, delimiter=",", skiprows=1)

# We're simply matching the batch size and num columns that were used in training here...
X = dataset[-11:-1,:8]

data = json.dumps(np.asarray(X).astype(float).tolist())

# modify this if you are sending to cloud. You'll also need to sign AWS credentials with access keys
response = post_data('http://localhost:8080/invocations', data)

