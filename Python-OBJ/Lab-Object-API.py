import os
import json
import base64
from nvision import ObjectDetection

model = ObjectDetection(api_key='cdb29f355cb4059995e05420dc89c26f637ec8ec3a5f2b5c7a88c5d526fae4fbf6c094cf861494ee2b466fca51475579ac')

# base64 encoed string
with open('lfc.jpeg', 'rb') as file:
    image = file.read()
    image = base64.b64encode(image).decode('utf-8')

# make a RESTful call to the Nvision API
response = model.predict(image)

# get the predictions (in JSON format) from the response
print(json.dumps(response.json(), indent=4, sort_keys=True))
