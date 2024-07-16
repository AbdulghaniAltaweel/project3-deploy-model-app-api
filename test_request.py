import json
import requests

data = {
    "age": 30,
    "workclass": "State-gov",
    "fnlgt": 141297,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "Asian-Pac-Islander",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "India"
}

response = requests.post('http://project3-deploy-model-app-api.onrender.com/', data=json.dumps(data))

# Actually the live testing using render get an error when printing print(response.json()).
# Thats why tried to figure out by switching to localhost api to test post request.
# With localhost api getting the correct results. 
# The rrror msg was:
# raise RequestsJSONDecodeError(e.msg, e.doc, e.pos)
# E   requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
# This error in render.com was the python version. 
# After setting python_version to 3.8.19 in env_variable is worked well.
# response = requests.post('http://localhost:8000/', data=json.dumps(data))

print(response.status_code)
print(response.json())