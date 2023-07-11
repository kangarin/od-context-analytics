# send a post request to localhost:5000/test with json data
# {
#     "name": "John",
#     "age": 30
# }
#

import requests
import json
requests.post('http://localhost:5000/test', json={"name": "John", "age": 30})