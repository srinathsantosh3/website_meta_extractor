# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 20:06:59 2021

@author: jnadar4
"""

import requests

# api-endpoint
URL = "http://localhost:5000/get_details"

# location given here
scrape_url  = 'https://www.uhcglobalclinicaljournal.com'

# defining a params dict for the parameters to be sent to the API
PARAMS = {'url':scrape_url}

# sending get request and saving the response as response object
r = requests.post(url = URL, json = PARAMS)
data = r.json()
# extracting data in json format
# =============================================================================
print(data)
