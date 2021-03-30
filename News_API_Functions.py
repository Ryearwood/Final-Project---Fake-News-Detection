# Import required Libraries
import pandas as pd
import os
from newsapi import NewsApiClient
from pandas import json_normalize
import json
import requests
from IPython.display import JSON

# API GET Call Function
def fetch_query(newsapi,user_des_source):
    headlines = newsapi.get_top_headlines(language='en',
                                          sources=user_des_source,
                                          page_size=100)
    return headlines

def create_data(json_data):
    # Create Datatable to capture information
    API_data = pd.DataFrame(columns = ['title', 'author', 'source'])

    # Loop to capture information from API results
    for index in range(len(json_data['articles'])):
            # Store Data in respective columns, row by row
            API_data = API_data.append({'title': json_data['articles'][index]['title'],
                                        'author':json_data['articles'][index]['author'],
                                        'source':json_data['articles'][index]['source']['name']},
                                        ignore_index=True)
            API_data = API_data.fillna('')
    return API_data