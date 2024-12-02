import numpy as np
import pandas as pd

def calculate_matrix_for_pagerank(data):

    number_of_articles = len(data.articles)

    links_matrix = np.zeros((number_of_articles,number_of_articles))

    #Create a number for each article corresponding to their index
    unique_values = pd.concat([data.links['1st article'], data.links['2nd article']]).unique()
    article_as_number = {value: idx for idx, value in enumerate(unique_values)}

    #Iterate through all articles, set to 1 if link exsists
    for _, row in data.links.iterrows():
        i = article_as_number[row['1st article']]
        j = article_as_number[row['2nd article']]
        links_matrix[i, j] = 1

    return links_matrix
