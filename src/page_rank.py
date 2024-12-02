import numpy as np
import os
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import unquote


def get_article_as_number(data):
    #Create a number for each article corresponding to their index
    unique_values = pd.concat([data.links['1st article'], data.links['2nd article']]).unique()
    article_as_number = {value: idx for idx, value in enumerate(unique_values)}

    return article_as_number

def read_file_names(folder_path):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.htm'):
                # Get the full path or relative path if needed
                file_paths.append(os.path.join(root, file))
    return file_paths


def calculate_matrix_for_pagerank(data):

    number_of_articles = len(data.articles)

    links_matrix = np.zeros((number_of_articles,number_of_articles))

    article_as_number = get_article_as_number(data)

    #Iterate through all articles, set to 1 if link exsists
    for _, row in data.links.iterrows():
        i = article_as_number[row['1st article']]
        j = article_as_number[row['2nd article']]
        links_matrix[i, j] = 1

    return links_matrix

def calculate_matrix_for_pagerank_with_weights(data, folder_path):
    # List to store link data from all files
    all_link_counts_list = []

    # Get list of file paths
    file_paths = read_file_names(folder_path)

    for file_path in file_paths:
        file_name = str(os.path.basename(file_path))
        file_name = file_name.strip()
        first_article = unquote(file_name.replace('.htm', '').replace('_', ' '))

        print(first_article)

        with open(file_path, "r", encoding="utf-8", errors='replace') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, "html.parser")
        
        links = [unquote(a['href'].replace('../../wp/', '').replace('.htm', '').replace('_', ' ')[2:])
            for a in soup.find_all('a', href=True) 
            if '../../wp/' in a['href']]

        link_counts = pd.DataFrame(links, columns=["2nd article"]).value_counts().reset_index(name="count")
        
        link_counts["1st article"] = first_article

        all_link_counts_list.append(link_counts)

    all_link_counts = pd.concat(all_link_counts_list, ignore_index=True)

    data.links = pd.merge(
        data.links, all_link_counts, 
        on=["1st article", "2nd article"], 
        how="left"
    )

    return data.links, all_link_counts
