import numpy as np
import os
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import unquote

#This file is not in use in the project atm
#This code extracts the number of times an article is linked to another


def read_file_names(folder_path):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.htm'):
                # Get the full path or relative path if needed
                file_paths.append(os.path.join(root, file))

            else: 
                continue
   
    return file_paths

def get_number_of_links(data, folder_path):
    # List to store link data from all files
    all_link_counts_list = []

    # Get list of file paths
    file_paths = read_file_names(folder_path)

    for file_path in file_paths:
        file_name = str(os.path.basename(file_path))
        file_name = file_name
        first_article = unquote(str(file_name)).replace('.htm', '').replace('_', ' ').replace(',', '')

        with open(file_path, "r", encoding="utf-8", errors='replace') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, "html.parser")
        
        links = [a['href'].replace('../../wp/', '').replace('.htm', '')[2:]
            for a in soup.find_all('a', href=True) if a['href'].startswith('../../wp/')]

        links = [unquote(unquote(str(link))).replace('_', ' ').replace(',', '') for link in links]

        link_counts = pd.DataFrame(links, columns=["2nd article"]).value_counts().reset_index(name="count")
        
        link_counts["1st article"] = first_article

        all_link_counts_list.append(link_counts)

    all_link_counts = pd.concat(all_link_counts_list, ignore_index=True)

    new_data = pd.merge(
        data.links, all_link_counts, 
        on=["1st article", "2nd article"], 
        how="left"
    )

    return new_data, all_link_counts
