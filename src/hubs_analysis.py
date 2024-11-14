import numpy as np
import pandas as pd


def get_hubs_dataframe(data):
    '''
    This function takes in a class of dataframes and combines it into one dataframe for further analysis
    The information extracted are the article name, the number of links to and from a wikipedia article, the categories
    that the article is classified as belonging to and calculates the mean shortest path to and from the article.
    '''

    df = pd.DataFrame()
    df["article_names"] = data.articles["article_name"]

    #Printing wikipages with most links
    source_counts = data.links['1st article'].value_counts()
    df["source_counts"] = df["article_names"].map(source_counts).fillna(0).astype(int)

    target_counts = data.links['2nd article'].value_counts()
    df["target_counts"] = df["article_names"].map(target_counts).fillna(0).astype(int)

    df["mean_shortest_path_to_article"] = np.nanmean(data.matrix, axis = 0)
    df["mean_shortest_path"] = np.nanmean(data.matrix, axis = 1)

    #Adding categories to hubs dataframe
    categories = np.unique(data.categories['1st cat'])
    # Add a column for each category in categories_ha
    for category in categories:
        articles_in_category = data.categories.loc[data.categories['1st cat'] == category, 'article_name']
        df[category] = np.where(df['article_names'].isin(articles_in_category), 1, 0)

    return df