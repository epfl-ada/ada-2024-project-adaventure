import numpy as np
import pandas as pd


def get_hubs_dataframe(data):

    df_hubs = pd.DataFrame()
    df_hubs["article_names"] = data.articles["article_name"]

    #Printing wikipages with most links
    source_counts = data.links['1st article'].value_counts()
    df_hubs["source_counts"] = df_hubs["article_names"].map(source_counts).fillna(0).astype(int)

    target_counts = data.links['2nd article'].value_counts()
    df_hubs["target_counts"] = df_hubs["article_names"].map(target_counts).fillna(0).astype(int)

    df_hubs["mean_shortest_path_to_article"] = np.nanmean(data.matrix, axis = 0)
    df_hubs["mean_shortest_path"] = np.nanmean(data.matrix, axis = 1)

    #Adding categories to hubs dataframe
    categories = np.unique(data.categories['1st cat'])
    # Add a column for each category in categories_ha
    for category in categories:
        articles_in_category = data.categories.loc[data.categories['1st cat'] == category, 'article_name']
        df_hubs[category] = np.where(df_hubs['article_names'].isin(articles_in_category), 1, 0)

    return df_hubs