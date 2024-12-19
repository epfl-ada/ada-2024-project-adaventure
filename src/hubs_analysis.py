import numpy as np
import pandas as pd

from scipy.stats import pearsonr
import networkx as nx

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


def calculate_correlation(df, cols):
    '''
    This function calculates correlation between features in data frame

    Args:
    df: dataframe containing features to calculate correlation on
    cols: columns to calculate correlation

    Return:
    p_values: array containing the p-values for each pair of columns.
    correlation_matrix: array containing the correlation coefficients for each pair of columns.

    '''
    size = len(cols)
    p_values = np.zeros((size,size))
    correlation_matrix = np.zeros((size,size))

    # Calculate p-values for each pair of columns
    for i,i_col_name in enumerate(cols):
        for j,j_col_name in enumerate(cols):
            if i_col_name == j_col_name:
                p_values[i,j] = 0  # p-value for correlation with itself
                correlation_matrix[i,j]= 1
            else:
                corr, p_val = pearsonr(df[i_col_name], df[j_col_name])
                p_values[i,j] = p_val
                correlation_matrix[i,j] = corr

    return p_values,correlation_matrix


def get_top_10_largest(df, column, print_statement, n = 10):
    '''
    Function to print the n largest values of a column and returns set of the articles
    '''
    top_n = df.nlargest(n, column)[['article_names', column]]
    print(print_statement)
    print(top_n)

    return set(top_n['article_names'])


def get_top_10_smallest(df, column, print_statement, n = 10):
    '''
    Function to print the n smallest values of a column and returns set of the articles
    '''
    top_n = df.nsmallest(n, column)[['article_names', column]]
    print(print_statement)
    print(top_n)

    return set(top_n['article_names'])


def calculate_pagerank(data, df_hubs):
    """
    Calculate standard PageRank scores and add them to the df_hubs DataFrame.

    Args:
    - data: A data object containing 'links' with '1st article' and '2nd article'
    - df_hubs: A pandas DF with 'article_names' column

    Returns:
    - Updated df_hubs DataFrame with 'pagerank_score' column.
    """
    # Create graph
    G = nx.DiGraph()

    # Add edges from the links data
    edges = list(zip(data.links['1st article'], data.links['2nd article']))
    G.add_edges_from(edges)

    # Calculate PageRank
    pagerank_scores = nx.pagerank(G)

    # Add PageRank scores to df_hubs
    df_hubs["pagerank_score"] = df_hubs["article_names"].map(pagerank_scores).fillna(0)

    return df_hubs

def calculate_outgoing_pagerank(data, df_hubs):
    """
    Calculate custom PageRank scores biased toward hubs and add them to the df_hubs DataFrame.

    Args:
    - data: A data object containing 'links' with '1st article' and '2nd article'
    - df_hubs: A pandas DF with 'article_names' column

    Returns:
    - Updated df_hubs DF with 'pagerank_outgoing_score' column
    """
    # Create graph
    G = nx.DiGraph()

    # Add edges from the 'links' data
    edges = list(zip(data.links['1st article'], data.links['2nd article']))
    G.add_edges_from(edges)

    # Add weights based on out-degree
    for node in G.nodes:
        G.nodes[node]['out_degree'] = G.out_degree(node)

    # Custom PageRank calculation, biased toward hubs
    personalization = {node: G.nodes[node]['out_degree'] for node in G.nodes}
    pagerank_scores = nx.pagerank(G, personalization=personalization)

    # Add the scores back to df_hubs
    df_hubs["pagerank_outgoing_score"] = df_hubs["article_names"].map(pagerank_scores).fillna(0)

    return df_hubs, G

def create_hub_score(df_hubs):
    """
    Calculate average of the other two hub scores to balance incoming and outgoing links

    Args: 
    - df_hubs: A pandas DF with 'pagerank_score' and 'pagerank_outgoing_score'

    Returns:
    - df_hubs with 'hub_score' column
     - df_filtered_hubs: Filtered DataFrame where 'target_counts' > 0.
    """
    df_hubs['hub_score'] = df_hubs.apply(
    lambda row: (row['pagerank_score'] + row['pagerank_outgoing_score']) / 2
    if row['target_counts'] > 0 else None,
    axis=1)

    df_filtered_hubs = df_hubs[df_hubs['target_counts'] > 0].copy()

    return df_hubs, df_filtered_hubs