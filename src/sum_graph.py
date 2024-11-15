import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from urllib.parse import unquote
import numpy as np
import os
import pprint
import random
import math
import seaborn as sns

import networkx as nx

def get_grouped_category_data(data, includeGeography=True):

    # Calculates the amount of articles in all cateogies
    node_nr = data.categories['1st cat'].value_counts()

    # Adds the categories of Source and Target to the links data via merging twice 
    links_nr = data.links.merge(right=data.categories[['article_name', '1st cat']], 
                                        left_on='1st article', 
                                        right_on = 'article_name', 
                                        how='left')

    links_nr = links_nr.merge(right=data.categories[['article_name', '1st cat']], 
                            left_on='2nd article', 
                            right_on = 'article_name', 
                            how='left')

    # Drops duplicate rows
    links_nr = links_nr.drop_duplicates(ignore_index=True).rename(columns={'1st cat_x': "Category 1st article",
                                                                        '1st cat_y': "Category 2nd article"})

    # Removes articles that does not have a category
    links_nr = links_nr.dropna()

    # Gives the amount of connections between all categories
    links_nr = links_nr.value_counts(subset=['Category 1st article', 'Category 2nd article']).reset_index(name='edge_weight')
    links_nr = links_nr.sort_values(by=['Category 1st article', 'Category 2nd article'], ascending=False)
        
    # Calculates the amount of articles there are in each category
    category_freq = data.categories['1st cat'].value_counts().reset_index(name="node_size")

    # Add the amount of articles in source-cateogry
    links_nr = links_nr.merge(right=category_freq, 
                            left_on='Category 1st article', 
                            right_on = '1st cat', 
                            how='left')

    # Only keeps links between different categories
    links_nr = links_nr[links_nr['Category 1st article'] != links_nr['Category 2nd article']].reset_index()

    if includeGeography is False:   
        links_nr = links_nr[links_nr['Category 1st article'] != 'Geography'].reset_index()

    return links_nr

def create_sum_graph(data, links_nr, title='Links between categories'):

        # Creates a graph between categories
        G = nx.from_pandas_edgelist(links_nr, 
                                source='Category 1st article',
                                target='Category 2nd article',
                                create_using=nx.DiGraph())

        # Calculates the amount of articles there are in each category
        category_freq = data.categories['1st cat'].value_counts().reset_index(name="node_size")

        # Fetches the node sizes from the dataframe category_freq
        node_sizes = [category_freq[category_freq['1st cat'] == n]['node_size'].values[0] for n in G.nodes()]

        # Sets the weights for each nope (amount for links divided by number of articles in source category)
        weights = links_nr['edge_weight'] / links_nr['node_size']

        # MinMax-normalises the node-sizes and edge-widths
        node_sizes_norm = (node_sizes - min(node_sizes)) / (max(node_sizes) - min(node_sizes))*2000
        edge_widht = (weights - weights.min()) / (weights.max() - weights.min())*10

        # Draws the network
        fig = plt.figure(figsize=(9, 5))
        pos = nx.circular_layout(G)
        nx.draw(G, 
                pos, 
                with_labels=True, 
                width=edge_widht, 
                node_size=node_sizes_norm, 
                edge_color='skyblue',
                font_size = 6,
                font_weight='bold', 
                node_color=sns.color_palette(n_colors=15))
        # fig.set_facecolor("skyblue")
        plt.title(title)
        plt.show()


def get_clustered_graph(data):

    # Creates gtaph out of all links
    G = nx.from_pandas_edgelist(data.links, source='1st article', target='2nd article')
    
    # Create a colormap for the nodes
    color_map = {category: color for category, color in zip(data.categories['1st cat'].unique(), plt.cm.tab20.colors)}

    # Iterates over all node and assign color
    for node in G.nodes:

        # Fetches the nodes category
        result = data.categories[data.categories['article_name'] == node]['1st cat']

        # Assigns color to node based on category
        if not result.empty:
            category = result.values[0]  # Get the first category value
            G.nodes[node]['color'] = color_map.get(category, "grey")
            G.nodes[node]['category'] = category
        else:   
            G.nodes[node]['color'] = "lightgrey"
            G.nodes[node]['category'] = "unknown"


    # Cluster nodes based on category
    clusters = {}
    for node, attr in G.nodes(data=True):
        category = attr['category']
        if category not in clusters:
            clusters[category] = []
        clusters[category].append(node)


    positions = {}
    r = 10 

    # Places all clusters in a circle
    for index, (category, nodes) in enumerate(clusters.items()):
        # Creates a subgraph for each category and positions them using spring_layput
        subG = G.subgraph(nodes)
        sub_positions = nx.spring_layout(subG, seed=1, k=0.1)

        # Displaces all nodes in subG on a circle around the origin
        angle = (2 * math.pi * index) / len(clusters)
        x = r * math.cos(angle)
        y = r * math.sin(angle)

        for node in sub_positions:
            positions[node] = [sub_positions[node][0] + x, sub_positions[node][1] + y]
        
    # Draws a plot clustered by category
    plt.figure(figsize=(10, 8))

    for category, nodes in clusters.items():
        nx.draw_networkx_nodes(G, pos=positions, 
                            node_size=10, 
                            nodelist=nodes, 
                            node_color=G.nodes[nodes[0]]['color'], 
                            label=category)
        
    nx.draw_networkx_edges(G, positions, alpha=0.2, width=0.01)
    plt.legend()
    plt.show()
