import pandas as pd
import os
import ast
import networkx as nx


def calculate_all_shortest_paths(data):

    # Creates df to store all paths taken in game
    shortest_paths = pd.DataFrame(index= range(data.paths_finished.shape[0] + data.paths_unfinished.shape[0]), 
                                columns=['source', 'target'])

    # Adds source and target for all finished paths
    for index, row in data.paths_finished.iterrows():
        source = row['path'][0]
        target = row['path'][-1]
        shortest_paths.at[index, 'source'] =  source
        shortest_paths.at[index, 'target'] = target

    #Adds source and target for all unfinished paths
    for index, row in data.paths_unfinished.iterrows():
        index = index + data.paths_finished.shape[0]
        source = row['path'][0]
        target = row['target']
        shortest_paths.at[index, 'source'] =  source
        shortest_paths.at[index, 'target'] = target

    # Remove duplicate rows
    shortest_paths = shortest_paths.drop_duplicates(ignore_index=True)

    # Path to shortest_path.csv
    csv_path = "./data/wikispeedia_paths-and-graph/shortest_paths.csv"

    # If shortest_path.csv not already saved down in directory, calculates it and saves it as csv
    if not os.path.exists(csv_path):

        # Creates directed graph between all articles 
        G = nx.from_pandas_edgelist(data.links, 
                                    source='1st article', 
                                    target='2nd article',
                                    create_using=nx.DiGraph())

        # Adds new column to shortest_paths, to keep all paths
        shortest_paths['shortest_paths'] = None

        # Initiate list indecies causing bugs
        bugIndecies = []

        # Calculates all possible shortest paths for all games and saves them in datarame "shortest_paths"
        for index, row in shortest_paths.iterrows():
            try:
                paths = list(nx.all_shortest_paths(G, source=row['source'], target=row['target']))
                # paths = nx.shortest_path(G, source=row['source'], target=row['target'])
                shortest_paths.at[index, 'shortest_paths'] = paths
            except nx.NetworkXNoPath:   # If no path exists
                bugIndecies.append(index)
            except nx.NodeNotFound:     # If node not found
                bugIndecies.append(index)

        # Removes all games causing bugs in code
        shortest_paths = shortest_paths.drop(index=bugIndecies)

        # Saves down results in a csv
        shortest_paths['shortest_paths'].to_csv(csv_path, index=False)

    return csv_path

# Most of the errors are due to misspellings of targits in data.paths_unfinished, but also some
# other errors, such as links that dont exist (but should) or NoPathsFound

def shortest_path_article_frequency(csv_path):

    # Reads in shortest_path.csv into dataframe
    shortest_paths  =  pd.read_csv(csv_path)

    # Converts each cell from string values to literal value (lists)
    shortest_paths = shortest_paths['shortest_paths'].apply(ast.literal_eval)

    # Explodes lists of paths into paths (one path per cell)
    shortest_paths = shortest_paths.explode(ignore_index=True)

    # Removes start and target articles from paths
    shortest_paths = shortest_paths.apply(lambda x: x[1:])
    shortest_paths = shortest_paths.apply(lambda x: x[:-1])

    # Explodes paths (lists) into articles (one article per cell)
    shortest_paths = shortest_paths.explode(ignore_index=True).to_frame()

    # Calculates the frequency of all articles in all paths
    shortest_paths_freq = shortest_paths.value_counts()

    # Return dataframe
    return shortest_paths_freq.to_frame().reset_index()