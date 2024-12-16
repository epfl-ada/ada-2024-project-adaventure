from tqdm import tqdm

def get_paths_oflength_k(data, k):
    """
    Get all paths of length k played by the users
    Input:
    - data: data object
    - k: length of the paths
    Output:
    - dict: dictionary with keys the paths and values the number of times the path has been taken by users
    """
    dict ={}
    # Go through all unfinished paths
    for path in tqdm(data.paths_unfinished.path, desc="Processing unfinished paths"):
        if len(path)<k: # If the path is shorter than k there is no subpath of length k
            continue
        for i in range(len(path)-k): # For each subpath of length k
            path_k = tuple(path[i:i + k]) # Get the subpath
            if path_k in dict:
                dict[path_k] += 1 # Increment the number of times the subpath has been taken
            else:
                dict[path_k] = 1 # Add the subpath to the dictionary
    
    # Go through all finished paths
    for path in tqdm(data.paths_finished.path, desc="Processing finished paths"):
        if len(path)<k: # If the path is shorter than k there is no subpath of length k
            continue
        for i in range(len(path)-k): # For each subpath of length k
            path_k = tuple(path[i:i + k]) # Get the subpath
            if path_k in dict:
                dict[path_k] += 1 # Increment the number of times the subpath has been taken
            else:
                dict[path_k] = 1 # Add the subpath to the dictionary
    return dict

def get_paths_oflength_k_ingame(data, k,start,end):
    """
    Get all paths of length k played by the users in a game going from start to end
    Input:
    - data: data object
    - k: length of the paths
    - start: start article
    - end: end article
    Output:
    - dict: dictionary with keys the paths and values the number of times the path has been taken by users
    """
    dict ={}
    paths = data.paths_finished[(data.paths_finished.start == start) & (data.paths_finished.end == end)] # Get all paths from start to end
    
    # Go through all paths
    for path in tqdm(paths.path, desc="Processing finished paths"):
        if len(path)<k: # If the path is shorter than k there is no subpath of length k
            continue
        for i in range(len(path)-k): # For each subpath of length k
            path_k = tuple(path[i:i + k]) # Get the subpath
            if path_k in dict:
                dict[path_k] += 1 # Increment the number of times the subpath has been taken
            else:
                dict[path_k] = 1 # Add the subpath to the dictionary
    return dict

def get_most_visited_articles(data):
    """
    Get the number of times each article has been visited by users
    Input:
    - data: data object
    Output:
    - dict: dictionary with keys the articles and values the number of times the article has been visited by users
    """
    dict ={}
    # Go through all unfinished paths
    for path in tqdm(data.paths_unfinished.path, desc="Processing unfinished paths"):
        for article in path:
            if article in dict:
                dict[article] += 1 # Increment the number of times the article has been visited
            else:
                dict[article] = 1 # Add the article to the dictionary

    # Go through all finished paths
    for path in tqdm(data.paths_finished.path, desc="Processing finished paths"):
        for article in path:
            if article in dict:
                dict[article] += 1 # Increment the number of times the article has been visited
            else:
                dict[article] = 1   # Add the article to the dictionary
    return dict

def get_most_visited_articles_ingame(data,start,end):
    """
    Get the number of times each article has been visited by users in a game going from start to end
    Input:
    - data: data object
    - start: start article
    - end: end article
    Output:
    - dict: dictionary with keys the articles and values the number of times the article has been visited by users
    """
    dict ={}
    paths = data.paths_finished[(data.paths_finished.start == start) & (data.paths_finished.end == end)] # Get all paths from start to end
    
    # Go through all paths
    for path in tqdm(paths.path, desc="Processing finished paths"):
        for article in path:
            if article in dict:
                dict[article] += 1 # Increment the number of times the article has been visited
            else:
                dict[article] = 1 # Add the article to the dictionary
    return dict

def get_most_visited_categories(data,cat_type = "1st cat"):
    """
    Get the number of times each category has been visited by users, normalized by the number of articles in the category
    Input:
    - data: data object
    - cat_type: type of category to consider (1st cat or 2nd cat)
    Output:
    - dict: dictionary with keys the categories and values the number of times the category has been visited by users normalized by the number of articles in the category
    """
    dict ={}

    # Go through all unfinished paths
    for path in tqdm(data.paths_unfinished.path, desc="Processing unfinished paths"):
        for article in path: # For each article in the path
            if article in data.categories["article_name"].values:   # If the article has a category
                cat = data.get_article_category(article,cat_type)
                if cat != None:
                    if cat in dict:
                        dict[cat] += 1 # Increment the number of times the category has been visited
                    else:
                        dict[cat] = 1 # Add the category to the dictionary
    
    # Go through all finished paths
    for path in tqdm(data.paths_finished.path, desc="Processing finished paths"):
        for article in path: # For each article in the path
            if article in data.categories["article_name"].values:  # If the article has a category
                cat = data.get_article_category(article,cat_type)
                if cat != None:
                    if cat in dict:
                        dict[cat] += 1 # Increment the number of times the category has been visited
                    else:
                        dict[cat] = 1 # Add the category to the dictionary

    # Normalize by the number of articles in the category
    for cat in dict.keys():
        nb_articles = len(data.categories["1st cat"]== cat)
        dict[cat] = dict[cat]/nb_articles
    return dict