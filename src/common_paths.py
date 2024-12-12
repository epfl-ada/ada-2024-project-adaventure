from tqdm import tqdm

def get_paths_oflength_k(data, k):
    dict ={}
    for path in tqdm(data.paths_unfinished.path, desc="Processing unfinished paths"):
        if len(path)<k:
            continue
    # Utiliser tqdm pour afficher la progression de la boucle interne
        for i in range(len(path)-k):
            path_k = path[i:i+k]
            path_k = tuple(path[i:i + k])
            if path_k in dict:
                dict[path_k] += 1
            else:
                dict[path_k] = 1
    
    for path in tqdm(data.paths_finished.path, desc="Processing finished paths"):
    # Utiliser tqdm pour afficher la progression de la boucle interne
        for i in range(len(path)-k):
            path_k = path[i:i+k]
            path_k = tuple(path[i:i + k])
            if path_k in dict:
                dict[path_k] += 1
            else:
                dict[path_k] = 1
    return dict

def get_paths_oflength_k_ingame(data, k,start,end):
    dict ={}
    paths = data.paths_finished[(data.paths_finished.start == start) & (data.paths_finished.end == end)]
    
    for path in tqdm(paths.path, desc="Processing finished paths"):
    # Utiliser tqdm pour afficher la progression de la boucle interne
        for i in range(len(path)-k):
            path_k = path[i:i+k]
            path_k = tuple(path[i:i + k])
            if path_k in dict:
                dict[path_k] += 1
            else:
                dict[path_k] = 1
    return dict

def get_most_visited_articles(data):
    dict ={}
    for path in tqdm(data.paths_unfinished.path, desc="Processing unfinished paths"):
        for article in path:
            if article in dict:
                dict[article] += 1
            else:
                dict[article] = 1
    for path in tqdm(data.paths_finished.path, desc="Processing finished paths"):
        for article in path:
            if article in dict:
                dict[article] += 1
            else:
                dict[article] = 1
    return dict

def get_most_visited_articles_ingame(data,start,end):
    dict ={}
    paths = data.paths_finished[(data.paths_finished.start == start) & (data.paths_finished.end == end)]
    
    for path in tqdm(paths.path, desc="Processing finished paths"):
        for article in path:
            if article in dict:
                dict[article] += 1
            else:
                dict[article] = 1
    return dict

def get_most_visited_categories(data,cat_type = "1st cat"):
    dict ={}
    for path in tqdm(data.paths_unfinished.path, desc="Processing unfinished paths"):
        for article in path:
            if article in data.categories["article_name"].values:
                cat = data.categories[data.categories["article_name"] == article][cat_type].values[0]
                if cat != None:
                    if cat in dict:
                        dict[cat] += 1
                    else:
                        dict[cat] = 1
    for path in tqdm(data.paths_finished.path, desc="Processing finished paths"):
        for article in path:
            if article in data.categories["article_name"].values:
                cat = data.categories[data.categories["article_name"] == article][cat_type].values[0]
                if cat != None:
                    if cat in dict:
                        dict[cat] += 1
                    else:
                        dict[cat] = 1
    return dict