from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


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


########################################## PLOT FUNCTIONS ##################################
def plot_top10_articles(data):
    dict_articles = get_most_visited_articles(data)
    top_10 = sorted(dict_articles.items(), key=lambda x: x[1], reverse=True)[:10]
    articles, visits = zip(*top_10)

    categories = [data.get_article_category(article,"1st cat") for article in articles]
    unique_categories = list(set(categories))

    color_list = sns.color_palette("Set2", len(unique_categories))
    dict_cat_color = {unique_categories[i]: color_list[i] for i in range(len(unique_categories))}

    # Assign a color to each article
    colors = [dict_cat_color[data.categories[data.categories["article_name"] == article]["1st cat"].values[0]] 
              for article in articles]
    
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=articles, y=visits, hue= articles , palette=colors)
    plt.xticks(rotation=90)
    plt.title(f"Top 10 most visited articles")
    plt.xlabel("Articles")
    plt.ylabel("Number of visits during games")

    handles = [plt.Rectangle((0,0),1,1, color=dict_cat_color[cat]) for cat in unique_categories]
    plt.legend(handles, unique_categories, title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_top10_links(data):
    dict_links = get_paths_oflength_k(data, 2)
    top_10 = sorted(dict_links.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10 = {f"{k[0]} -> {k[1]}": v for k, v in top_10}
    articles, visits = zip(*top_10.items())
    
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=visits, y=articles,orient='h')
    plt.title(f"Top 10 most visited links")
    plt.ylabel("Links")
    plt.xlabel("Number of visits during games")

    plt.tight_layout()
    plt.show()

def fig_top_cat(data):
    dict_categories = get_most_visited_categories(data, "1st cat")
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="coolwarm_r",  
        max_words=200,
        contour_color='black',  
        contour_width=2,  
    ).generate_from_frequencies(dict_categories)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Category Importance", fontsize=12)

    plt.show()