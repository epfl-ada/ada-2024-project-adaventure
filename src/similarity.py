
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import random
import pickle



## DIFFERENT FUNCTIONS TO COMPUTE THE DISTANCE BETWEEN TWO PATHS #########################

def distance_Jaccard(data,path1,path2):
    """
    Compute the Jaccard distance between two paths
    """
    inter = len(set(path1).intersection(set(path2)))
    union = len(set(path1).union(set(path2)))
    return  1-inter/union

def distance_matrix(data,path1,path2):
    """
    Compute the distance between two paths as the number of clicks to reach one path from the other
    """
    # Get the index of the articles in the paths
    path1 = [int(data.articles[data.articles["article_name"]== x].index[0]) for x in path1]
    path2 = [int(data.articles[data.articles["article_name"]== x].index[0]) for x in path2]

    n1 = len(path1)
    n2 = len(path2)
    D = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            # Compute the distance between the articles in the two paths
            D[i,j] = data.matrix[path1[i],path2[j]]
    min_d = np.min(D,axis=1)
    return (np.max(min_d)) # mean :  in average how many clicks to reach an article in the second path / max : the maximum number of clicks to reach an article in the second path

def embedding_Bert(paths):
    """
    Compute the embeddings of the paths using a pretrained BERT model
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    for path in paths:
        path = ' '.join(path)
        embeddings.append(model.encode(path))
    return embeddings
    

def distance_Bert(embeddings,i,j):
    """
    Compute the distance between two paths using the cosine similarity between their embeddings
    """
    embedding1 = embeddings[i]
    embedding2 = embeddings[j]
    return cosine(embedding1,embedding2)

##########################################################################################

def get_games(data):
    """
    Get the finished games played by the users
    """
    games = pd.DataFrame(columns=["start","end"])
    games[["start","end"]] = pd.DataFrame(data.paths_finished["path"].apply(lambda x: [x[0], x[-1]]).tolist(), index=data.paths_finished.index)
    games.drop_duplicates(inplace=True)

    # Get the number of games played from start to end
    data.paths_finished["start"] = data.paths_finished["path"].apply(lambda x: x[0])
    data.paths_finished["end"] = data.paths_finished["path"].apply(lambda x: x[-1])
    games["nb_games"] = 0
    for i in tqdm(range(len(data.paths_finished))):
        start = data.paths_finished.loc[i,"start"]
        end = data.paths_finished.loc[i,"end"]
        games.loc[(games["start"]==start) & (games["end"]==end), "nb_games"] += 1

    games = games.reset_index(drop=True)

    return games

def get_sim_matrix(data,start,end,distance,is_Bert =False):
    """
    Compute the similarity matrix for a game played from start to end
    """
    paths = data.paths_finished[(data.paths_finished["start"] == start) & (data.paths_finished["end"] == end)]
    matrix_distance = np.zeros((len(paths),len(paths)))
    if len(paths) == 0:
        raise ValueError("No game played from ",start," to ",end) # No game played from start to end
    if len(paths) == 1:
        print("Only one game played from ", start, " to ", end) # Only one game played from start to end (no similarity)
        return None
    if is_Bert:
        embeddings = embedding_Bert(paths["path"].values)
    for i in range(len(paths)):
        path1 = paths["path"].values[i]
        for j in range(len(paths)):
            path2 = paths["path"].values[j]
            if is_Bert:
                sim = distance(embeddings,i,j)
            else:
                sim = max(distance(data,path1,path2),distance(data,path2,path1)) # Compute the similarity between the two paths

            matrix_distance[i][j] = sim
    return matrix_distance

def get_mean_sim_distrib(data,games,distance = distance_Bert,load=True):
    """
    Computes the mean similarity between two players paths for every finished games. 
    """
    if load :
        with open('data/mean_sim_distrib.pkl', 'rb') as f:
            mean_sim_distrib = pickle.load(f)
    else:
        mean_sim_distrib = []
        for i in range(len(games)):
            start = games.loc[i,"start"]
            end = games.loc[i,"end"]
            matrix_sim = get_sim_matrix(data,start,end,distance)
            if matrix_sim is not None:
                mean = np.mean(matrix_sim)
                mean_sim_distrib.append(mean)
        with open('data/mean_sim_distrib.pkl', 'wb') as f:
            pickle.dump(mean_sim_distrib, f)
    return mean_sim_distrib

def get_mean_sim_random(data,n_pairs):
    mean =0
    for k in tqdm(range(n_pairs)):
        path1 = get_random_path(data)
        path2 = get_random_path(data)
        embeddings = embedding_Bert([path1,path2])
        d = distance_Bert(embeddings,0,1)
        mean += d
    return mean/n_pairs

def plot_sim_distrib(data,games,distance = distance_Bert,n_pairs_random=100):
    mean_sim_distrib = get_mean_sim_distrib(data,games,distance=distance)
    mean_random = get_mean_sim_random(data,n_pairs_random)
    mean_games = np.mean(mean_sim_distrib)

    plt.figure(figsize=(7, 6))  
    sns.histplot(mean_sim_distrib, kde=False, color="darkblue")

    plt.axvline(x=mean_random, linestyle="--", color="darkgreen", label="Mean distance for random paths")
    plt.axvline(x=mean_games, linestyle="--", color="red", label="Mean distance for all finished games")

    plt.title('Distribution of the mean semantic distance\nbetween paths for all finished games', fontsize=14)
    plt.xlabel('Mean distance', fontsize=12)
    plt.ylabel('Number of games', fontsize=12)
    plt.xlim(0, 1)  
    plt.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_sim_matrices(data,starts,ends,distance = distance_Jaccard,title = "Similarity matrix",is_Bert=False,**kwargs):
    """
    Plot the similarity matrices for the games played from starts to ends
    """
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        matrix_sim = get_sim_matrix(data,start,end,distance,is_Bert=is_Bert)
        if matrix_sim is not None:
            print("Game from ",start," to ",end)
            plt.figure(figsize=(7,5))
            sns.heatmap(matrix_sim, cmap="Blues",vmin =0, **kwargs)
            plt.title(title)
            plt.show()
            mean = np.mean(matrix_sim)
            print("Mean distance for game from ",start," to ",end, " : ",mean)

##########################################################################################

def following_article(article,paths):
    """
    Get the articles visited by different players after an article
    """
    list=[]
    for path in paths:
        if article in path:
            indexes = [i for i, x in enumerate(path) if x == article]
            for i in indexes:
                if i < len(path)-1:
                    list.append(path[i+1])
                elif len(path)==1:
                    list.append(article)
    dict ={x:list.count(x) for x in list} # Count the number of players who visited each article in dictionnary after the article
    return dict

def distance_first_article(data,start,end,distance=distance_matrix):
    """
    Compute the mean distance of first articles visited after start
    """
    paths = data.paths_finished[(data.paths_finished["start"] == start) & (data.paths_finished["end"] == end)]
    d=0
    for i in range(len(paths)):
        for j in range(len(paths)):
            if i!=j:
                path1 = paths["path"].values[i]
                path2 = paths["path"].values[j]
                d += distance_matrix(data,paths["path"].values[i],paths["path"].values[j]) #Compute the distance between the first articles visited after start by two players
    k = len(paths)*(len(paths)-1)
    return d/k

def plot_first_article_bar_chart(data, start, end):
    """
    Plot a bar chart of the first articles visited after start
    """
    paths = data.paths_finished[(data.paths_finished["start"] == start) & (data.paths_finished["end"] == end)]["paths"]
    dict_next = following_article(start, paths)

    # Colors for each category
    unique_categories = data.categories["1st cat"].unique()
    color_list = sns.color_palette("Set2", len(unique_categories))
    dict_cat_color = {unique_categories[i]: color_list[i] for i in range(len(unique_categories))}

    # Assign a color to each article
    colors = [dict_cat_color[data.categories[data.categories["article_name"] == article]["1st cat"].values[0]] 
              for article in dict_next.keys()]

    
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=list(dict_next.keys()), y=list(dict_next.values()), hue= list(dict_next.keys()) , palette=colors)
    plt.xticks(rotation=90)
    plt.title(f"First article visited after {start}")
    plt.xlabel("Articles")
    plt.ylabel("Count")

    handles = [plt.Rectangle((0,0),1,1, color=dict_cat_color[cat]) for cat in unique_categories]
    plt.legend(handles, unique_categories, title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

def get_random_path(data):
    """
    Create a random path
    """
    len_path = random.randint(1,10) # Length of the path
    path =[]
    i = random.randint(0,len(data.links)) # First article

    path.append(data.links.loc[i,"1st article"])
    for i in range(len_path):
        possible_links = data.links[data.links["1st article"]== path[-1]] # Possible links from the last article
        i = random.choice(possible_links.index)
        path.append(data.links.loc[i,"2nd article"]) # Add a random article to the path
    return path

    


        
