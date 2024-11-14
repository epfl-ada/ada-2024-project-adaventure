
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns



## DIFFERENT FUNCTIONS TO COMPUTE THE DISTANCE BETWEEN TWO PATHS #########################

def distance_Jaccard(data,path1,path2):
    inter = len(set(path1).intersection(set(path2)))
    union = len(set(path1).union(set(path2)))
    return  1-inter/union

def distance_matrix(data,path1,path2):
    path1 = [int(data.articles[data.articles["article_name"]== x].index[0]) for x in path1]
    path2 = [int(data.articles[data.articles["article_name"]== x].index[0]) for x in path2]
    if(len(path1)<len(path2)):
        path1,path2 = path2,path1
    n1 = len(path1)
    n2 = len(path2)
    D = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            D[i,j] = data.matrix[path1[i],path2[j]]
    min_d = np.min(D,axis=1)
    return (np.max(min_d)) # mean :  in average how many clicks to reach an article in the second path / max : the maximum number of clicks to reach an article in the second path

##########################################################################################

def get_games(data):
    games = pd.DataFrame(columns=["start","end"])
    games[["start","end"]] = pd.DataFrame(data.paths_finished["path"].apply(lambda x: [x[0], x[-1]]).tolist(), index=data.paths_finished.index)
    games.drop_duplicates(inplace=True)

    data.paths_finished["start"] = data.paths_finished["path"].apply(lambda x: x[0])
    data.paths_finished["end"] = data.paths_finished["path"].apply(lambda x: x[-1])
    games["nb_games"] = 0
    for i in tqdm(range(len(data.paths_finished))):
        start = data.paths_finished.loc[i,"start"]
        end = data.paths_finished.loc[i,"end"]
        games.loc[(games["start"]==start) & (games["end"]==end), "nb_games"] += 1

    games = games.reset_index(drop=True)

    return games

def get_sim_matrix(data,start,end,distance):
    paths = data.paths_finished[(data.paths_finished["start"] == start) & (data.paths_finished["end"] == end)]
    matrix_distance = np.zeros((len(paths),len(paths)))
    if len(paths) == 0:
        raise ValueError("No game played from ",start," to ",end)
    if len(paths) == 1:
        print("Only one game played from ", start, " to ", end)
        return None
    for i in range(len(paths)):
        path1 = paths["path"].values[i]
        for j in range(len(paths)):
            path2 = paths["path"].values[j]
            sim = max(distance(data,path1,path2),distance(data,path2,path1))
            matrix_distance[i][j] = sim
    return matrix_distance

def plot_sim_matrices(data,starts,ends,distance = distance_Jaccard,title = "Similarity matrix",**kwargs):
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        matrix_sim = get_sim_matrix(data,start,end,distance)
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
    list=[]
    for path in paths:
        if article in path:
            indexes = [i for i, x in enumerate(path) if x == article]
            for i in indexes:
                if i < len(path)-1:
                    list.append(path[i+1])
                elif len(path)==1:
                    list.append(article)
    dict ={x:list.count(x) for x in list}
    return dict

def distance_first_article(data,start,end):
    paths = data.paths_finished[(data.paths_finished["start"] == start) & (data.paths_finished["end"] == end)]
    d=0
    for i in range(len(paths)):
        for j in range(len(paths)):
            if i!=j:
                path1 = paths["path"].values[i]
                path2 = paths["path"].values[j]
                d += distance_matrix(data,paths["path"].values[i],paths["path"].values[j])
    k = len(paths)*(len(paths)-1)
    return d/k

def plot_first_article_bar_chart(data, start, end):
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
