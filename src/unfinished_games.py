import pandas as pd
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import plotly.graph_objects as go





def get_games_unfinished(data):
    """
    Get the unfinished games played by the users
    """
    games = pd.DataFrame(columns=["start","target"])
    games["start"] = data.paths_unfinished["path"].apply(lambda x: x[0])
    games["target"] = data.paths_unfinished["target"]
    games.drop_duplicates(inplace=True)

    # Get the number of games played from start to target
    data.paths_unfinished["start"] = data.paths_unfinished["path"].apply(lambda x: x[0])
    games["nb_games"] = 0
    for i in tqdm(range(len(data.paths_unfinished))):
        start = data.paths_unfinished.loc[i,"start"]
        target = data.paths_unfinished.loc[i,"target"]
        games.loc[(games["start"]==start) & (games["target"]==target), "nb_games"] += 1

    games = games.reset_index(drop=True)

    return games

def success_rate(games_unfinished, games_finished):
    """
    Compute the success rate of the unfinished games
    """
    games_unfinished["success_rate"] = None
    
    for i in range(len(games_unfinished)):
        row = games_unfinished.iloc[i]
        start = row["start"]
        end = row["target"]
        
        g = games_finished[(games_finished["start"] == start) & (games_finished["end"] == end)]
        
        if len(g) > 0:
            success_rate = g["nb_games"].iloc[0] / (g["nb_games"].iloc[0] + row["nb_games"]) # Success rate = number of games finished / (number of games finished + number of games unfinished)
            games_unfinished.at[i, "success_rate"] = success_rate
    
    return games_unfinished

def success_rate_category(data):
    """
    Compute the success rate of the unfinished games by category of the target article
    """
    
    article_to_subject = data.categories.set_index('article_name')["1st cat"].to_dict()
    target_unfinished_count = data.paths_unfinished['target'].map(article_to_subject).value_counts()
    target_finished_count = data.paths_finished["end"].map(article_to_subject).value_counts()

    target_count = target_unfinished_count + target_finished_count
    target_success = target_finished_count / target_count 
    target_success = target_success.sort_values(ascending=False)
    return target_success

def success_rate_category_pair(data):
    """
    Compute the success rate of the games by pair (start category, target category)
    """

    # Mapping des articles vers leurs catégories
    article_to_category = data.categories.set_index('article_name')["1st cat"].to_dict()

    # Extraire les catégories de départ et de cible pour les chemins finis
    finished_start_categories = data.paths_finished['start'].map(article_to_category)
    finished_target_categories = data.paths_finished['end'].map(article_to_category)

    # Extraire les catégories de départ et de cible pour les chemins non finis
    unfinished_start_categories = data.paths_unfinished['start'].map(article_to_category)
    unfinished_target_categories = data.paths_unfinished['target'].map(article_to_category)

    # Compter les paires (start_category, target_category) pour les chemins finis et non finis
    finished_pairs = list(zip(finished_start_categories, finished_target_categories))
    unfinished_pairs = list(zip(unfinished_start_categories, unfinished_target_categories))

    finished_count = Counter(finished_pairs)
    unfinished_count = Counter(unfinished_pairs)

    # Calculer les compteurs totaux et les taux de réussite par paire
    total_count = Counter(finished_count) + Counter(unfinished_count)
    success_rate = {
        pair: finished_count[pair] / total_count[pair] if unfinished_count[pair] > 5 and finished_count[pair] > 0 else np.nan
        for pair in total_count if total_count[pair] > 0
    }

    # Trier par taux de réussite décroissant
    success_rate = dict(sorted(success_rate.items(), key=lambda x: x[1], reverse=True))

    return success_rate


def get_abandon_point(data):
    """
    Get the abandon point of the unfinished games
    """
    abandon = pd.DataFrame(columns=["abandon_point"])
    abandon["abandon_point"] = data.paths_unfinished["path"].apply(lambda x: x[-1])
    abandon.drop_duplicates(inplace=True)

    # Get the number of games stopped at the abandon point
    data.paths_unfinished["abandon_point"] = data.paths_unfinished["path"].apply(lambda x: x[-1])
    abandon["nb_games"] = 0
    for i in tqdm(range(len(data.paths_unfinished))):
        abandon_pt = data.paths_unfinished.loc[i,"abandon_point"]
        abandon.loc[(abandon["abandon_point"]==abandon_pt), "nb_games"] += 1

    abandon = abandon.reset_index(drop=True)

    return abandon

def target_category(data,unfinished_games,finished_games): 
    """
    Compute the distribution of the target articles by category in the finished and unfinished games
    """
    target_finished = finished_games[["end","nb_games"]]
    target_unfinished = unfinished_games[["target","nb_games"]].copy()

    # Get the category of the target articles in finished games
    target_finished["category"] = target_finished["end"].apply(
        lambda x: data.categories.loc[data.categories["article_name"] == x, "1st cat"].values[0]
        if not data.categories.loc[data.categories["article_name"] == x, "1st cat"].empty else None)

    # Get the category of the target articles in unfinished games
    target_unfinished["category"] = target_unfinished["target"].apply(
        lambda x: data.categories.loc[data.categories["article_name"] == x, "1st cat"].values[0]
        if not data.categories.loc[data.categories["article_name"] == x, "1st cat"].empty else None)
    
    # Get the number of games played by category in finished and unfinished games
    cat_count_finished = target_finished.groupby("category")["nb_games"].sum().reset_index()
    cat_count_unfinished = target_unfinished.groupby("category")["nb_games"].sum().reset_index()

    cat_count_finished["nb_games"]= cat_count_finished["nb_games"] /target_finished["nb_games"].sum() *100
    cat_count_unfinished["nb_games"]= cat_count_unfinished["nb_games"] /target_unfinished["nb_games"].sum() *100

    return cat_count_finished,cat_count_unfinished




## GET ABANDON REASON 

def is_deadend (data,article,t):
    """
    Check if an article is a deadend
    Input:
    - data: data object
    - article: article to check
    - t: threshold for the number of outgoing links
    Output:
    - boolean: True if the article is a deadend (less than t outgoing links), False otherwise
    """
    links = data.links.loc[(data.links["1st article"] == article)]
    if (len(links)<t):
        return True
    return False

def get_distance_totarget(data):
    """ 
    Get the semantic distance between the stopping article and the target article
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    stop_target_points = data.paths_unfinished[["stop_point", "target"]].drop_duplicates().values
    for stop, target in tqdm(stop_target_points):
        emb1 = model.encode(stop)
        emb2 = model.encode(target)
        d = cosine(emb1,emb2)
        data.paths_unfinished.loc[(data.paths_unfinished["stop_point"]==stop) & (data.paths_unfinished["target"]==target),"distance_to_target"] = d

def get_clics_totarget(data,article,target):
    """
    Get the number of clicks to reach the target article from the stopping article
    """

    if article not in data.articles["article_name"].values or target not in data.articles["article_name"].values:
        return 10 # If the article is not in the dataset, return a high number of clicks
    idx_article = int(data.articles[data.articles["article_name"]== article].index[0]) # Get the index of the article in the matrix
    idx_target = int(data.articles[data.articles["article_name"]== target].index[0]) #Get the index of the target in the matrix
    d = data.matrix[idx_article,idx_target] # Get the shortest path length between the article and the target 
    return d

def abandon_reason(x,t,data):
    stop_point = x["stop_point"]
    abandon_reason = None
    if is_deadend(data,stop_point,t): # If the stopping point is a deadend
        abandon_reason = "Reached Deadend"
    else:
        d = x["distance_to_target"] # Get the semantic distance to the target
        c = x["clicks_to_target"]
        if d < 0.1 or c<2:
            abandon_reason = "Target reached" # If the semantic distance is low, or article is less than two clicks away, the player may consider the target as reached
        elif d >0.9:
            abandon_reason = "Target far" # If the semantic distance is high, the player may consider the target as too far and abandon the game
    return abandon_reason


def get_reason_abandon(data,t_dead_end = 1):
    """
    Get the reason for abandoning the unfinished games
    """
    tqdm.pandas()
    data.paths_unfinished["stop_point"] = data.paths_unfinished["path"].progress_apply(lambda x : x[-1]) # Get the stopping point of the game
    data.paths_unfinished["abandon_reason"] = None

    data.paths_unfinished["clicks_to_target"] = data.paths_unfinished.progress_apply(lambda x: get_clics_totarget(data,x["stop_point"],x["target"]),axis=1) # Get the number of clicks missing to reach the target
    get_distance_totarget(data) # Get the semantic distance from the stopping point to the target

    data.paths_unfinished["abandon_reason"] = data.paths_unfinished.progress_apply(lambda x:abandon_reason(x,t_dead_end,data),axis=1)

def get_stoptarget (data):
    dict ={}
    for i in tqdm(range(len(data.paths_unfinished))):
        clicks = data.paths_unfinished.loc[i,"clicks_to_target"]
        distance = data.paths_unfinished.loc[i,"distance_to_target"]
        if clicks ==2 and distance >0.5:
            stop_point = data.paths_unfinished.loc[i,"stop_point"]
            target = data.paths_unfinished.loc[i,"target"]
            if (stop_point,target) in dict:
                dict[(stop_point,target)] += 1
            else:
                dict[(stop_point,target)] = 1
    return dict

def plot_success_rate_heatmap(data):
    """
    Plots a heatmap of success rates by (cat_start, cat_target).
    
    Parameters:
    success_rate (dict): A dictionary where keys are (cat_start, cat_target) pairs 
                         and values are success rates.
    """

    success_rate = success_rate_category_pair(data)
    df = pd.DataFrame(list(success_rate.items()), columns=['Category Pair', 'Success Rate'])
    df[['Start Category', 'Target Category']] = pd.DataFrame(df['Category Pair'].tolist(), index=df.index)
    df.drop(columns=['Category Pair'], inplace=True)

    heatmap_data = df.pivot(index='Start Category', columns='Target Category', values='Success Rate')

    plt.figure(figsize=(7,5))
    sns.heatmap(heatmap_data, cmap = 'Blues')
    plt.title("Success Rate by (Start Category, Target Category)")
    plt.xlabel("Target Category")
    plt.ylabel("Start Category")

    plt.show()


def fig_to_target(data):
    clicks_to_target = data.paths_unfinished['clicks_to_target']
    click_counts = pd.Series(clicks_to_target).value_counts().sort_index()
    distance_to_target = data.paths_unfinished['distance_to_target']

    plt.figure()
    sns.barplot(x=click_counts.index,y=click_counts.values)
    plt.title("Number of clicks missing to reach the target")
    plt.xlabel("Number of clicks")
    plt.ylabel("Number of unfinished games")

    plt.show()

    plt.figure()
    sns.histplot(x=distance_to_target,color= "Red")
    plt.title("Semantic distance from stopping point to target article")
    plt.xlabel("Distance to target")
    plt.ylabel("Number of unfinished games")

    plt.show()




def bar_plot_connections(data):
    dict_pairs = get_stoptarget(data)
    dict_common = {k: v for k, v in dict_pairs.items() if v >= 10}
    dict_c = {}
    for articles in dict_common.keys():
        stop = articles[0]
        target = articles[1]
        if stop in data.categories["article_name"].values and target in data.categories["article_name"].values:
            cat_stop = data.get_article_category(stop, "1st cat")
            cat_target = data.get_article_category(target, "1st cat")
            if (cat_stop, cat_target) in dict_c:
                dict_c[(cat_stop, cat_target)] += 1
            else:
                dict_c[(cat_stop, cat_target)] = 1

    data_dict = {"Start": [], "Target": [], "Value": []}
    for (cat_start, cat_end), value in dict_c.items():
        data_dict["Start"].append(cat_start)
        data_dict["Target"].append(cat_end)
        data_dict["Value"].append(value)

    df = pd.DataFrame(data_dict)

    # Transform to plot
    pivot = df.pivot(index="Start", columns="Target", values="Value").fillna(0)

    colors = sns.color_palette("Set2", len(pivot.columns))  
    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 8),
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )

    ax.set_ylabel("Commonly unkown connections between categories", fontsize=12)
    ax.set_xlabel("Start Category", fontsize=12)
    ax.legend(
        title="Target Category",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,
        title_fontsize=12,
    )

    plt.xticks(rotation=45, fontsize=10, ha="right")
    plt.yticks(fontsize=10)
    plt.title("Commonly unkown connections between categories", fontsize=14)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    plt.show()



