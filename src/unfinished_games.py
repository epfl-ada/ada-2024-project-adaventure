import pandas as pd
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from collections import Counter


model = SentenceTransformer('all-MiniLM-L6-v2')


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
    links = data.links.loc[(data.links["1st article"] == article)]
    if (len(links)<t):
        return True
    return False

def get_distance_totarget(article,target):
    emb1 = model.encode(article)
    emb2 = model.encode(target)
    return cosine(emb1, emb2)

def get_clics_totarget(data,article,target):
    if article not in data.articles["article_name"].values or target not in data.articles["article_name"].values:
        return 10
    idx_article = int(data.articles[data.articles["article_name"]== article].index[0])
    idx_target = int(data.articles[data.articles["article_name"]== target].index[0])
    d = data.matrix[idx_article,idx_target]
    return d


def get_reason_abandon(data,t_dead_end = 1):
    data.paths_unfinished["stop_point"] = data.paths_unfinished["path"].apply(lambda x : x[-1])
    data.paths_unfinished["abandon_reason"] = None
    data.paths_unfinished["clicks_to_target"] = data.paths_unfinished.apply(lambda x: get_clics_totarget(data,x["stop_point"],x["target"]),axis=1) 
    data.paths_unfinished["distance_to_target"] = data.paths_unfinished.apply(lambda x: get_distance_totarget(x["stop_point"],x["target"]),axis=1)
    for i in tqdm(range(len(data.paths_unfinished))):
        stop_point = data.paths_unfinished.loc[i,"stop_point"]
        abandon_reason = None
        if is_deadend(data,stop_point,t_dead_end):
            abandon_reason = "deadend"
        else:
            d = data.paths_unfinished.loc[i,"distance_to_target"]
            if d < 0.1:
                abandon_reason = "target_reached"
            elif d >0.9:
                abandon_reason = "target_far"
        data.paths_unfinished.loc[i,"abandon_reason"] = abandon_reason
