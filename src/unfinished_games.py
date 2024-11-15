import pandas as pd
from tqdm import tqdm


import pandas as pd
from tqdm import tqdm


def get_games_unfinished(data):
    games = pd.DataFrame(columns=["start","target"])
    games["start"] = data.paths_unfinished["path"].apply(lambda x: x[0])
    games["target"] = data.paths_unfinished["target"]
    games.drop_duplicates(inplace=True)

    data.paths_unfinished["start"] = data.paths_unfinished["path"].apply(lambda x: x[0])
    games["nb_games"] = 0
    for i in tqdm(range(len(data.paths_unfinished))):
        start = data.paths_unfinished.loc[i,"start"]
        target = data.paths_unfinished.loc[i,"target"]
        games.loc[(games["start"]==start) & (games["target"]==target), "nb_games"] += 1

    games = games.reset_index(drop=True)

    return games

def success_rate(games_unfinished, games_finished):
    games_unfinished["success_rate"] = None
    
    for i in range(len(games_unfinished)):
        row = games_unfinished.iloc[i]
        start = row["start"]
        end = row["target"]
        
        g = games_finished[(games_finished["start"] == start) & (games_finished["end"] == end)]
        
        if len(g) > 0:
            success_rate = g["nb_games"].iloc[0] / (g["nb_games"].iloc[0] + row["nb_games"])
            games_unfinished.at[i, "success_rate"] = success_rate
    
    return games_unfinished

def success_rate_category(data, games_unfinished):
    
    article_to_subject = data.categories.set_index('article_name')["1st cat"].to_dict()
    target_unfinished_count = data.paths_unfinished['target'].map(article_to_subject).value_counts()
    target_finished_count = data.paths_finished["end"].map(article_to_subject).value_counts()

    target_count = target_unfinished_count + target_finished_count
    target_success = target_finished_count / target_count 
    target_success = target_success.sort_values(ascending=False)
    return target_success


def get_abandon_point(data):
    abandon = pd.DataFrame(columns=["abandon_point"])
    abandon["abandon_point"] = data.paths_unfinished["path"].apply(lambda x: x[-1])
    abandon.drop_duplicates(inplace=True)

    data.paths_unfinished["abandon_point"] = data.paths_unfinished["path"].apply(lambda x: x[-1])
    abandon["nb_games"] = 0
    for i in tqdm(range(len(data.paths_unfinished))):
        abandon_pt = data.paths_unfinished.loc[i,"abandon_point"]
        abandon.loc[(abandon["abandon_point"]==abandon_pt), "nb_games"] += 1

    abandon = abandon.reset_index(drop=True)

    return abandon

def target_category(data,unfinished_games,finished_games): 
    target_finished = finished_games[["end","nb_games"]]
    target_unfinished = unfinished_games[["target","nb_games"]].copy()

    target_finished["category"] = target_finished["end"].apply(
        lambda x: data.categories.loc[data.categories["article_name"] == x, "1st cat"].values[0]
        if not data.categories.loc[data.categories["article_name"] == x, "1st cat"].empty else None)


    target_unfinished["category"] = target_unfinished["target"].apply(
        lambda x: data.categories.loc[data.categories["article_name"] == x, "1st cat"].values[0]
        if not data.categories.loc[data.categories["article_name"] == x, "1st cat"].empty else None)
    
    cat_count_finished = target_finished.groupby("category")["nb_games"].sum().reset_index()
    cat_count_unfinished = target_unfinished.groupby("category")["nb_games"].sum().reset_index()

    cat_count_finished["nb_games"]= cat_count_finished["nb_games"] /target_finished["nb_games"].sum() *100
    cat_count_unfinished["nb_games"]= cat_count_unfinished["nb_games"] /target_unfinished["nb_games"].sum() *100

    return cat_count_finished,cat_count_unfinished
