import numpy as np
import pandas as pd
from src.shortest_paths import calculate_all_shortest_paths
import ast
import matplotlib.pyplot as plt

def get_hub_positions(data, df_hubs):
    '''this calculates the percentage of hub usage at each position in the path'''
    # Sample data: replace with actual data arrays
    finished_paths = data.paths_finished['path']
    unfinished_paths = data.paths_unfinished['path']
    csv_path = calculate_all_shortest_paths(data)
    csv_path = calculate_all_shortest_paths(data)
    # load the shortest paths
    shortest_paths = pd.read_csv(csv_path)
    shortest_paths = shortest_paths['shortest_paths']
    shortest_paths = shortest_paths.apply(ast.literal_eval)
    shortest_paths = [path_[0] for path_ in shortest_paths]
    
    # define the top 10% of hubs.
    upper_10_percent = np.nanpercentile(df_hubs["hub_score"], 90)
    df_hubs['is_hub'] = df_hubs['hub_score'] > upper_10_percent  # Define hub threshold
    print('found ', len(df_hubs), 'articles, of which ', sum(df_hubs['is_hub']), 'are in the top 10percent of hubs')

    # Count hub usage at each position for each type of path
    finished_hub_positions = np.zeros(max(len(path) for path in finished_paths)-1)
    finished_position_usage = finished_hub_positions.copy()
    for path in finished_paths:
        for i, article in enumerate(path[1:-1]):
            if df_hubs[df_hubs['article_names'] == article]['is_hub'].values[0]:
                finished_hub_positions[i] += 1
            finished_position_usage[i] += 1
    unfinished_hub_positions = np.zeros(max(len(path) for path in unfinished_paths)-1)
    unfinished_position_usage = unfinished_hub_positions.copy()
    for path in unfinished_paths:
        for i, article in enumerate(path[1:-1]):
            if df_hubs[df_hubs['article_names'] == article]['is_hub'].values[0]:
                unfinished_hub_positions[i] += 1
            unfinished_position_usage[i] += 1
    shortest_hub_positions = np.zeros(max(len(path) for path in unfinished_paths)-1)
    shortest_position_usage = unfinished_hub_positions.copy()
    for path in shortest_paths:
        for i, article in enumerate(path[1:-1]):
            if df_hubs[df_hubs['article_names'] == article]['is_hub'].values[0]:
                shortest_hub_positions[i] += 1
            shortest_position_usage[i] += 1

    # calculate the percentage of hub usage at each position
    finished_hub_positions = np.divide(finished_hub_positions, finished_position_usage, where=finished_position_usage!=0)
    unfinished_hub_positions = np.divide(unfinished_hub_positions, unfinished_position_usage, where=unfinished_position_usage!=0)
    shortest_hub_positions = np.divide(shortest_hub_positions, shortest_position_usage, where=shortest_position_usage!=0)
    return finished_hub_positions, unfinished_hub_positions, shortest_hub_positions

def plot_hub_positions(finished_hub_positions, unfinished_hub_positions, shortest_hub_positions):
    ''' plots the results of get_hub_positions'''
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(finished_hub_positions) + 1), finished_hub_positions, label='Finished Games', color='green')
    plt.plot(range(1, len(unfinished_hub_positions) + 1), unfinished_hub_positions, label='Unfinished Games', color='red')
    plt.plot(range(1, 4), shortest_hub_positions[:3], label='Shortest Paths', color='blue')
    plt.xlim(1, 10)
    plt.xlabel('Path Position')
    plt.ylabel('Top Hub Usage in Percentage')
    plt.title('Hub Usage at Different Path Positions')
    plt.legend()
    plt.grid(True)
    plt.savefig('hub_usage_plot.png')
    plt.show()

