import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def load_overlooked_score(data, df_hubs):
    '''we calculate a score showing how often any one article is overlooked by a player
    despite having a <= distance to the target and having a better PR score
    
    Returns:
        the hubs dataframe updated by the score and a count
        and a matrix holding the score.
    '''
    # Load shortest path matrix and initialize matrices
    shortest_path_matrix = data.load_shortest_paths()
    # Counts how many times a link is not used despite being equally distant from the target as the used one
    unused_links = np.zeros_like(shortest_path_matrix)  
    # counts how many times an unused link has a higher PR score than the used one
    higherPR_smallerD = np.zeros_like(shortest_path_matrix) 

    # Set article names as the index for shortest_path_matrix
    article_names = df_hubs['article_names'].tolist()
    shortest_path_df = pd.DataFrame(shortest_path_matrix, index=article_names, columns=article_names)

    # map PageRank scores to article names
    pagerank_scores = df_hubs.set_index('article_names')['pagerank_score']
    # create a mapping of article names to indices for computationally faster matrix access
    article_index = {name: idx for idx, name in enumerate(article_names)}

    # Track results
    higher_pagerank_count = 0
    total_comparisons = 0

    # create a list of all links from an article like [[links from article1],[links from a2],[..]....]
    links, prev_link, indices = [], None, []
    for i, link in enumerate(data.links['1st article']):
        if link == prev_link:
            links[-1].append(data.links['2nd article'][i])
        else:
            links.append([data.links['2nd article'][i]])
            indices.append(link)
        prev_link = link


    # convert to numpy arrays for faster access
    pagerank_scores_np = pagerank_scores.to_numpy()
    shortest_path_matrix_np = shortest_path_df.to_numpy()

    # iterate through each path
    for path in data.paths_finished['path']:
        target_article = path[-1]
        target_index = article_index[target_article]

        for i in range(len(path) - 1):
            current_article = path[i]
            next_article = path[i + 1]
            current_index = article_index[current_article]
            next_index = article_index[next_article]

            # get distance and pagerank of next article for comparison
            next_to_target_dist = shortest_path_matrix_np[next_index, target_index]
            next_article_pagerank = pagerank_scores_np[next_index]

            # get all valid articles linked from the current article
            index = np.where(np.array(indices) == current_article)[0][0]
            valid_indices = np.array([article_index[article] for article in links[index] if article in article_index])
            # get their distance to the target and PR score
            valid_distances = shortest_path_matrix_np[valid_indices, target_index]
            valid_pagerank_scores = pagerank_scores_np[valid_indices]

            # articles with equal or shorter distance and higher PageRank
            valid_articles_higherPR_smallerD = valid_indices[(valid_distances <= next_to_target_dist) & (valid_pagerank_scores > next_article_pagerank)]
            # articles with equal or shorter distance
            valid_articles = valid_indices[(valid_distances <= next_to_target_dist)]

            # update matrices
            unused_links[current_index, valid_articles] += 1
            higherPR_smallerD[current_index, valid_articles_higherPR_smallerD] += 1

            # Count comparisons
            if valid_articles_higherPR_smallerD.size > 0:
                higher_pagerank_count += 1

            total_comparisons += 1

    # Display the results
    print(f"Total choices made: {total_comparisons}")
    print(f"Choices with higher PageRank alternatives: {higher_pagerank_count}")
    print(f"Percentage: {(higher_pagerank_count / total_comparisons) * 100:.2f}%")
    
    # sum up the column for each article
    column_sums_unused = np.array([np.sum(col[col > 0]) for col in (unused_links).T])
    column_sums_higherPR_smallerD = np.array([np.sum(col[col > 0]) for col in (higherPR_smallerD).T])
    share_overlooked = np.where((column_sums_unused > 10) & (column_sums_unused != 0), column_sums_higherPR_smallerD / column_sums_unused, np.nan)
    
    # filter out only those articles that have been an option more than once 
    Overlooked_indices = np.where((1 > share_overlooked))[0]
    Overlooked_articles_df = pd.DataFrame({
        'Article': [article_names[i] for i in Overlooked_indices],
        'Overlooked Value': [share_overlooked[i] for i in Overlooked_indices],
        'Column Sums UL': [column_sums_unused[i] for i in Overlooked_indices],
    })
    
    # merge into df_hubs
    df_hubs = df_hubs.merge(Overlooked_articles_df[['Article', 'Column Sums UL', 'Overlooked Value']], left_on='article_names', right_on='Article', how='left')
    df_hubs.drop(columns=['Article'], inplace=True)
    # add the first category to the df_hubs
    df_hubs = df_hubs.merge(right= data.categories[['article_name', '1st cat']], 
                                    how= 'left',
                                    left_on='article_names',
                                    right_on='article_name').drop(columns=['article_name'])
    return share_overlooked, df_hubs


def plot_badChoices(column_sums):
    # plot a histogram of the share of suboptimal choices
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.histplot(data=column_sums, bins=40, kde=False, color="#0f4584", alpha=1, ax=ax)
    ax.set_title('Distribution of suboptimal choices')
    ax.set_xlabel('Share of forgoing the more central link per article')
    ax.set_ylabel('Amount of articles')
    ax.set_yscale('log')
    ax.grid(True, axis='y', color='white', linestyle='-', linewidth=0.5)
    plt.savefig('DistributionBadChoices.png')
    plt.show()

def plot_hubScoreVSoverlooked(df_hubs, category=None):
    '''plots the hub score against the overlooked value, 
    while coloring the points according to category.'''
    # Ignore NaN and choices that appear less than x times
    df_hubs = df_hubs.dropna(subset=['hub_score', 'Overlooked Value'])
    df_hubs = df_hubs[df_hubs['Column Sums UL'] >= 50]
    
    # color according to category
    categories = ['Science', 'Geography', 'Countries', 'History', 'People', 'Religion', 'Citizenship', 'Everyday life',
             'Design and Technology', 'Language and literature', 'IT', 'Business Studies', 'Music', 'Mathematics', 'Art']
    color = '1st cat'
    category_orders = {'1st cat': categories}
    
    # make the scatter plot
    fig = px.scatter(
        df_hubs,
        x='hub_score',
        y='Overlooked Value',
        color=color,
        category_orders=category_orders,
        hover_data=['article_names', 'Overlooked Value', 'Column Sums UL'],
        log_x=True,
        log_y=True,
        title='Overlooked articles?',
        labels={'Overlooked Value': 'Overlooked Share',
                 'Column Sums UL': 'Number of Times Link Was Available',
                   'hub_score': 'PageRank Score',
                   'article_names': 'Article',
                   '1st cat': 'Category'},
        )
    
    fig.update_layout(plot_bgcolor="#ebeaf2")
    fig.update_traces(marker=dict(size=4, opacity=1))
    
    # save plot as html
    if category is None:
        fig.write_html("hubscore_vs_forgone.html", config={"displayModeBar": False})
    else:
        fig.write_html(f"hubscore_vs_forgone{category}.html", config={"displayModeBar": False})
    
    fig.show()

def plot_views_forgone(data,df_hubs):
    df_hubs["category"] = df_hubs["Article"].apply(lambda x: data.get_article_category(x,"1st cat"))
    metadata = pd.read_csv('data/metadata.csv')
    mean_views = metadata['views'].mean()
    df_hubs = df_hubs.merge(metadata[['article_name', 'views']], 
                        left_on='Article', 
                        right_on='article_name', 
                        how='left')

    biggest_forgone = df_hubs.groupby( 'category', group_keys=False).apply(lambda x: x.nsmallest(10, 'Forgone Value'))
    plt.figure(figsize=(10, 6)) 
        
    # Double barplot per category and quadrant
    sns.barplot(data=biggest_forgone, x='category', y='views', hue='category',errorbar=None)
        
    # Add line for the mean views across all articles 
    plt.axhline(mean_views, color='red', linestyle='--', linewidth=1, label='Mean number views across all articles')
    plt.title('Average monthly views of the biggest overlooked articles', fontsize=16)
    plt.xlabel('Quadrant', fontsize=14)
    plt.xticks(rotation=90)
    plt.ylabel('Average monthly views', fontsize=14)

    plt.legend(title='Category', loc='upper right')
    plt.tight_layout()
    plt.savefig("views_forgone.png", dpi=300)
    plt.show()




