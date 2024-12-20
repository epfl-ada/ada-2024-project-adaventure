import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def load_fork_matrix(data, df_hubs):
    # Load shortest path matrix and initialize matrices
    shortest_path_matrix = data.load_shortest_paths()
    unused_links = np.zeros_like(shortest_path_matrix)  # Counts how many times a link is used
    matrix1 = np.zeros_like(shortest_path_matrix)
    matrix2 = np.zeros_like(shortest_path_matrix)

    # Set article names as the index for shortest_path_matrix
    article_names = df_hubs['article_names'].tolist()
    shortest_path_df = pd.DataFrame(shortest_path_matrix, index=article_names, columns=article_names)

    # Map PageRank scores to article names
    pagerank_scores = df_hubs.set_index('article_names')['pagerank_score']

    # Create a mapping of article names to indices for faster matrix access
    article_index = {name: idx for idx, name in enumerate(article_names)}

    # Track results
    higher_pagerank_count = 0
    total_comparisons = 0

    links, prev_link, indices = [], None, []
    for i, link in enumerate(data.links['1st article']):
        if link == prev_link:
            links[-1].append(data.links['2nd article'][i])
        else:
            links.append([data.links['2nd article'][i]])
            indices.append(link)
        prev_link = link


    # Convert necessary data to numpy arrays for faster access
    pagerank_scores_np = pagerank_scores.to_numpy()
    shortest_path_matrix_np = shortest_path_df.to_numpy()

    # Iterate through each finished path
    for path in data.paths_finished['path']:
        target_article = path[-1]
        target_index = article_index[target_article]

        for i in range(len(path) - 1):
            current_article = path[i]
            next_article = path[i + 1]
            
            current_index = article_index[current_article]
            next_index = article_index[next_article]

            # Get the shortest distance from the next article to the target
            next_to_target_dist = shortest_path_matrix_np[next_index, target_index]
            next_article_pagerank = pagerank_scores_np[next_index]

            # Get all valid articles (links) from the current article
            index = np.where(np.array(indices) == current_article)[0][0]
            valid_indices = np.array([article_index[article] for article in links[index] if article in article_index])

            # Filter articles with the same or shorter distance to the target
            valid_distances = shortest_path_matrix_np[valid_indices, target_index]
            valid_pagerank_scores = pagerank_scores_np[valid_indices]

            # Articles with equal distance and higher PageRank
            valid_articles_equal = valid_indices[
                (valid_distances == next_to_target_dist) & (valid_pagerank_scores > next_article_pagerank)
            ]

            # Articles with shorter distance and higher PageRank
            valid_articles_less = valid_indices[
                (valid_distances < next_to_target_dist) & (valid_pagerank_scores > next_article_pagerank)
            ]
            valid_articles_page_rank = valid_indices[
                (valid_pagerank_scores > next_article_pagerank)
            ]

            # Update matrices
            unused_links[current_index, valid_articles_page_rank] += 1
            matrix1[next_index, valid_articles_equal] += 1
            matrix2[next_index, valid_articles_less] += 1

            # Count comparisons
            if valid_articles_equal.size > 0 or valid_articles_less.size > 0:
                higher_pagerank_count += 1

            total_comparisons += 1

    # Display the results
    print(f"Total choices made: {total_comparisons}")
    print(f"Choices with higher PageRank alternatives: {higher_pagerank_count}")
    print(f"Percentage: {(higher_pagerank_count / total_comparisons) * 100:.2f}%")
    return matrix1, matrix2, unused_links, article_names


def plot_badChoices(matrix_forgone, unused_links, article_names, df_hubs, data):

    # sum up the columns per article
    column_sums_forgone = np.array([np.sum(col[col > 0]) for col in (matrix_forgone).T])
    column_sums_unused = np.array([np.sum(col[col > 0]) for col in (unused_links).T])
    column_sums = np.where(column_sums_unused > 10, column_sums_forgone / column_sums_unused, np.nan)
    

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.histplot(data=column_sums, bins=40, kde=False, color="#0f4584", alpha=1, ax=ax)
    ax.set_title('Distribution of suboptimal choices')
    ax.set_xlabel('Share of forgoing the more central link per article')
    ax.set_ylabel('Amount of articles')
    ax.set_yscale('log')
    ax.grid(True, axis='y', color='white', linestyle='-', linewidth=0.5)
    plt.show()
    

    # filter out only those articles that have been an option more than once 
    forgone_indices = np.where((1 > column_sums))[0]
    forgone_articles_df = pd.DataFrame({
        'Article': [article_names[i] for i in forgone_indices],
        'Forgone Value': [column_sums[i] for i in forgone_indices],
        'Column Sums UL': [column_sums_unused[i] for i in forgone_indices]
    })

    # sort descending and print
    top_forgone_articles = forgone_articles_df.sort_values(by='Forgone Value', ascending=False)
    print('The articles that are forgone the most are:')
    print(top_forgone_articles.head(20))

    # merge into df_hubs
    df_hubs = df_hubs.merge(forgone_articles_df[['Article', 'Column Sums UL', 'Forgone Value']], left_on='article_names', right_on='Article', how='left')
    df_hubs.drop(columns=['Article'], inplace=True)
    # add the first category to the df_hubs
    df_hubs = df_hubs.merge(right= data.categories[['article_name', '1st cat']], 
                                    how= 'left',
                                    left_on='article_names',
                                    right_on='article_name').drop(columns=['article_name'])
    
    return df_hubs, top_forgone_articles

def plot_hubScoreVSforgone(df_hubs, category=None):
    # Ignore NaN and choices that appear less than x times
    df_hubs = df_hubs.dropna(subset=['hub_score', 'Forgone Value'])
    df_hubs = df_hubs[df_hubs['Column Sums UL'] >= 50]
    
    # color according to category
    categories = ['Science', 'Geography', 'Countries', 'History', 'People', 'Religion', 'Citizenship', 'Everyday life',
             'Design and Technology', 'Language and literature', 'IT', 'Business Studies', 'Music', 'Mathematics', 'Art']
    color = '1st cat'
    category_orders = {'1st cat': categories}
    
    fig = px.scatter(
        df_hubs,
        x='hub_score',
        y='Forgone Value',
        color=color,
        category_orders=category_orders,
        hover_data=['article_names', 'Forgone Value', 'Column Sums UL'],
        log_x=True,
        log_y=True,
        title='Forgone percentage vs PageRank',
        labels={'Forgone Value': 'Forgone percentage', 'Column Sums UL': 'Number of Times Link Was Available'},
        )
    
    fig.update_layout(plot_bgcolor="#ebeaf2", xaxis_title="PageRank")
    fig.update_traces(marker=dict(size=4, opacity=1))
    
    # fig.update_xaxes(range=[0.00001, 0.01])
    # fig.update_yaxes(range=[0.002, 0.6])
    
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




