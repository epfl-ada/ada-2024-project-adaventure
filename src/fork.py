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
    print(f"Total comparisons made: {total_comparisons}")
    print(f"Paths with higher PageRank alternatives: {higher_pagerank_count}")
    print(f"Percentage: {(higher_pagerank_count / total_comparisons) * 100:.2f}%")
    return matrix1, matrix2, unused_links, article_names


def plot_badChoices(matrix1, unused_links, article_names):

    # Sum up the columns of matrix1
    column_sums_m1 = np.array([np.sum(col[col > 0]) for col in (matrix1).T])
    column_sums_ul = np.array([np.sum(col[col > 0]) for col in (unused_links).T])
    column_sums = np.where(column_sums_ul > 10, column_sums_m1 / column_sums_ul, np.nan)
    print(len(column_sums))
    non_zero_column_sums_count = np.count_nonzero(column_sums)
    print(f"Number of non-zero values in column_sums: {non_zero_column_sums_count}")
    # Plot the distribution of the column sums as a barplot with 200 bins
    plt.figure(figsize=(10, 6))
    sns.histplot(column_sums, bins=40, color='purple', label='Column Sums Distribution', kde=False)
    plt.xlabel('Share of forgoing the more central link')
    plt.ylabel('Count')
    plt.title('Distribution of Column Sums of Matrix1 / Used Links')
    plt.legend()
    plt.show()

    # Filter articles with column sums above 0.5
    print(len(column_sums))
    forgone_indices = np.where((1 > column_sums))[0]

    # Create a DataFrame to associate articles with their forgone values and column sums ul
    forgone_articles_df = pd.DataFrame({
        'Article': [article_names[i] for i in forgone_indices],
        'Forgone Value': [column_sums[i] for i in forgone_indices],
        'Column Sums UL': [column_sums_ul[i] for i in forgone_indices]
    })

    # Sort by forgone value in descending order and get the top 100 articles
    top_forgone_articles = forgone_articles_df.sort_values(by='Forgone Value', ascending=False)
    print(top_forgone_articles.head(20))
    return top_forgone_articles, column_sums


# Function to create an interactive scatter plot
def plot_forgone_vs_usage(forgone_df, title='Forgone Value vs Column Sums UL'):
    # Create the interactive scatter plot with Plotly
    fig = px.scatter(
        forgone_df,
        x='Column Sums UL',
        y='Forgone Value',
        text='Article',
        hover_data=['Article', 'Forgone Value', 'Column Sums UL'],
        title=title,
        labels={'Forgone Value': 'Forgone Value', 'Column Sums UL': 'Number of Times Link Was Used'},
    )

    # Customize marker size and opacity
    fig.update_traces(marker=dict(size=8, opacity=0.7), textposition='top center')

    # Add axis lines for better context
    fig.add_vline(x=forgone_df['Column Sums UL'].mean(), line_dash="dash", line_color="red", annotation_text="Mean UL")
    fig.add_hline(y=forgone_df['Forgone Value'].mean(), line_dash="dash", line_color="red", annotation_text="Mean Forgone")

    # Show the plot
    fig.show()
