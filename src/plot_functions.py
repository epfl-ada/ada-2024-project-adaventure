import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_category_distribution(column, title):
    fig, ax = plt.subplots(1, figsize=(10, 6))
    column.value_counts().plot(kind='bar', ax=ax, color="#0f4584", title=title)
    ax.set_xlabel("")
    ax.set_ylabel("Number of articles")
    plt.tight_layout()
    plt.show()

def plot_box_plot(df, column_names, titles):
    
    n = len(column_names)

    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(12, 6))

    y_min = min(df[column_names].min())
    y_max = max(df[column_names].max())

    for i in range(n):
        df[column_names[i]].plot(kind='box', ax=axes[i], title=titles[i])

    for ax in axes:
        ax.set_ylim([y_min, y_max])
        #ax.set_yscale('log')

    plt.tight_layout()
    plt.show()


def plot_two_distributions(df, column_name1, column_name2, title1, title2,
                           xlabel1, ylabel1,xlabel2, ylabel2,log_scale = False):
    """
    This function plots the distribution of two provided columns in a DataFrame
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    sns.histplot(data=df, x=column_name1, bins=40, kde=False, ax=ax[0], color="#0f4584", alpha = 1)
    ax[0].set_title(title1)
    ax[0].set_xlabel(xlabel1)
    ax[0].set_ylabel(ylabel1)
    ax[0].grid(True, axis='y', color='white', linestyle='-', linewidth=0.5) 

    # Plot for source_counts
    sns.histplot(data=df, x=column_name2, bins=40, kde=False, ax=ax[1], color="#0f4584", alpha = 1)
    ax[1].set_title(title2)
    ax[1].set_xlabel(xlabel2)
    ax[1].set_ylabel(ylabel2)
    ax[1].grid(True, axis='y', color='white', linestyle='-', linewidth=0.5)

    if log_scale == True:
        ax[0].set_yscale('log') 
        ax[1].set_yscale('log') 

    plt.tight_layout()
    plt.savefig(f"images/png/combined_histograms{column_name1}.png")
    plt.show()


def plot_top10_pagerank(df_filtered_hubs):
    top_10_pr = df_filtered_hubs.sort_values(by="hub_score", ascending=False).head(10)
    inverted_palette = sns.color_palette("Blues", n_colors=10)


    #bar plot 
    plt.figure(figsize=(10, 6))
    bar_plt = sns.barplot(
        data=top_10_pr,
        y="article_names",
        x="hub_score",
        hue="hub_score",
        palette= inverted_palette
    )

    # Add labels and title
    plt.xlabel("PageRank score", fontsize=12)
    plt.ylabel("Article Names", fontsize=12)
    plt.title("Top 10 Articles by PageRank score", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.legend().remove()
    plt.tight_layout()
    plt.savefig("images/png/top_page_rank_articles.png")
    plt.show()


def get_correlation_matrix_row(df_filtered_hubs, cols):
    df_hubs_corr = df_filtered_hubs[cols].corr()
    df_hubs_corr.rename(columns={'hub_score': 'PageRank score'}, inplace=True)
    df_hubs_corr.rename(index={'hub_score': 'PageRank score', 
                            'source_counts': 'Source count', 'target_counts': 'Target count',
                            'mean_shortest_path': 'Mean shortest path from article','mean_shortest_path_to_article': 'Mean shortest path to article'}, inplace=True)

    plt.figure(figsize=(12,4))
    corr_figure = sns.heatmap(df_hubs_corr[['PageRank score']].T,
                vmin=-1,
                cmap='coolwarm',
                annot=True)

    plt.xticks(rotation=30, ha='right')
    plt.title('Correlation between PageRank score and other variables')
    plt.savefig("corr_figure.png")

def plot_page_score_categories(df_filtered_hubs):
    categories = df_filtered_hubs.columns[5:20]

    # Create matrix for plotting
    matrix = pd.DataFrame({
        "PageRank score": df_filtered_hubs[categories].multiply(df_filtered_hubs["hub_score"], axis=0).sum()})
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(matrix, annot=True, cmap="Blues", cbar_kws={'label': 'Mean Value'}, annot_kws={'rotation': 45})
    plt.title('Sum of PageRank score Score by Category in percent')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("images/png/sum_pagerank_hub_score_percent.png")
    plt.show()

def barplot_page_score_by_category(df_filtered_hubs):
    categories = df_filtered_hubs.columns[5:20]

    # Create matrix for plotting
    matrix = pd.DataFrame({
        "PageRank score": df_filtered_hubs[categories].multiply(df_filtered_hubs["hub_score"], axis=0).sum()}).T

    number_of_articles_order = ['Science', 'Geography','People','History','Everyday life','Design and Technology','Countries',
    'Citizenship','Language and literature','Religion','Music','Business Studies','IT','Mathematics',
    'Art']
    # Plot bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=matrix.columns, y=matrix.iloc[0],order=number_of_articles_order, color="#0f4584")

    plt.title('Sum of PageRank score by Category')
    plt.ylabel('PageRank score')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("images/png/sum_pagerank_hub_score_percent_barchart.png")
    plt.show()

def barplot_page_score_by_category_normalized(df_filtered_hubs):
    categories = df_filtered_hubs.columns[5:20]

    number_articles = df_filtered_hubs[categories].sum(axis=0)
    # Create matrix for plotting
    matrix = pd.DataFrame({
        "PageRank score": df_filtered_hubs[categories].multiply(df_filtered_hubs["hub_score"], axis=0).sum()}).T

    number_of_articles_order = ['Science', 'Geography','People','History','Everyday life','Design and Technology','Countries',
    'Citizenship','Language and literature','Religion','Music','Business Studies','IT','Mathematics',
    'Art']
    # Plot bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=matrix.columns, y=matrix.iloc[0]/number_articles.values,order=number_of_articles_order, color="#0f4584")

    plt.title('Sum of PageRank score Score by Category')
    plt.ylabel('PageRank score normalized by article count')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("images/png/sum_pagerank_hub_score_percent_barchart.png")
    plt.show()

    return matrix/number_articles.values


def hub_network(df_hubs, G):
    # Random generation of positions for articles in 3D space
    num_nodes = len(G.nodes())
    x_vals = np.random.uniform(-0.2, 0.2, num_nodes)  
    y_vals = np.random.uniform(-0.2, 0.2, num_nodes)  
    z_vals = np.random.uniform(-0.2, 0.2, num_nodes)  

    # Get PageRank scores and normalize them for point sizes
    pagerank_scores = df_hubs.set_index("article_names")["hub_score"].to_dict()
    sizes = [pagerank_scores.get(node, 0) * 3000 for node in G.nodes()] 
    colors = [pagerank_scores.get(node, 0) for node in G.nodes()]  


    # Create a Plotly 3D scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='PageRank Score')
        ),
        text=list(G.nodes()),  # Add node names for hover info
        hoverinfo='text'
    ))

    # Update layout for better visualization and display plot
    fig.update_layout(
        title='Interactive 3D Visualization of Articles by PageRank',
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        paper_bgcolor='white',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

def static_hub_3d(df_hubs, G):
    # Random generation of positions for articles in 3D space
    num_nodes = len(G.nodes())
    x_vals = np.random.uniform(-0.2, 0.2, num_nodes)
    y_vals = np.random.uniform(-0.2, 0.2, num_nodes)  
    z_vals = np.random.uniform(-0.2, 0.2, num_nodes)  

    # Get hub scores and normalize them for point sizes
    pagerank_scores = df_hubs.set_index("article_names")["hub_score"].to_dict()
    sizes = [pagerank_scores.get(node, 0) * 3000 for node in G.nodes()] 
    colors = [pagerank_scores.get(node, 0) for node in G.nodes()]  

    # Creating scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x_vals, y_vals, z_vals, c=colors, s=sizes, cmap='viridis', alpha=0.8)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Hub score')

    ax.set_title('3D Visualization of Articles by Hub score')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Show plot
    plt.tight_layout()
    plt.show()
