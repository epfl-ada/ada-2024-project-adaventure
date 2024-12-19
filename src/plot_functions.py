import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.hubs_analysis import calculate_correlation

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
    plt.xlabel("Hub Score", fontsize=12)
    plt.ylabel("Article Names", fontsize=12)
    plt.title("Top 10 Articles by Hub Score", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.legend().remove()
    plt.tight_layout()
    plt.savefig("top_page_rank_articles.png")
    plt.show()


def get_correlation_matrix_row(df_filtered_hubs, cols):
    df_hubs_corr = df_filtered_hubs[cols].corr()
    df_hubs_corr.rename(columns={'hub_score': 'Hub score'}, inplace=True)
    df_hubs_corr.rename(index={'hub_score': 'Hub score', 
                            'source_counts': 'Source count', 'target_counts': 'Target count',
                            'mean_shortest_path': 'Mean shortest path from article','mean_shortest_path_to_article': 'Mean shortest path to article'}, inplace=True)

    plt.figure(figsize=(12,4))
    corr_figure = sns.heatmap(df_hubs_corr[['Hub score']].T,
                vmin=-1,
                cmap='coolwarm',
                annot=True)

    plt.xticks(rotation=30, ha='right')
    plt.title('Correlation between Hub score and other variables')
    plt.savefig("corr_figure.png")

def plot_page_score_categories(df_filtered_hubs):
    categories = df_filtered_hubs.columns[5:20]

    # Create matrix for plotting
    matrix = pd.DataFrame({
        "Hub score": df_filtered_hubs[categories].multiply(df_filtered_hubs["hub_score"], axis=0).sum()})
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(matrix, annot=True, cmap="Blues", cbar_kws={'label': 'Mean Value'}, annot_kws={'rotation': 45})
    plt.title('Sum of Hub score Score by Category in percent')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("sum_pagerank_hub_score_percent.png")
    plt.show()

def barplot_page_score_by_category(df_filtered_hubs):
    categories = df_filtered_hubs.columns[5:20]

    # Create matrix for plotting
    matrix = pd.DataFrame({
        "Hub score": df_filtered_hubs[categories].multiply(df_filtered_hubs["hub_score"], axis=0).sum()}).T

    number_of_articles_order = ['Science', 'Geography','People','History','Everyday life','Design and Technology','Countries',
    'Citizenship','Language and literature','Religion','Music','Business Studies','IT','Mathematics',
    'Art']
    # Plot bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=matrix.columns, y=matrix.iloc[0],order=number_of_articles_order, color="#0f4584")

    plt.title('Sum of Hub Score by Category')
    plt.ylabel('Hub score')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("sum_pagerank_hub_score_percent_barchart.png")
    plt.show()

def barplot_page_score_by_category_normalized(df_filtered_hubs):
    categories = df_filtered_hubs.columns[5:20]

    number_articles = df_filtered_hubs[categories].sum(axis=0)
    # Create matrix for plotting
    matrix = pd.DataFrame({
        "Hub score": df_filtered_hubs[categories].multiply(df_filtered_hubs["hub_score"], axis=0).sum()}).T

    number_of_articles_order = ['Science', 'Geography','People','History','Everyday life','Design and Technology','Countries',
    'Citizenship','Language and literature','Religion','Music','Business Studies','IT','Mathematics',
    'Art']
    # Plot bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=matrix.columns, y=matrix.iloc[0]/number_articles.values,order=number_of_articles_order, color="#0f4584")

    plt.title('Sum of Hub Score Score by Category')
    plt.ylabel('Hub Score normalized by article count')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("sum_pagerank_hub_score_percent_barchart.png")
    plt.show()

    return matrix/number_articles.values
