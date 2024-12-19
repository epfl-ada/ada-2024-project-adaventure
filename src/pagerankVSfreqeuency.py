import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_pageVSfreq_data(data, df_hubs):

    # Making copies of finished and unfinished paths
    pf = data.paths_finished.copy(deep=True)
    pu = data.paths_unfinished.copy(deep=True)

    # Remove source and target from the dataframes
    pf['path'] = pf['path'].apply(lambda x: x[1:-1])
    pu['path'] = pu['path'].apply(lambda x: x[1:])

    # Count total number of times an article has been used in both 
    pf_article_fre = pf['path'].explode().dropna()
    pu_article_fre = pu['path'].explode().dropna()
    article_fre = pd.concat([pf_article_fre, pu_article_fre], 
                            ignore_index=True).value_counts().to_frame().reset_index()

    # Merge to also get pagerank score and category for each article
    freVpr = article_fre.merge(right= df_hubs[['article_names', 'pagerank_score']], 
                                how= 'left',
                                left_on='path', 
                                right_on='article_names').drop(columns=['path'])
    freVpr = freVpr.merge(right= data.categories[['article_name', '1st cat', '2nd cat']], 
                                how= 'left',
                                left_on='article_names', 
                                right_on='article_name').drop(columns=['article_names'])
    freVpr = freVpr.rename(columns={'count': 'user_freq', '1st cat': 'Category', '2nd cat': 'Under Category'})

    # Normalise user_frequency to take into account the probability of stumbling upon that article 
    N  = data.articles.shape[0]
    freVpr['user_freq'] = freVpr.apply(
        lambda x: x['user_freq'] / (freq if (freq := data.links[data.links['2nd article'] == x['article_name']].shape[0]/N) != 0 else np.nan),
        axis=1)

    # Remove NaN values created from spelling mistakes in data.links
    freVpr = freVpr.dropna(subset=['user_freq'])
    return freVpr



def plot_pageVSfreq(freVpr, category= None):
    # Decide what data should be in the plot
    if category is not None:
        freVpr = freVpr[freVpr['Category'] == category]
        categ = freVpr[freVpr['Category'] == category]['Under Category'].unique().tolist()
        color='Under Category'
        category_orders={'Under Category': categ}

    else:
        # Define the order of categories
        categ = ['Science', 'Geography', 'Countries', 'History', 'People', 'Religion', 'Citizenship', 'Everyday life',
                'Design and Technology', 'Language and literature', 'IT', 'Business Studies', 'Music', 'Mathematics', 'Art']
        color='Category'
        category_orders={'Category': categ}
        
    categ
    # Create and plot interactive scatter plot 
    if category is not None:
        fig = px.scatter(
            freVpr,
            x='pagerank_score',
            y='user_freq',
            color=color,
            category_orders=category_orders, 
            hover_data=['article_name'], 
            log_x=True, 
            log_y=True,  
            title='User Frequency vs PageRank Score: ' + category,
            labels={'pagerank_score': 'PageRank Score', 'user_freq': 'User Frequency'}
        )
    else: 
        fig = px.scatter(
            freVpr,
            x='pagerank_score',
            y='user_freq',
            color=color,
            category_orders=category_orders, 
            hover_data=['article_name'], 
            log_x=True, 
            log_y=True,  
            title='User Frequency vs PageRank Score',
            labels={'pagerank_score': 'PageRank Score', 'user_freq': 'User Frequency'}
        )
        
    fig.add_vline(x=freVpr['pagerank_score'].mean(), line_dash="dash", line_color="red")
    fig.add_hline(y=freVpr['user_freq'].mean(), line_dash="dash", line_color="red")
    fig.update_traces(marker=dict(size=4, opacity=1))
    fig.update_xaxes(range=[-4.5, -2])
    fig.update_yaxes(range=[2, 5.7])

    if category == None:
        fig.write_html("pagerank_vs_frequency.html",config={"displayModeBar": False})
    else:
        fig.write_html("pagerank_vs_frequency_" + category + ".html",config={"displayModeBar": False})

    fig.show()

def plot_pageVSfreq_static(freVpr, category=None):
    # Set theme for plot
    sns.set_theme()
    plt.figure(figsize=(12, 8))
    
    # Filter data if a category is provided
    if category is not None:
        freVpr = freVpr[freVpr['Category'] == category]
        color_col = 'Under Category'
        unique_categories = freVpr['Under Category'].unique()
    else:
        color_col = 'Category'
        unique_categories = freVpr['Category'].unique()
    
    # Generate a color palette
    num_colors = len(unique_categories)
    palette = sns.color_palette("tab20", num_colors)
    color_mapping = dict(zip(unique_categories, palette))
    
    # Create a log scaled scatter plot 
    ax = sns.scatterplot(
        data=freVpr,
        x='pagerank_score',
        y='user_freq',
        hue=color_col,
        palette=color_mapping,
        legend='full',
        s=40, 
        alpha=0.7  
    )
    
    # Plot vertical and horizontal lines for the means
    plt.axvline(freVpr['pagerank_score'].mean(), color='red', linestyle='--', linewidth=1)
    plt.axhline(freVpr['user_freq'].mean(), color='red', linestyle='--', linewidth=1)
    
    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add axis labels, title, legends and sets limits on axis
    plt.xlabel('PageRank Score', fontsize=14)
    plt.ylabel('User Frequency', fontsize=14)
    if category is not None:
        plt.title('User Frequency vs PageRank Score: ' + category, fontsize=16)
    else: 
        plt.title('User Frequency vs PageRank Score ', fontsize=16)
    plt.xlim([10**-4.5, 10**-2])
    plt.ylim([10**2, 10**5.7])
    plt.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show the plot
    plt.show()

def get_quadrant_views(freVpr):
    # Add views from metadata to freVpr
    metadata = pd.read_csv('data/metadata.csv')
    freVpr = freVpr.merge(right=metadata[['article_name', 'views']], how='left', on='article_name')

    # Calculates correlation that views has to pagerank score and user frequency
    correlationPR = freVpr['pagerank_score'].corr(freVpr['views'], method='spearman')
    correlationUF = freVpr['user_freq'].corr(freVpr['views'], method='spearman')

    # Max min normalises the data
    freVpr_normalised = freVpr.copy(deep=True)
    freVpr_normalised['pagerank_score'] = (freVpr_normalised['pagerank_score'] - freVpr_normalised['pagerank_score'].min()) / (freVpr_normalised['pagerank_score'].max() - freVpr_normalised['pagerank_score'].min())
    freVpr_normalised['user_freq'] = (freVpr_normalised['user_freq'] - freVpr_normalised['user_freq'].min()) / (freVpr_normalised['user_freq'].max() - freVpr_normalised['user_freq'].min())

    # Calculate boundary for each
    mean_ps = freVpr_normalised['pagerank_score'].mean()
    mean_uf = freVpr_normalised['user_freq'].mean()

    # Calculate the euclidean distance for each point to the mean
    freVpr_normalised['dist_to_mean'] = np.sqrt((freVpr_normalised['pagerank_score'] - mean_ps)**2 + (freVpr_normalised['user_freq'] - mean_uf)**2)

    # Define what article belongs to what quadrant
    conditions = [
        (freVpr_normalised['pagerank_score'] <= mean_ps) & (freVpr_normalised['user_freq'] <= mean_uf),
        (freVpr_normalised['pagerank_score'] <= mean_ps) & (freVpr_normalised['user_freq'] >= mean_uf),
        (freVpr_normalised['pagerank_score'] >= mean_ps) & (freVpr_normalised['user_freq'] <= mean_uf),
        (freVpr_normalised['pagerank_score'] >= mean_ps) & (freVpr_normalised['user_freq'] >= mean_uf)]
    categories = ['lower left', 'upper left', 'lower right', 'upper right']

    # Appoint corresponing quadrant to each article
    freVpr_normalised['quadrant'] = np.select(conditions, categories, default='no quadrant')

    # Keep the 10 biggest outliers in each quadrant
    biggest_outliers = freVpr_normalised.groupby('quadrant', group_keys=False).apply(lambda x: x.nlargest(20, 'dist_to_mean'))

    # Calculate mean of the lower right and upper left quadrants
    quadrant_data = biggest_outliers.groupby(by='quadrant')['views'].mean().reset_index()
    quadrant_data = quadrant_data[(quadrant_data['quadrant'] == 'lower right') | (quadrant_data['quadrant'] == 'upper left')]

    # Create and save plot
    plt.figure(figsize=(10, 6)) 
    sns.barplot(data=quadrant_data, x='quadrant', y='views', color="#0f4584")
    plt.title('Average monthly views of the biggest outliers', fontsize=16)
    plt.xlabel('Quadrant', fontsize=14)
    plt.ylabel('Average monthly views', fontsize=14)

    plt.savefig("views_per_quadrant.png", bbox_inches='tight', dpi=300)
    plt.show()

    print("Spearman correlation between amount of views and user frequency: ", correlationUF)
    print("Spearman correlation between amount of views and pagerank score: ", correlationPR)