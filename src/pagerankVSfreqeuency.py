import plotly.express as px
import pandas as pd
import numpy as np

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

    # Merge to also geat pagerank score and category for each article
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
    fig.update_yaxes(range=[-0.1, 4])

    if category == None:
        fig.write_html("pagerank_vs_frequency.html",config={"displayModeBar": False})
    else:
        fig.write_html("pagerank_vs_frequency_" + category + ".html",config={"displayModeBar": False})

    fig.show()