import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np

from src.preprocessing import *
from src.similarity import *
from src.shortest_paths import *
from src.sum_graph import *
from src.hubs_analysis import *
from src.unfinished_games import *
from src.common_paths import *
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots



def fig_Heatmap_bert(data):
    start = "Calculus"
    end = "Paul McCartney"
    matrix_sim = get_sim_matrix(data,start,end,distance_Bert)
    mean = np.mean(matrix_sim)
    print("Mean distance for game from ",start," to ",end, " : ",mean)

    # Create heatnmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix_sim,
        colorscale='Blues',  
        colorbar=dict(title="Distance"),
        zmin=0,  
        zmax=1,
        xaxis='x',  
        yaxis='y',  
        hovertemplate='1st Path: %{x}<br>2nd Path: %{y}<br>Distance: %{z}<extra></extra>'  

    ))

    fig.update_layout(
        title='Semantic distance for different paths<br> for game from "Calculus" to "Paul McCartney"',
        title_font=dict(
            size=15  
        ),
        title_x=0.5,  
        title_xanchor='center',  
        xaxis=dict(
            title='1st path',
            tickmode='linear',
            ticks='outside',
            ticklen=5,
            showgrid=False  
        ),
        yaxis=dict(
            title='2nd path',
            tickmode='linear',
            ticks='outside',
            ticklen=5,
            showgrid=False  
        ),
        width=550,  
        height=500, 
        margin=dict(l=20, r=20, t=60, b=40)  
    )
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    # Save heatmap as HTML
    fig.write_html('heatmap_distance.html',config={"displayModeBar": False})


    fig.show(config={"displayModeBar": False})



def fig_top10_articles(data):
    dict_articles = get_most_visited_articles(data)
    top_10 = sorted(dict_articles.items(), key=lambda x: x[1], reverse=True)[:10]
    articles, visits = zip(*top_10)

    categories = [data.get_article_category(article,"1st cat") for article in articles]
    unique_categories = list(set(categories))
    cmap = pc.qualitative.Prism
    color_map = {category: cmap[i] for i, category in enumerate(unique_categories)}
    
    fig = go.Figure()
    for  category in unique_categories:
        indices = [i for i, cat in enumerate(categories) if cat == category]
        fig.add_trace(go.Bar(
            x=[articles[i] for i in indices],
            y=[visits[i] for i in indices],
            name=category,
            marker=dict(color=color_map[category]),
        ))
        
    fig.update_layout(
        title="Top 10 most visited articles",
        title_x=0.5,  
        xaxis_title="Articles",
        yaxis_title="Number of visits during games",
        template="plotly_white",  
        showlegend=True,
        xaxis={'categoryorder':'total descending'}

    )
    fig.update_xaxes(tickangle=0)
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    fig.write_html("top_10_articles.html",config={"displayModeBar": False})

    fig.show(config={"displayModeBar": False})


def fig_top10_links(data):
    dict_links = get_paths_oflength_k(data, 2)
    top_10 = sorted(dict_links.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10 = {f"{k[0]} -> {k[1]}": v for k, v in top_10}
    articles, visits = zip(*top_10.items())
    fig = go.Figure(go.Bar(
        y=articles,    
        x=visits,      
        orientation='h',  
        marker=dict(color='darkblue'),
        text=visits,   
        textposition='auto', 
    ))

    fig.update_layout(
        title="Top 10 most visited links",
        title_x=0.5, 
        xaxis_title="Number of visits during games",
        yaxis_title="Link",
        template="plotly_white",  
        showlegend=False
    )
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    fig.write_html("top_10_links_visites.html",config={"displayModeBar": False})

    fig.show(config={"displayModeBar": False})


def fig_top_cat(data):
    dict_categories = get_most_visited_categories(data, "1st cat")


    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="coolwarm_r",  
        max_words=200,
        contour_color='black',  
        contour_width=2,  
    ).generate_from_frequencies(dict_categories)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Category Importance", fontsize=12)

    plt.savefig("wordcloud_categories.png", bbox_inches="tight")
    plt.show()

def fig_success_rate_heatmap(data):
    """
    Plots a heatmap of success rates by (cat_start, cat_target).
    
    Parameters:
    success_rate (dict): A dictionary where keys are (cat_start, cat_target) pairs 
                         and values are success rates.
    """

    success_rate = success_rate_category_pair(data)
    df = pd.DataFrame(list(success_rate.items()), columns=['Category Pair', 'Success Rate'])
    df[['Start Category', 'Target Category']] = pd.DataFrame(df['Category Pair'].tolist(), index=df.index)
    df.drop(columns=['Category Pair'], inplace=True)

    heatmap_data = df.pivot(index='Start Category', columns='Target Category', values='Success Rate')

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,  
            x=heatmap_data.columns,  
            y=heatmap_data.index,  
            colorscale='Blues',  
            colorbar=dict(title="Success Rate", titleside='right'),
            hovertemplate='Target: %{x}<br>Start: %{y}<br>Sucess rate: %{z}<extra></extra>'
        )
    )

    fig.update_layout(
        title="Success Rate by (Start Category, Target Category)",
        title_x=0.5,  # Centrer le titre
        xaxis_title="Target Category",
        yaxis_title="Start Category",
        template="plotly_white",
        height=600,
        width=800
    )
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    fig.write_html("success_rate_heatmap.html",config={"displayModeBar": False})
    
    fig.show(config={"displayModeBar": False})



def fig_to_target(data):
    clicks_to_target = data.paths_unfinished['clicks_to_target']
    click_counts = pd.Series(clicks_to_target).value_counts().sort_index()
    distance_to_target = data.paths_unfinished['distance_to_target']

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Number of clicks missing to reach the target", "Semantic distance to the target"))
    fig.add_trace(go.Bar(x=click_counts.index,y=click_counts.values, name="Number of clicks"), row=1, col=1)
    fig.add_trace(go.Histogram(x=distance_to_target, name="Distance to target"), row=1, col=2)
    
    fig.update_layout(
    title="Distance from stopping point to target article",
    title_x=0.5,  
    showlegend=False,  
    template="plotly_white",  
    xaxis_title="Number of clicks",  
    yaxis_title="Number of unfinished games",  
    xaxis2_title="Distance to target", 
    yaxis2_title="Number of unfinished games",
    )
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    fig.write_html("dist_to_target.html",config={"displayModeBar": False})

    fig.show(config={"displayModeBar": False})

def get_stoptarget (data):
    dict ={}
    for i in tqdm(range(len(data.paths_unfinished))):
        clicks = data.paths_unfinished.loc[i,"clicks_to_target"]
        distance = data.paths_unfinished.loc[i,"distance_to_target"]
        if clicks ==2 and distance >0.5:
            stop_point = data.paths_unfinished.loc[i,"stop_point"]
            target = data.paths_unfinished.loc[i,"target"]
            if (stop_point,target) in dict:
                dict[(stop_point,target)] += 1
            else:
                dict[(stop_point,target)] = 1
    return dict

def sankey(data):
    dict_pairs = get_stoptarget(data)
    dict_common = {k: v for k, v in dict_pairs.items() if v >= 10}
    dict_c ={}
    for articles in dict_common.keys():
        stop = articles[0]
        target = articles[1]
        if stop in data.categories["article_name"].values and target in data.categories["article_name"].values:
            cat_stop = data.get_article_category(stop,"1st cat")
            cat_target = data.get_article_category(target,"1st cat")
            if (cat_stop,cat_target) in dict_c:
                dict_c[(cat_stop,cat_target)][0] += 1
                dict_c[(cat_stop,cat_target)][1].append(f"{articles[0]}->{articles[1]}")
            else:
                dict_c[(cat_stop,cat_target)] = [1,[f"{articles[0]}->{articles[1]}"]]

    categories_start = list(set(cat_start for cat_start, _ in dict_c.keys()))
    categories_end = list(set(cat_end for _, cat_end in dict_c.keys()))

    categories = categories_start + categories_end

    category_to_index = {}
    index = 0
    for cat in categories_start:
        category_to_index[f"s_{cat}"] = index
        index += 1

    for cat in categories_end:
        if f"e_{cat}" not in category_to_index:
            category_to_index[f"e_{cat}"] = index
            index += 1

    sources = [category_to_index[f"s_{cat_start}"] for cat_start, _ in dict_c.keys()]
    targets = [category_to_index[f"e_{cat_end}"] for _, cat_end in dict_c.keys()]
    values = [dict_c[cat][0] for cat in dict_c.keys()]
    links= [dict_c[cat][1] for cat in dict_c.keys()]

    c = pc.sequential.Blues[2:]
    colors = [c[s] for s in sources]



    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[f"{cat[2:]}" for cat in category_to_index.keys()],
            color = c[2]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label = links,
            color = colors
        )
    ))
    

    fig.update_layout(title_text="Commonly unkown connections between stop point and target",
                      font_size=10)
    fig.write_html("sankey.html",config={"displayModeBar": False})

    fig.show(config={"displayModeBar": False})


