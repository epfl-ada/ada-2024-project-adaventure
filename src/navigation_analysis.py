import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from urllib.parse import unquote
from matplotlib.patches import Polygon

def get_category_navigation_matrix(data):
    # create a matrix of the subjects with all 0.
    categories = data.categories['1st cat'].unique()
    subject_connections = pd.DataFrame(0, index=categories, columns=categories)
    # get all the played paths
    paths = pd.concat([data.paths_finished["path"], data.paths_unfinished["path"]])
    # Create a mapping from article to subject
    #make this a dict from dataframes["categories"]["article"] to dataframes["categories"][1]
    article_to_subject = data.categories.set_index('article_name')["1st cat"].to_dict()

    # Iterate through each path and update subject connections
    # turn numpy before for faster runtime
    subjects = {subject: idx for idx, subject in enumerate(subject_connections.index)}
    subject_connections_df = subject_connections.to_numpy()
    for path in paths:
        prevSubject = None
        for article in path:
            nextSubject = article_to_subject.get(article)
            if prevSubject in subjects and nextSubject in subjects:
                subject_connections_df[subjects.get(prevSubject), subjects.get(nextSubject)] += 1
            prevSubject = nextSubject
    # return to pandas dataframe
    subject_connections = pd.DataFrame(subject_connections_df, index=subject_connections.index, columns=subject_connections.columns)
    return subject_connections, subjects

def plot_category_connections(subject_connections, subject_count, subjects):
    # Get the maximum and minimum weights, with or without the diagonal
    max_weight = subject_connections.values[~np.eye(subject_connections.shape[0], dtype=bool)].max()
    max_weight_diag = subject_connections.values[np.eye(subject_connections.shape[0], dtype=bool)].max()

    # Create a color palette
    color_map = sns.color_palette("flare_r", len(subjects)+7)[7:]
    colormap = mcolors.ListedColormap(plt.cm.viridis(np.linspace(0.2, 1, 256)))
    norm = mcolors.Normalize(vmin=0, vmax=max_weight)
    colormap_diag = mcolors.LinearSegmentedColormap.from_list("YellowRed", ["yellow", "red"])
    norm_diag = mcolors.Normalize(vmin=0, vmax=max_weight_diag)

    # Create positions for 16 points arranged in a circle
    angle = np.linspace(0, 2 * np.pi, len(subjects), endpoint=False)
    positions = {subject: (x, y) for (x, y), subject in zip(np.c_[np.cos(angle), np.sin(angle)], subject_connections.index)}

    # initiate plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal', adjustable='box') # make it Carré. Pratique. Gourmand!
    plt.xticks([]) # removing the irrelevant axis labels
    plt.yticks([])

    max_width = 0.15
    self_loop_radius = 0.18
    arrow_length = 0.1  # Length of the arrow pointing towards the center (adjust as needed)
    # max_weight, min_weight = 0, 0
    # Draw the connections between subjects
    for i, subject1 in enumerate(subjects):
        for j, subject2 in enumerate(subjects):
            if i < j:
                continue
            weight = subject_connections.loc[subject1, subject2]
            line_color = colormap(norm(weight))
            line_width = weight / 1000  # Scale line width for better visualization
            # max_weight = max(max_weight, weight)
            # min_weight = min(min_weight, weight)
            if weight > 0 and i != j:
                ax.plot([positions[subject1][0], positions[subject2][0]], [positions[subject1][1], positions[subject2][1]], color=line_color, lw=line_width, zorder=1)
            elif weight > 0 and i == j:
                (x, y) = positions[subject1]
                line_color = colormap_diag(norm_diag(weight))
                line_width = line_width / 100  # Convert line width to points
                # Calculate direction towards the center
                dx = x/np.sqrt(x**2 + y**2)
                dy = y/np.sqrt(x**2 + y**2)
                # Arrowhead position
                arrow_tip_x = x + (np.sqrt(subject_count[subject1]) / 180) * dx
                arrow_tip_y = y + (np.sqrt(subject_count[subject1]) / 180) * dy
                
                # Calculate the base points of the triangle
                base_x1 = arrow_tip_x - line_width/2 * dy + arrow_length * dx
                base_y1 = arrow_tip_y + line_width/2 * dx + arrow_length * dy
                base_x2 = arrow_tip_x + line_width/2 * dy + arrow_length * dx
                base_y2 = arrow_tip_y - line_width/2 * dx + arrow_length * dy

                # Calculate the distance between base 1 and base 2
                distance = np.sqrt((base_x2 - base_x1)**2 + (base_y2 - base_y1)**2)*1000
                # Create and draw the triangle
                arrow_triangle = Polygon(
                    [[arrow_tip_x, arrow_tip_y], [base_x1, base_y1], [base_x2, base_y2]],
                    closed=True, color=line_color, zorder=3
                )
                ax.add_patch(arrow_triangle)

                #wanna do loops instead of arrows? uncomment the following code
                # line_width = line_width / 5  # Convert line width 

                # # Calculate the direction for the loop placement
                # (x, y) = positions[subject1]
                # angle_offset = np.arctan2(y, x)  # Angle from the origin to the blob
                # loop_x = x + self_loop_radius * np.cos(np.linspace(0, 2 * np.pi, 100)) + self_loop_radius * np.cos(angle_offset) * 1.2
                # loop_y = y + self_loop_radius * np.sin(np.linspace(0, 2 * np.pi, 100)) + self_loop_radius * np.sin(angle_offset) *1.2
                # ax.plot(loop_x, loop_y, color=line_color, lw=line_width, zorder=1)

    # draw the blobs
    for i, subject1 in enumerate(subjects):
        (x,y) = positions[subject1]
        color = mcolors.to_hex(color_map[i])
        circle = plt.Circle((x, y), np.sqrt(subject_count[subject1]) / 180, color=color, ec=None, lw=1.5, zorder=2)
        ax.add_patch(circle)
        plt.text(x, y, f"{subject1}\n{subject_count[subject1]}", ha='center', va='center', fontsize=8, color='black')

    # add color bar 1
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Trans-category connection Weight')
    # add color bar 2
    sm = plt.cm.ScalarMappable(cmap=colormap_diag, norm=norm_diag)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Self Weight')

    plt.title("Subject Connections with Proportional Blob Sizes and Connection Weights")
    plt.show()