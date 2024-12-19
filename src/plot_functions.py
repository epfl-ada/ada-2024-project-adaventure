import matplotlib.pyplot as plt
import seaborn as sns

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

    # Show the combined plot
    plt.show()