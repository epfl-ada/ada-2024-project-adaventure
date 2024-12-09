import matplotlib.pyplot as plt

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