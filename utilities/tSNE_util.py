import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tSNE(ax, x, y, fig_path='t-SNE.png'):
    """Visualize high-dimensional data using tSNE."""

    x_embedded = TSNE(n_components=2).fit_transform(x)
    df = pd.DataFrame()
    df['sne-1'] = x_embedded[:, 0]
    df['sne-2'] = x_embedded[:, 1]
    df['cuisine'] = y

    classes = np.unique(y)
    
    sns.scatterplot(
        ax=ax,
        x='sne-1',
        y='sne-2',
        hue='cuisine',
        palette=sns.color_palette('hls', len(classes)),
        data=df,
        legend='full',
        alpha=0.5
    )
    
    # add text for each cluster
    txts = []
    for i in classes:
        cx, cy = np.median(x_embedded[y == i, :], axis=0)
        txt = ax.text(cx, cy, i)
        txts.append(txt)

    plt.savefig(fig_path)
    return df, ax, txts