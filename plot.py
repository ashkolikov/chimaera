import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.ndimage import rotate, zoom, gaussian_filter
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as PathEffects

def get_2d(array):
    if len(array.shape) == 4:
        return array[0, ..., 0]
    if len(array.shape) == 3:
        if array.shape[0] == array.shape[1]:
            return array[..., 0]
        else:
            return array[0, ...]
    return array

def plot_map(map, ax=None, show=True, hic_cmap='Reds'):
    ax = ax if ax is not None else plt
    ax.imshow(get_2d(map), cmap = hic_cmap, interpolation = 'none')
    ax.axis('off')
    if show:
        plt.show()

def plot_results(y_pred, y_true, save = False, title = None):
    fig, ax = plt.subplots(3, 8, figsize = (14, 7))
    for n in range(len(y_pred)):
        y_pos = n // 3
        x_pos = 2 * n % 6
        x_pos = x_pos + x_pos // 2
        plot_map(y_true[n], ax = ax[y_pos, x_pos], show = False)
        plot_map(y_pred[n], ax = ax[y_pos, x_pos + 1], show = False)
        ax[y_pos, x_pos].set_title('True')
        ax[y_pos, x_pos + 1].set_title('Predicted')
    for i in ax.flat:
        i.axis('off')
        i.title.set_fontsize(15)
    if title is not None:
        fig.suptitle(title, fontsize=18).set_y(0.95)
        fig.subplots_adjust(top=0.85)
    if save:
        plt.savefig(save)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()

def plot_pca(
                y_pred_train,
                y_pred_val,
                y_true_train,
                y_true_val,
                transformed_background,
                save = None,
                colors = ('blue', 'purple', 'orange', 'red')):
    
    fig = plt.figure(figsize = (7, 5))
    plt.scatter(*transformed_background, c='grey', alpha=0.3)
    plt.scatter(*y_pred_train, c = colors[0])
    plt.scatter(*y_true_train, c = colors[1])
    plt.scatter(*y_pred_val, c = colors[2])
    plt.scatter(*y_true_val, c = colors[3])
    plt.legend(['Other', 'Train pred', 'Train true', 'Val pred', 'Val true'])

    for i,j in zip(y_true_train.T, y_pred_train.T):
        plt.plot(*np.stack((i, j)).T, c = colors[0], alpha = 0.3)
    for i,j in zip(y_true_val.T, y_pred_val.T):
        plt.plot(*np.stack((i, j)).T, c = colors[2], alpha = 0.3)

    if save:
        x_min = min(transformed_background[0].min(), y_true_train[0].min(), y_true_val[0].min()) - 0.5
        x_max = max(transformed_background[0].max(), y_true_train[0].max(), y_true_val[0].max()) + 0.5
        y_min = min(transformed_background[1].min(), y_true_train[1].min(), y_true_val[1].min()) - 0.5
        y_max = max(transformed_background[1].max(), y_true_train[1].max(), y_true_val[1].max()) + 0.5
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(save)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()

def light_cmap(a):
    N = 256
    vals = np.ones((N, 4))
    b = 256-a
    vals[:a, 0] = np.linspace(1,1, a)
    vals[:a, 1] = np.linspace(1,0, a)**0.5
    vals[:a, 2] = np.linspace(1,0, a)**0.5
    vals[a:, 0] = np.linspace(1, 0, b)**2
    vals[a:, 1] = np.zeros(b)
    vals[a:, 2] = np.zeros(b)
    return matplotlib.colors.ListedColormap(vals)

def dark_cmap(a):
    N = 256
    vals = np.ones((N, 4))
    b = 256-a
    vals[:a, 0] = np.linspace(0,0.1, a)
    vals[:a, 1] = np.linspace(0,0, a)
    vals[:a, 2] = np.linspace(0,0.3, a)
    vals[a:, 0] = np.linspace(0.1, 1, b)
    vals[a:, 1] = np.linspace(0, 0, b)
    vals[a:, 2] = np.linspace(0.3, 0.1, b)
    return matplotlib.colors.ListedColormap(vals)


def plot_filter_analisis(pred, 
                         y_pred, 
                         hic, 
                         filters,
                         theme = 'dark', 
                         color_shifts = {'heatmap': 50, 'filters': 200}):

    cmap = dark_cmap if theme == 'dark' else light_cmap
    hic_cmap = 'gist_heat_r' if theme == 'dark' else 'Reds'
    fig = plt.figure(figsize=(20,18))
    gs1 = fig.add_gridspec(nrows=94, ncols=83, left=0.01, right=0.58)

    ax1 = fig.add_subplot(gs1[28:70, :80])
    ax2 = fig.add_subplot(gs1[28:70, 81:])

    ax3 = fig.add_subplot(gs1[7:28, :80])
    ax4 = fig.add_subplot(gs1[70:91, :80])


    gs2 = fig.add_gridspec(nrows=5, ncols=2, left=0.62, right=0.98, top=0.8)
    axes = []
    for i in range(5):
        for j in range(2):
            axes.append(fig.add_subplot(gs2[i, j]))

    n_filters = len(pred)
    ax1.imshow(pred, aspect='auto', interpolation='none', cmap=cmap(color_shifts['heatmap']))
    ax1.set_yticks(np.arange(n_filters//2)*2)
    ax1.set_xticks(())

    outliers = (pred.max(axis=1)-pred.mean(axis=1))[...,None]
    ax2.imshow(outliers, aspect='auto', cmap=cmap(color_shifts['heatmap']))
    ax2.tick_params(right=True, bottom=False, left=False, labelright=True, labelleft=False, labelbottom=False)
    ax2.set_yticks(np.arange(n_filters//2)*2)

    zoom_rate = 256 // len(hic)
    hic = zoom(hic, (zoom_rate, zoom_rate))
    hic = rotate(hic, 45, mode='constant', cval=hic.min())
    hic = hic[:len(hic) // 2 + 1]
    ax3.imshow(hic, aspect='auto', cmap=hic_cmap)
    ax3.axis('off')

    zoom_rate = 256 // len(y_pred)
    y_pred = zoom(y_pred, (zoom_rate, zoom_rate))
    y_pred = rotate(y_pred, 45, mode='constant', cval=np.min(y_pred))
    y_pred = y_pred[len(y_pred) // 2:]
    ax4.imshow(y_pred, aspect='auto', cmap=hic_cmap)
    ax4.axis('off')

    best_filters = np.argsort(pred.max(axis=1) - pred.mean(axis=1))[::-1][:10]
    for i, number in enumerate(best_filters):
        axes[i].imshow(filters[number], cmap=cmap(color_shifts['filters']))
        axes[i].axis('off')
        axes[i].set_title(number)

    plt.show()
    return filters[best_filters]

def plot_filters(filters, figsize = (16, 10), cmap = 'coolwarm', normalize=False):
    a = len(filters) // 8
    fig,ax=plt.subplots(a, 8, figsize=figsize)
    vmin, vmax = filters.min(), filters.max()
    for n,i in enumerate(filters):
        if not normalize:
            ax[n//8,n%8].imshow(i, cmap=cmap)
        else:
            ax[n//8,n%8].imshow(i, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[n//8,n%8].set_title(n)
        ax[n//8,n%8].axis('off')
    plt.tight_layout()
    plt.show()

def plot_chromosomes(chroms, train, val, spare, test=None):
    fig, axs = plt.subplots(1, len(chroms), figsize = (20,5))
    chrom_sets = []
    for i in train:
        axs[i[0]].plot

    
def plot_latent_mapping(maps, latent, background):
    pass

def plot_mutations(maps, titles):
    pass

def plot_score(metric_name,
                   r_train,
                   r_train_negative_control,
                   r_val,
                   r_val_negative_control,
                   r_test,
                   r_test_negative_control,
                   train_val_names,
                   test_names,
                   kind = 'violin',
                   cmap = 'plasma'):
    train_val_names = ', '.join(train_val_names)
    if test_names:
        test_names = ', '.join(test_names)

    df = pd.DataFrame(columns=['Correlation coefficient',
                            'Sample',
                            'Order'])
    data = [r_train,
            r_train_negative_control,
            r_val,
            r_val_negative_control,
            r_test,
            r_test_negative_control]
    for i, values, in enumerate(data):
        if values is None:
            continue
        order = 'correct' if not i % 2 else 'permuted'
        if i in [0,1]:
            sample = 'train'
        elif i in [2,3]:
            sample = 'val'
        else:
            sample = 'test'
        data_ = np.array([values, [sample]*len(values), [order]*len(values)]).T
        new_df = pd.DataFrame(data= data_, columns=df.columns)
        df = df.append(new_df, ignore_index=True)
    df = df.astype({'Correlation coefficient': float})

    plt.figure(figsize=(16,9))
    if kind == 'violin':
        box_plot = sns.violinplot(x="Sample", y="Correlation coefficient", hue="Order", data=df, palette=cmap, inner=None)
    else:
        box_plot = sns.boxplot(x="Sample", y="Correlation coefficient", hue="Order", data=df, palette=cmap)
    Means = df.groupby(['Sample', 'Order'])['Correlation coefficient'].mean()

    for i, xtick in enumerate(box_plot.get_xticklabels()):
        xtick = xtick.get_text()
        txt = box_plot.text(i-0.2, Means[xtick]['correct'], 
                    f"Mean:\n{Means[xtick]['correct']:.2f}", 
                    horizontalalignment='center', color='w', weight='semibold', fontsize=15)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
        txt = box_plot.text(i+0.2, Means[xtick]['permuted'],
                    f"Mean:\n{Means[xtick]['permuted']:.2f}", 
                    horizontalalignment='center', color='w', weight='semibold', fontsize=15)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    plt.title(metric_name.capitalize()+' correlations')
    plt.show()
            
