import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.ndimage import rotate, zoom, gaussian_filter
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from scipy import stats

def get_2d(array):
    if len(array.shape) == 4:
        return array[0, ..., 0]
    if len(array.shape) == 3:
        return array[..., 0]
    return array


def hic_cmap():
    N = 300
    vals = np.ones((N, 4))
    colors = (230,230,250), (90,150,210), (140,60,60), (90,20,20)
    zones = [0, 100, 300-100, 300]
    for i in range(3):
        for j in range(3):
            vals[zones[i]:zones[i+1], j] = np.linspace(colors[i][j]/256,
                                                       colors[i+1][j]/256,
                                                       zones[i+1]-zones[i])
    return matplotlib.colors.ListedColormap(vals)

def plot_map(map, ax=None, show=True, hic_cmap="hic_cmap", name=None, colorbar=False, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(get_2d(map), cmap = hic_cmap, interpolation = 'none', **kwargs)
    ax.set_yticks([])
    ax.set_xticks([])
    if colorbar:
        annotate_colorbar(im, ax)
    if name:
        ax.set_ylabel(name)
    if show:
        plt.show()



def plot_results(y_pred, y_true, sample=None, numbers=None, data=None,
                 save = False, equal_scale=False):

    fig = plt.figure(figsize=(17,8))
    gs1 = fig.add_gridspec(nrows=111, ncols=110, left=0, right=1, top=1, bottom=0)
    for i in range(3):
        for j in range(3):
            ax1 = fig.add_subplot(gs1[i*40:i*40+15, j*37:j*37+30])
            ax2 = fig.add_subplot(gs1[i*40+15:i*40+30, j*37:j*37+30])
            y1 = y_true[i*3+j]
            y2 = np.flip(y_pred[i*3+j],axis=0)
            if equal_scale:
                vmin=min(y1.min(), y2.min())
                vmax=max(y1.max(), y2.max())
            else:
                vmin=None
                vmax=None
            if data is None:
                plot_map(y1, ax=ax1, show=False,
                         vmin=vmin, vmax=vmax, name='True')
                plot_map(y2, ax=ax2, show=False,
                         vmin=vmin, vmax=vmax, name='Pred')
                ax1.axis('off')
                ax2.axis('off')
            else:
                data.plot_annotated_map(sample,
                                        numbers[i*3+j],
                                        ax=ax1,
                                        y=y1,
                                        axis=None,
                                        show_position=True,
                                        colorbar=not equal_scale,
                                        full_name=False,
                                        vmin=vmin,
                                        vmax=vmax,
                                        name='True',
                                        show=False)
                data.plot_annotated_map(sample,
                                        numbers[i*3+j],
                                        ax=ax2,
                                        y=y2,
                                        axis=None,
                                        show_position=False,
                                        colorbar=not equal_scale,
                                        full_name=False,
                                        vmin=vmin,
                                        vmax=vmax,
                                        name='Pred',
                                        show=False)
    if save:
        plt.savefig(save)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()

    
def plot_motiff_effect(numbers, sites, names):
    sample_name = []
    x = []
    y = []
    for site, name in zip(sites, names):
        sample_name += [name] * np.prod(site.shape)    
        x.append(np.tile(numbers, site.shape[0]))
        y.append(site.flatten())
    x = np.concatenate(x)
    y = np.concatenate(y)
    df = pd.DataFrame({'Number of inserted sites': x,
                       'Mut signal - wt signal': y,
                       'Sample': sample_name})
    plt.figure(figsize=(13,6))
    plt.rcParams.update({'font.size': 15})
    sns.violinplot(x='Number of inserted sites', y='Mut signal - wt signal', data=df,
                   hue='Sample', palette='hic_cmap', scale='width')
    plt.legend(loc='lower left')
    plt.rcParams.update({'font.size': 12})
    plt.show()


def plot_filter_analisis(pred, 
                         y_pred, 
                         y_true, 
                         filters,
                         theme = 'dark', 
                         color_shifts = {'heatmap': 50, 'filters': 200}):

    cmap_hic = "hic_cmap"
    fig = plt.figure(figsize=(20,13))
    gs1 = fig.add_gridspec(nrows=80, ncols=83, left=0.01, right=0.75)

    ax1 = fig.add_subplot(gs1[15:65, :80])
    ax2 = fig.add_subplot(gs1[15:65, 81:])

    ax3 = fig.add_subplot(gs1[0:15, :80])
    ax4 = fig.add_subplot(gs1[65:80, :80])


    gs2 = fig.add_gridspec(nrows=8, ncols=1, left=0.8, right=0.98)
    axes = []
    for i in range(8):
        axes.append(fig.add_subplot(gs2[i]))

    n_filters = len(pred)
    ax1.imshow(pred, aspect='auto', interpolation='none', cmap=cmap_hic)
    ax1.set_yticks(np.arange(n_filters//2)*2)
    ax1.set_xticks(())

    outliers = (pred.max(axis=1)-pred.mean(axis=1))[...,None]
    ax2.imshow(outliers, aspect='auto', cmap=cmap_hic)
    ax2.tick_params(right=True, bottom=False, left=False, labelright=True,
                    labelleft=False, labelbottom=False)
    ax2.set_yticks(np.arange(n_filters//2)*2)

    ax3.imshow(get_2d(y_true), aspect='auto', cmap=cmap_hic)
    ax3.axis('off')

    ax4.imshow(np.flip(get_2d(y_pred),axis=0), aspect='auto', cmap=cmap_hic)
    ax4.axis('off')

    best_filters = np.argsort(pred.max(axis=1) - pred.mean(axis=1))[::-1][:8]
    for i, number in enumerate(best_filters):
        axes[i].imshow(filters[number], cmap=cmap_hic)
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


def plot_score(metric_name,
               correct,
               permuted,
               x,
               best_only):
    
    
    permuted = np.array(permuted)
    correct = np.array(correct)
    if best_only:
        fig, ax = plt.subplots(1,1,figsize=(4,6))

        means = correct.mean(axis=0)
        best = correct[:, np.argmax(means)]
        control = permuted[:, np.argmax(means)]
        x = x[np.argmax(means)]
        y_name = f'{metric_name.capitalize()} correlation'
        x_name = ''
        n = len(best)
        sns.violinplot(data=pd.DataFrame({x_name: ['Predictions']*n + ['Control']*n,
                                    y_name: np.concatenate([best, control])}),
                       x=x_name, y=y_name, palette='hic_cmap', ax=ax)
        ax.set_ylim(-1.2,1.2)
        txt = ax.text(0.1,
                best.mean(),
                f'''Mean = {best.mean():.2f}
Median = {np.median(best):.2f}
P-value = {stats.ttest_1samp(best, 0).pvalue:.2f}''',
                ha='center', 
                color='w',
                weight='semibold',
                fontsize=14)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2,
                                                            foreground='black')])
        txt = ax.text(0.9,
                control.mean(),
                f'''Mean = {control.mean():.2f}
Median = {np.median(control):.2f}
P-value = {stats.ttest_1samp(control, 0).pvalue:.2f}''',
                ha='center', 
                color='w',
                weight='semibold',
                fontsize=14)
        ax.set_title(f'Correlations between true and predicted contacts\nat the distance with best score ({int(x)} bp)')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2,
                                                            foreground='black')])
    else:
        fig, ax = plt.subplots(1,1,figsize=(20,6))

        x = np.tile(x[1:], len(correct)).astype(int)
        y_name = f'{metric_name.capitalize()} correlation'
        x_name = 'Genomic distance'
        sns.boxplot(data=pd.DataFrame({x_name: x,
                                        y_name: correct.flat}),
                    x=x_name, y=y_name,
                    color='#55a0f0', ax=ax)
        sns.boxplot(data=pd.DataFrame({x_name: x,
                                        y_name: permuted.flat}),
                    x=x_name, y=y_name,
                    color='#8b288d', ax=ax)
        ax.set_ylim(-1.1,1.1)
        for i in range(len(x)-1):
            txt = ax.text(i, correct[:,i].mean(), 
                            f'{correct[:,i].mean():.2f}', ha='center', 
                            color='w', weight='semibold', fontsize=10)
            txt.set_path_effects([PathEffects.withStroke(linewidth=2,
                                                            foreground='black')])
        for i in range(len(x)-1):
            txt = ax.text(i, permuted[:,i].mean(),
                            f'{permuted[:,i].mean():.2f}', ha='center',
                            color='w', weight='semibold', fontsize=10)
            txt.set_path_effects([PathEffects.withStroke(linewidth=2,
                                                            foreground='black')])
        plt.xticks(rotation = 90)
    plt.show()

def plot_attention_analysis(mha_mtx, q_sum, k_sum, coords,
                            y1, y2, eps_power=10, log=True):
        fig = plt.figure(figsize=(14,14))
        gs1 = fig.add_gridspec(nrows=800, ncols=800, left=0, right=1, top=1, bottom=0)
        ax1 = fig.add_subplot(gs1[:100, 200:600])
        ax2 = fig.add_subplot(gs1[100:200, 200:600])
        ax3 = fig.add_subplot(gs1[200:600, 200:600])
        ax4 = fig.add_subplot(gs1[600:800, 200:600])
        ax5 = fig.add_subplot(gs1[200:600, 0:100])
        ax6 = fig.add_subplot(gs1[200:600, 100:200])
        ax7 = fig.add_subplot(gs1[200:600, 600:800])

        epsilon = 10**(-eps_power)
        if log:
            mtx = np.log10(mha_mtx+epsilon)
        else:
            mtx = mha_mtx
        ax3.imshow(mtx, interpolation='none', cmap='inferno')
        ax3.axis('off')
        plot_map(y1, ax=ax1, show=False)
        plot_map(np.flip(y2, axis=0), ax=ax2, show=False)
        plot_map(y1.T, ax=ax5, show=False)
        plot_map(np.flip(y2, axis=0).T, ax=ax6, show=False)
        ax4.plot(coords, q_sum, c='#882211')
        ax4.set_xlim(coords[0], coords[-1])
        ax7.plot(k_sum, coords, c='#882211')
        ax7.set_ylim(coords[0], coords[-1])
        ax7.invert_xaxis()
        ax7.invert_yaxis()
        ax7.yaxis.tick_right()
        plt.show()         

   
def annotate(ax, start, end, axis='both', h=32, w=128, constant=0, position='bottom'):
    if not axis:
        ax.set_xticks([])
        ax.set_yticks([])
        return 0.1, 0.1
    if axis != 'y':
        if end > 99999:
            x = [f'{i / 1000:.1f} kb' for i in np.linspace(start, end, 16)]
        else:
            x = [f'{int(i):,}' for i in np.linspace(start, end, 16)]
        l = np.linspace(0, w, len(x))
        ax.set_xticks(l)
        ax.set_xticklabels(x, rotation = 90)
        if position == 'top':
            ax.xaxis.set_ticks_position('top')
    if axis != 'x':
        y_max = int((end-start) * (2*h/w))
        if end > 99999:
            y = reversed([f'{i / 1000:.1f} kb' for i in np.linspace(y_max / h, y_max, 5)])
        else:
            y = reversed([f'{int(i):,}' for i in np.linspace(y_max / h, y_max, 5)])
        ax.set_yticks(np.linspace(-.5, h-.5, 5))
        ax.set_yticklabels(y)
    if axis == 'x':
        ax.set_yticks([])
    elif axis == 'y':
        ax.set_xticks([])
    if position == 'bottom' and axis != 'y':
        return 0.6, 0.1
    elif position == 'top' and axis != 'y':
        return 0.1, 0.6
    else:
        return 0.1, 0.1


def annotate_coord(ax, chrom, start, end, organism=None, assembly=None,
                   constant=0.1, position='top'):
    string = f'{chrom}: {start:,} - {end:,}'
    if assembly:
        string = 'assembly: ' + assembly + '. ' + string
    if organism:
        string = organism + ', ' + string
    if position == 'top':    
        ax.annotate(string,
                    xy=(0.5, 1 + constant), 
                    xycoords='axes fraction',
                    ha='center')
    else:
        ax.annotate(string,
                    xy=(0.5, -constant), 
                    xycoords='axes fraction',
                    ha='center')
    return 0.1 + constant
    
    
def annotate_mutations(ax, positions, dna_positions, names=None, w=128, constant=0):
    n = len(positions)
    k = np.ceil(n / 2)
    if n == 0:
        return constant
    for i in range(n):
        start, end = positions[i]
        center = (start + end) / 2
        if names:
            prefix = names[i] + ': '
        else:
            prefix = '' #Δ: 
        dna_start, dna_end = dna_positions[i]
        if dna_end > 99999:
            name = prefix + f'{dna_start / 1000:.2f}kb - {dna_end / 1000:.2f}kb'
        else:
            name = prefix + f'{int(dna_start):,} - {int(dna_end):,}'
        if n == 1:
            ha = 'center'
            xytext = (center / w,  -0.3 - constant)
        else:
            if i < k:
                ha = 'right'
                xytext = (0,  -(i % k * 0.2 + 0.3) - constant)
            else:
                ha = 'left'
                xytext = (1, -(k * 0.2 + 0.1 - i % k * 0.2) - constant)
        hight = - (i % k * 0.2 + 0.3)
        mut_w = 18.5 * (end-start) / w
        ax.annotate(name,
                    xy=(center / w, -constant),
                    xycoords='axes fraction',
                    xytext=xytext,
                    ha=ha,
                    arrowprops=dict(arrowstyle=f"-[, widthB={mut_w:.3f}, lengthB=1",
                                    linewidth=1.5,
                                    edgecolor = 'k',
                                    facecolor = 'r',
                                    connectionstyle='angle,angleA=0,angleB=-90'))
    return 0.3 + 0.2 * k + constant

def annotate_genes(ax, positions, names=None, w=128, constant=0.1):
    n = len(positions)
    if n == 0:
        return constant
    for i in range(n):
        if names:
            name = names[i]
        else:
            name = ''
        start, end = positions[i]
        if start == end:
            continue
        if start < end:
            ax.annotate('',
                        xy=(end / w, -constant - 0.1),
                        xycoords='axes fraction',
                        xytext=(start / w, -constant - 0.1),
                        arrowprops=dict(linewidth=1.5,
                                        width=5,
                                        headlength=5,
                                        edgecolor = 'k',
                                        facecolor =  '#5a96d2'))
        if start > end:
            ax.annotate('',
                        xy=(end / w, -constant - 0.1),
                        xycoords='axes fraction',
                        xytext=(start / w, -constant - 0.1),
                        arrowprops=dict(linewidth=1.5,
                                        width=5,
                                        headlength=5,
                                        edgecolor = 'k',
                                        facecolor = '#8c3c3c'))
        if name:
            ax.annotate(name,
                        xy=(np.mean([start, end]) / w, -constant - 0.3 + i % 2 * 0.35),
                        xycoords='axes fraction',
                        ha='center',
                        color='k')
    if names:
        return 0.5 + constant
    else:
        return 0.4 + constant
    
def annotate_motifs(ax, positions, names=None, w=128, constant=0.1):
    pass
        
def annotate_colorbar(im, ax, vmin=None, vmax=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    a = im.get_array()
    if vmin is None:
        b, c = a.min(), a.max()
    else:
        b, c = vmin, vmax
    d = (c-b) / 5
    b, c = b+d, c-d
    ticks = [b, (b+c)/2, c]
    cbar = plt.colorbar(im, cax=cax, ticks=ticks)
    cbar.ax.set_yticklabels([f'{i:.2f}' for i in ticks])


    
    
