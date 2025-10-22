# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import zoom
from scipy import stats

def get_2d(array, channel=0):
    '''Removes extra dimensions'''
    if len(array.shape) == 4:
        return array[0, ..., channel]
    if len(array.shape) == 3:
        return array[..., channel]
    return array

def plot_map(map, ax=None, show=False, cmap="RdBu_r",
             name=None, return_im=False, colorbar=True,
             vmin=None, vmax=None, hide_axis=True, zero_centred=False,
             title=None,
             experiment_index=0):
    '''Plots Hi-C map'''
    if ax is None:
        _, ax = plt.subplots(1,1)
    map = get_2d(map, experiment_index)
    if zero_centred:
        vmin, vmax = np.nanmin(map), np.nanmax(map)
        abs_max = max(np.abs(vmin), np.abs(vmax))
        vmin, vmax = -abs_max, abs_max

    im = ax.imshow(map, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax,
                   extent=[0, map.shape[1], map.shape[0], 0])
    if hide_axis:
        ax.axis('off')
    if name:
        fontsize = max(ax.bbox.height/12, 12)
        txt = ax.text(1,5,f'{name}', fontsize=fontsize, color='white', va='center')
        txt.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                                                     foreground='black')])
    if title:
        ax.set_title(title)
    if colorbar:
        annotate_colorbar(im=im, ax=ax, vmin=vmin, vmax=vmax)
    if show:
        plt.show()
    if return_im:
        return im

def plot_significance_between(samples):
    '''Plots significance stars between boxplots (many pairs possible)
Shape should be (2, m, n), where m is number of samples and n is sample size'''
    pvals = []
    for i, sample_pair in enumerate(zip(samples[0], samples[1])):
        first_sample, second_sample = sample_pair
        pvals.append(stats.mannwhitneyu(first_sample, second_sample).pvalue)
    pvals_adjusted = stats.false_discovery_control(pvals)

    for i, sample_pair in enumerate(zip(samples[0], samples[1])):
        first_sample, second_sample = sample_pair
        threshold = 0.05
        p = pvals_adjusted[i]
        if p > 0.05:
            symbol = 'ns'
        elif p > 0.01:
            symbol = '*'
        elif p > 0.001:
            symbol = '**'
        elif p > 0.0001:
            symbol = '***'
        else:
            symbol = '****'
        x1, x2 = i-0.2, i+0.2
        y1, y2 = first_sample.max() * 1.05, second_sample.max() * 1.05
        h = (first_sample.max() - first_sample.min()) / 30
        top = max(y1, y2) + h
        plt.plot([x1, x1, x2, x2], [y1, top, top, y2], lw=1, c='k')
        plt.text((x1 + x2) / 2, top, symbol, ha='center', va='bottom', fontsize=10)

def plot_results(y_pred, y_true, sample=None, numbers=None, data=None,
                 save = False, equal_scale=False, zero_centred=False,
                 experiment_name=None, experiment_index=0,
                 top_name='True', bottom_name='Pred'):
    '''Plots model predictions compared to real maps'''
    plt.rcParams.update({'font.size': 11})
    if len(y_pred) == 9:
        k, l = 3, 3
        fig = plt.figure(figsize=(17*1.1,8*1.1))
        gs1 = fig.add_gridspec(nrows=131, ncols=130,
                               left=0, right=1, top=1, bottom=0)
    else:
        k, l = 2, 4
        fig = plt.figure(figsize=(26,8.5))
        gs1 = fig.add_gridspec(nrows=131, ncols=170,
                               left=0, right=1, top=1, bottom=0)

    if equal_scale == 'global':
        vmin = min(np.nanmin(y_true), np.nanmin(y_pred))
        vmax = max(np.nanmax(y_true), np.nanmax(y_pred))
    elif equal_scale == 'samples':
        vmin1, vmin2 = np.nanmin(y_true), np.nanmin(y_pred)
        vmax1, vmax2 = np.nanmax(y_true), np.nanmax(y_pred)
        vmin=None
        vmax=None
    else:
        vmin=None
        vmax=None
    for i in range(k):
        for j in range(l):
            ax1 = fig.add_subplot(gs1[20+i*40 : 20+i*40+15,
                                      20+j*33 : 20+j*33+30])
            ax2 = fig.add_subplot(gs1[20+i*40+15 : 20+i*40+30,
                                      20+j*33 : 20+j*33+30])
            try:
                y1 = y_true[i*l+j]
            except IndexError:
                ax1.axis('off')
                ax2.axis('off')
                continue
            y2 = np.flip(y_pred[i*l+j],axis=0)
            if equal_scale == 'pairs':
                vmin = min(np.nanmin(y1), np.nanmin(y2))
                vmax = max(np.nanmax(y1), np.nanmax(y2))
            if zero_centred:
                if vmin is not None:
                    max_abs = max(np.abs([vmin, vmax]))
                    vmin1, vmin2 = -max_abs, -max_abs
                    vmax1, vmax2 = max_abs, max_abs
                else:
                    if equal_scale != 'samples':
                        vmin1, vmin2 = np.nanmin(y1), np.nanmin(y2)
                        vmax1, vmax2 = np.nanmax(y1), np.nanmax(y2)
                    max_abs1 = max(np.abs([vmin1, vmax1]))
                    max_abs2 = max(np.abs([vmin2, vmax2]))
                    vmin1, vmin2 = -max_abs1, -max_abs2
                    vmax1, vmax2 = max_abs1, max_abs2
            elif not zero_centred and equal_scale != 'samples':
                vmin1, vmin2 = vmin, vmin
                vmax1, vmax2 = vmax, vmax


            if data is None:
                plot_map(y1, ax=ax1, show=False,
                         vmin=vmin1, vmax=vmax1, name='True')
                plot_map(y2, ax=ax2, show=False,
                         vmin=vmin2, vmax=vmax2, name='Pred')
                ax1.axis('off')
                ax2.axis('off')
            else:
                axis = ('y','y_inv') if j==0 else (None, None)
                data.plot_annotated_map(sample=sample,
                                        index=numbers[i*l+j],
                                        ax=ax1,
                                        hic_map=y1,
                                        axis=axis[0],
                                        y_label_shift=True,
                                        show_position=True,
                                        colorbar=True,
                                        experiment_name=experiment_name,
                                        experiment_index=experiment_index,
                                        full_name=False,
                                        vmin=vmin1,
                                        vmax=vmax1,
                                        name=top_name,
                                        show=False)
                data.plot_annotated_map(sample=sample,
                                        index=numbers[i*l+j],
                                        ax=ax2,
                                        hic_map=y2,
                                        axis=axis[1],
                                        experiment_name=experiment_name,
                                        experiment_index=experiment_index,
                                        show_position=False,
                                        colorbar=True,
                                        full_name=False,
                                        vmin=vmin2,
                                        vmax=vmax2,
                                        name=bottom_name,
                                        show=False)
    plt.rcParams.update({'font.size': 9})
    if save:
        plt.savefig(save)
        fig.clear()
        plt.close(fig)

def plot_metrics_history(train_metric_history, val_metric_history, save=None):
    '''Plots history of training metrics'''
    fig, ax = plt.subplots(nrows=1, ncols=2,
                           figsize=(12,6))
    for i, metric in enumerate(train_metric_history.keys()):
        train_values = train_metric_history[metric]
        ax_i = ax[0] if i==0 else ax[1]
        ax_i.plot(train_values, label='train '+metric)
    for i, metric in enumerate(val_metric_history.keys()):
        val_values = val_metric_history[metric]
        ax_i = ax[0] if i==0 else ax[1]
        ax_i.plot(val_values, label='val '+metric)
    for ax_i in ax:
        ax_i.legend()
    if save:
        plt.savefig(save)
        fig.clear()
        plt.close(fig)

def compare_mean_gradients(target, control, name='Target'):
    samples = [name] * len(target) + ['Control'] * len(control)
    values = np.concatenate([target, control])
    df = pd.DataFrame({'Sample': samples,
                       'Integrated gradients value': values,
                       '':['']*len(samples)
                        })
    fig,ax = plt.subplots(figsize=(5,4))
    if int(sns.__version__.split('.')[1]) < 13:
        sns.violinplot(
            x='',
            hue='Sample',
            y='Integrated gradients value',
            data=df,
            palette='RdBu_r',
            scale='width',
            ax=ax,
            cut=0
        )
        plt.legend('')
    else:
        sns.violinplot(
            x='',
            hue='Sample',
            y='Integrated gradients value',
            data=df,
            palette='RdBu_r',
            density_norm='width',
            legend=False,
            ax=ax,
            cut=0
        )
    plot_significance_between(np.array([target,control])[:,None,:])
    plt.xticks((-0.2, 0.2), labels=(name, 'Control'))

def plot_motiff_effect(numbers, sites, names):
    '''Boxplots of predicted mutant seq projections on some vecter'''
    sample_name = []
    x = []
    y = []
    for site, name in zip(sites, names):
        sample_name += [name] * np.prod(site.shape)
        x.append(np.repeat(numbers, site.shape[-1]))
        y.append(site.flatten())
    x = np.concatenate(x)
    y = np.concatenate(y)
    df = pd.DataFrame({'Number of inserted sites': x,
                       'Projection': y,
                       'Sample': sample_name})
    fig,ax = plt.subplots(figsize=(10,4))
    sns.boxplot(x='Number of inserted sites',
                   y='Projection',
                   data=df,
                   hue='Sample',
                   palette='RdBu_r')
    plt.legend(loc='upper left')
    plt.grid(True, axis='y')
    ax.set_axisbelow(True)

def plot_score_basic(
        metric_name,
        correct,
        permuted,
        x,
        title=None
    ):
    control = np.array(permuted)
    correct = np.array(correct)
    fig, ax = plt.subplots(1,1,figsize=(2.2,3))
    y_name = f'{metric_name.capitalize()} correlation'
    x_name = ''
    n = len(correct)
    sns.violinplot(data=pd.DataFrame({x_name: ['Predictions']*n + ['Control']*n,
                                y_name: np.concatenate([correct, control])}),
                    x=x_name, y=y_name, hue=x_name, palette='RdBu_r', ax=ax, cut=0, legend=False)
    ax.set_ylim(-1.2,1.2)
    txt = ax.text(0.1,
            correct.mean(),
            f'''Median = {np.median(correct):.2f}\nSize = {len(correct)}''',
            ha='center',
            color='k',
            weight='semibold',
            fontsize=9)
    txt = ax.text(0.9,
            control.mean(),
            f'''Median = {np.median(control):.2f}\nSize = {len(control)}''',
            ha='center',
            color='k',
            weight='semibold',
            fontsize=9)
    ax.set_title(title)

def plot_score_line(
        metric,
        scores,
        control_scores,
        x,
        title
    ):
    plt.plot(x[:-1], scores, '-o', c='b', alpha=0.3)
    plt.plot(x[:-1], control_scores, '-o', c='r', alpha=0.3)
    plt.xticks([x[i] for i in range(0, len(x), 4)],
               labels=[str(int(x[i]/1000))+'kb' for i in range(0, len(x), 4)])
    plt.ylabel(metric.capitalize())
    plt.xlabel('Genomic distance')
    plt.title(title)
    plt.ylim(-1, 1)

def plot_score_one_distance(
        metric_name,
        correct,
        permuted,
        x,
        title=None
    ):
    '''Plots model testing metrics'''
    x = np.flip(x)
    permuted = np.array(permuted)
    correct = np.array(correct)
    fig, ax = plt.subplots(1,1,figsize=(4,6))
    medians = np.median(correct, axis=0)
    best = correct[:, np.argmax(medians)]
    control = permuted[:, np.argmax(medians)]
    x = x[np.argmax(medians)]
    y_name = f'{metric_name.capitalize()} correlation'
    x_name = ''
    n = len(best)
    sns.violinplot(data=pd.DataFrame({x_name: ['Predictions']*n + ['Control']*n,
                                y_name: np.concatenate([best, control])}),
                    x=x_name, y=y_name,  hue=x_name, palette='RdBu_r', ax=ax, cut=0, legend=False)
    ax.set_ylim(-1.2,1.2)
    txt = ax.text(0.1,
            best.mean(),
            f'''Mean = {best.mean():.2f}
Median = {np.median(best):.2f}
P-value = {stats.ttest_1samp(best, 0, alternative='greater').pvalue:.2f}''',
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
P-value = {stats.ttest_1samp(control, 0, alternative='greater').pvalue:.2f}''',
            ha='center',
            color='w',
            weight='semibold',
            fontsize=14)
    ax.set_title(f'''Correlations between true and predicted contacts
at the distance with the best score ({int(x)} bp)''')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2,
                                                foreground='black')])
    if title:
        plt.suptitle(title)

def plot_score_full(metric_name,
               correct,
               permuted,
               x,
               title=None,
               folder=''):
    '''Plots model testing metrics'''
    permuted = np.flip(np.array(permuted), axis=1)
    correct = np.flip(np.array(correct), axis=1)
    # np.save(os.path.join(folder, title+' predictions.npy'), correct)
    # np.save(os.path.join(folder, title+' control.npy'), permuted)
    fig, ax = plt.subplots(1,1,figsize=(20,6))
    x = np.tile(x[1:], len(correct)).astype(int)
    y_name = f'{metric_name.capitalize()} correlation'
    x_name = 'Genomic distance'
    if x[1]-x[0] > 1000:
        x = x // 100 / 10
        x_name += ', kb'
    df = pd.DataFrame({x_name: x,
                        y_name: correct.flat,
                        ' ': 'Predictions'})
    df = pd.concat([df,
                    pd.DataFrame({x_name: x,
                        y_name: permuted.flat,
                        ' ': 'Control'})])
    sns.boxplot(data=df, x=x_name, y=y_name, hue=' ',
                palette='RdBu_r', ax=ax)
    plt.text(-0.5, 1.3, 'Mean:', ha='right')
    plt.text(-0.5, 1.2, 'Median:', ha='right')
    for i in range(correct.shape[1]):
        plt.text(i, 1.3, f'{correct[:, i].mean():.2f}', ha='center')
        plt.text(i, 1.2, f'{np.median(correct[:, i]):.2f}', ha='center')
    ax.set_ylim(-1.1,1.4)
    ax.set_yticks([-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0])
    ax.grid(axis='y',linestyle='--')
    ax.set_xlim(-2, correct.shape[1])
    plt.xticks(rotation = 90)
    ax.grid(True, axis='y')
    ax.set_axisbelow(True)
    plot_significance_between([correct.T, permuted.T])
    plt.legend(loc='lower left')
    if title:
        plt.title(title.strip() + ', test sample size = ' + str(len(correct)))


def annotate(ax, start, end, axis='both',
             h=32, remove_first_diag=0, w=128, vertical_shift=0,
             brief=True,
             y_label_shift=False, x_top=False):
    '''Annotates map coordinates on axes'''
    resolution = (end-start)//w
    ax.tick_params(direction='out')
    if not axis:
        ax.set_xticks([])
        ax.set_yticks([])
        return 0.1, 0.1
    if axis == 'x' or axis == 'both':
        if brief:
            l = np.arange(0, w+1, 32)
            x = [f'{int((i*resolution+start)//1000):,}\nkb' for i in l]
        else:
            l = np.arange(0, w+1, 12)
            x = [f'{int((i*resolution+start)):,}' for i in l]
        ax.set_xticks(l)
        ax.set_xticklabels(x)
        if x_top:
            ax.xaxis.tick_top()

    if axis == 'y' or axis == 'y_inv' or axis == 'both':
        y_max = int((end-start) * (2*(h+remove_first_diag)/w))
        y_min = int((end-start) * (2*remove_first_diag/w))
        y = [f'{int(i):,}' for i in np.linspace(y_min, y_max, 5)]
        if axis != 'y_inv':
            y = reversed(y)
            ax.set_yticks(np.linspace(0.5, h, 5))
            s =  h//2 if y_label_shift else 0
            ax.text(-30, h//2+s, 'Genomic distance', rotation=90, va='center')
            ax.set_yticklabels(y)
        else:
            ax.set_yticks(np.linspace(0, h-.5, 5))
            ax.set_yticklabels(['']+y[1:])

    if axis == 'x':
        ax.set_yticks([])
    elif axis == 'y' or axis == 'y_inv':
        ax.set_xticks([])
    return 0.1, 0.2


def annotate_title(ax, chrom, start, end, organism=None, assembly=None,
                   experiment_name='',
                   vertical_shift=0.05, position='top'):
    '''Annotatets map coordinates and other information in title'''
    string = f'{chrom}: {start:,} - {end:,}'
    if experiment_name:
        string = experiment_name + ', ' + string
    if assembly:
        string = 'assembly: ' + assembly + '. ' + string
    if organism:
        string = organism + ', ' + string

    if position == 'top':
        ax.annotate(string,
                    xy=(0.5, 1 + vertical_shift),
                    xycoords='axes fraction',
                    ha='center')
    else:
        ax.annotate(string,
                    xy=(0.5, -vertical_shift),
                    xycoords='axes fraction',
                    ha='center')
    return 0.1 + vertical_shift


def annotate_mutations(
        ax,
        positions,
        dna_positions=None,
        names=None,
        w=128,
        vertical_shift=0
    ):
    '''Annotates mutations with arrows'''
    n = len(positions)
    k = np.ceil(n / 2)
    if n == 0:
        return vertical_shift
    for i in range(n):
        start, end = positions[i]
        center = (start + end) / 2
        if names:
            prefix = names[i] + ': '
        else:
            prefix = ''
        if dna_positions is not None:
            dna_start, dna_end = dna_positions[i]
            if dna_end > 99999:
                name = prefix + f'{dna_start/1000:.2f}kb - {dna_end/1000:.2f}kb'
            else:
                name = prefix + f'{int(dna_start):,} - {int(dna_end):,}'
        else:
            name = names[i]
        if n == 1:
            ha = 'center'
            xytext = (center / w,  -0.3 - vertical_shift)
        else:
            if i < k:
                ha = 'right'
                xytext = (0,  -(i % k * 0.2 + 0.3) - vertical_shift)
            else:
                ha = 'left'
                xytext = (1, -(k * 0.2 + 0.1 - i % k * 0.2) - vertical_shift)
        hight = - (i % k * 0.2 + 0.3)
        mut_w = 18.5 * (end-start) / w
        ax.annotate(
            name,
            xy=(center / w, -vertical_shift),
            xycoords='axes fraction',
            xytext=xytext,
            ha=ha,
            arrowprops=dict(
                arrowstyle=f"-[, widthB={mut_w:.3f}, lengthB=1",
                linewidth=1.5,
                edgecolor = 'k',
                facecolor = 'r',
                connectionstyle='angle,angleA=0,angleB=-90'
                )
            )
    return 0.3 + 0.2 * k + vertical_shift

def annotate_boxes(ax, data, boxes, names=None, w=128, vertical_shift=0.1):
    '''Plots genes/motifs location, orientation, names and scores'''
    if isinstance(boxes, pd.DataFrame):
        new_boxes = []
        names = []
        for _, row in boxes.iterrows():
            start, end = row.start, row.end
            if hasattr(row, 'strand'):
                if row.strand == '-':
                    start, end = end, start
            new_boxes.append([start, end])
            if 'name' in row.index:
                if row.name:
                    names.append(row['name'])
                else:
                    names.append('')
        if hasattr(boxes, 'score'):
            scores = np.array(boxes.score)
            if len(scores):
                scores -= scores.min()
                scores /= scores.max()
        else:
            scores = None
        if hasattr(boxes, 'data_name'):
            data_name = boxes.data_name
        else:
            data_name = ''
        boxes = new_boxes
    else:
        scores = None
        data_name = ''


    boxes = np.array(boxes).reshape((-1,2))
    boxes = (boxes - data.offset) / data.resolution
    n = len(boxes)
    if n == 0:
        return vertical_shift
    ax.annotate('',
                xy=(1, -vertical_shift + 0.07),
                xycoords='axes fraction',
                xytext=(0, -vertical_shift + 0.07),
                arrowprops=dict(arrowstyle="->"))
    ax.annotate('',
                xy=(0, -vertical_shift),
                xycoords='axes fraction',
                xytext=(1, -vertical_shift),
                arrowprops=dict(arrowstyle="->"))
    ax.annotate(data_name,
                xy = (-0.05, -vertical_shift + 0.035),
                xycoords='axes fraction',
                fontsize=15,
                ha='center',
                va='center')
    for i in range(n):
        if names:
            name = names[i]
        else:
            name = ''
        start, end = boxes[i]
        if scores is not None:
            brightness = scores[i]
        else:
            brightness = 1
        if start == end:
            continue
        headlength = 5 if np.abs(end-start) > 1 else 0.01
        if start < end:
            color = (0.7, (1-brightness)*0.7, (1-brightness)*0.7)
            ax.annotate('',
                        xy=(end / w, -vertical_shift + 0.07),
                        xycoords='axes fraction',
                        xytext=(start / w, -vertical_shift + 0.07),
                        arrowprops=dict(linewidth=1.5,
                                        width=5,
                                        headlength=headlength,
                                        edgecolor=color,
                                        facecolor=color
                                        )
                        )
            if name:
                ax.annotate(name,
                            xy=(np.mean([start, end])/w, -vertical_shift+0.04),
                            xycoords='axes fraction',
                            ha='center',
                            va='center',
                            color='k')

        else:
            color = ((1-brightness)*0.7, (1-brightness)*0.7, 0.7)
            ax.annotate('',
                        xy=(end / w, -vertical_shift),
                        xycoords='axes fraction',
                        xytext=(start / w, -vertical_shift),
                        arrowprops=dict(linewidth=1.5,
                                        width=5,
                                        headlength=headlength,
                                        edgecolor=color,
                                        facecolor=color
                                        )
                        )
            if name:
                ax.annotate(name,
                            xy=(np.mean([start, end])/w, -vertical_shift+0.03),
                            xycoords='axes fraction',
                            ha='center',
                            va='center',
                            color='k')
    if names:
        return 0.5 + vertical_shift
    else:
        return 0.4 + vertical_shift

def annotate_colorbar(im, ax, vmin=None, vmax=None):
    '''Add colorbar'''
    divider = make_axes_locatable(ax)
    size = min(ax.bbox.width * 0.05, 0.25)
    cax = divider.append_axes("right", size=size, pad=0.05)
    a = im.get_array()
    if vmin is None:
        b, c = a.min(), a.max()
    else:
        b, c = vmin, vmax
    d = (c-b) / 20
    b, c = b+d, c-d
    ticks = [b, (b+c)/2, c]
    cbar = plt.colorbar(im, cax=cax, ticks=ticks)
    cbar.ax.set_yticklabels([f'{i:.2f}' for i in ticks])
    ticks = cbar.ax.yaxis.get_majorticklabels()
    plt.setp(ticks[0], va="bottom")
    plt.setp(ticks[-1], va="top")

class StaticColorAxisBBox(matplotlib.patches.FancyBboxPatch):
    '''Util for chromosomes drawing'''
    def set_edgecolor(self, color):
        if hasattr(self, "_original_edgecolor"):
            return
        self._original_edgecolor = color
        self._set_edgecolor(color)

    def set_linewidth(self, w):
        super().set_linewidth(0.5)

def _round_axes(aspect):
    '''Util for chomosomes drawing'''
    class RoundedAxes(matplotlib.axes.Axes):
        name = "rounded" + str(aspect)
        _edgecolor: str

        def __init__(self, *args, **kwargs):
            self._edgecolor = kwargs.pop("edgecolor", None)
            super().__init__(*args, **kwargs)

        def _gen_axes_patch(self):
            return StaticColorAxisBBox(
                (0, 0),
                1.0,
                1.0,
                boxstyle="round, rounding_size=0.5, pad=0",
                mutation_aspect=aspect,
                edgecolor=self._edgecolor,
                linewidth=2,
            )
    return RoundedAxes

def plot_samples(chromsizes, df):
    '''Plots what chromosome parts in what samples are'''
    max_bins = 1000000
    n = len(chromsizes)
    max_size = max(list(chromsizes.values()))
    binsize = max_size / max_bins
    fig = plt.figure(figsize=(n*0.7, 7))
    gs = fig.add_gridspec(nrows=1000, ncols=n)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_yticks(np.linspace(0, max_bins, 20))
    lab = [f'{int(i)//1000:,} kb' for i in np.linspace(0, max_size, 20)]
    ax.set_yticklabels(lab)
    ax.set_xticks(())
    ax.spines[["bottom", "left", "right", "top"]].set_visible(False)
    for i, item in enumerate(sorted(chromsizes.items())):
        chrom, size = item
        chrom_df = df.loc[df['chrom'] == chrom]
        n_bins = int(size / binsize)
        aspect = max_bins / 7 / n_bins
        matplotlib.projections.register_projection(_round_axes(aspect))
        h = n_bins // (max_bins // 1000)
        ax = fig.add_subplot(gs[-h:, i], projection="rounded"+str(aspect))
        ax.spines[["bottom", "left", "right", "top"]].set_visible(False)
        ax.set_xticks(())
        ax.set_yticks(())

        for sample, cmap in zip(['train', 'val', 'test'],
                                ['Greys', 'Reds', 'Blues']):
            arr = np.zeros(n_bins)
            for _, row in chrom_df[chrom_df['sample'] == sample].iterrows():
                bin_index_0 = int(round(row.start / binsize))
                bin_index_1 = int(round(row.end / binsize))
                arr[bin_index_0 : bin_index_1+1] += 1
            arr[arr == 0] = np.nan
            arr[arr >= 0] = 1
            ax.imshow(arr[:,None], cmap=cmap, aspect='auto',
                      vmin=0, vmax=2)
        ax.set_ylim(0, n_bins)
        ax.set_title(chrom, fontsize=10)
    if n > 2:
        train_patch = matplotlib.patches.Patch(color='grey', label='Train')
        val_patch = matplotlib.patches.Patch(color='orangered', label='Val')
        test_patch = matplotlib.patches.Patch(color='skyblue', label='Test')
        ax = fig.add_subplot(gs[:100, -1])
        ax.axis('off')
        ax.legend(handles=[train_patch, val_patch, test_patch])
    plt.show()


def plot_ig(ig,
            y_true,
            y_pred,
            peak_table,
            data,
            region,
            annotation,
            experiment_index,
            annotate_peaks=True):
    '''Plots summary for integrated gradients'''
    fig = plt.figure(figsize=(20,8))

    gs1 = fig.add_gridspec(nrows=1000, ncols=5, left=0.01, right=0.99,
                           top=1, bottom=0)
    ax1 = fig.add_subplot(gs1[:500, :4])
    ax3 = fig.add_subplot(gs1[500:, :4])
    ax2 = fig.add_subplot(gs1[100:900, :4])
    ax4 = fig.add_subplot(gs1[0:1000, 4])

    chrom, start, end = data._parse_region(region)
    data.plot_annotated_map(
        hic_map=y_true[0],
        ax=ax1,
        colorbar=False,
        name='True',
        experiment_index=experiment_index,
        chrom=chrom,
        start=start,
        end=end,
        x_top=True,
        brief=False,
        axis='x'
        )
    data.plot_annotated_map(
        hic_map=np.flip(y_pred[0], axis=0),
        ax=ax3,
        colorbar=False,
        name='Predicted',
        experiment_index=experiment_index,
        chrom=chrom,
        start=start,
        end=end,
        show_position=False,
        axis=None)

    if annotation is not None:
        for i, boxes in enumerate(annotation):
            annotate_boxes(ax3, data, boxes, vertical_shift=0.1+i*0.2)

    ax2.plot(ig, c='black')
    ax2.set_xlim(0, len(ig))
    marginal = max(ig)
    ax2.set_ylim(-marginal*1.5, marginal*1.5)
    ax2.axis('off')

    if annotate_peaks:
        for i, row in peak_table.iterrows():
            peak_position = (row.start + row.end) // 2 - start
            seq = row.seq.upper()
            h = ig[peak_position]
            ax2.scatter([peak_position], [h+marginal/8], s=450, color='k')
            ax2.scatter([peak_position], [h+marginal/8], s=400, color='w')
            ax2.text(peak_position,
                     h+marginal/8,str(i+1),
                     color='k', size=16,
                     ha='center', va='center')
            peak_region = f'{row.chrom}:{row.start}-{row.end}'
            text = f'{int(i)+1}. {peak_region}\n{seq}\n'

            ax4.text(0, 1-i*(1/len(peak_table)), text,  size=16, va='top')

    ax4.axis('off')

def plot_gene_composition(y, data, boxes, n_replicates, sd):
    '''Plots prediction of a composition of genes in specified orientation'''
    _, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
    if sd is not None:
        y/=sd
    else:
        y/=2
    plot_map(y, ax=ax, zero_centred=True)
    h, w = y.shape[0], y.shape[1]
    annotate_boxes(ax, data, boxes)
    ax.set_xlim(0,w)
    ax.set_ylim(h,0)

def show_min_max_projections(data, channel, df, ax, vmin=None, vmax=None, score=None, amount=500):
    '''Plots mean maps from regions withs highest and lowest projections on \
specified vectors in the latent space '''
    for i, proj in enumerate(df.columns[5:]):
        best_hits=[]
        if score is None:
            sorted_proj = df.sort_values(by=[proj])
            selected = sorted_proj.iloc[-amount:]
        else:
            selected = df.loc[df[proj]>score]
        for _,row in selected.iterrows():
            chrom, hic_start, hic_end = row.chrom, row.map_start, row.map_end
            hic_map = data._slice_map(chrom, hic_start, hic_end)[..., channel]
            best_hits.append(hic_map)
        title = proj.capitalize()
        name =  data.experiment_names[channel]
        if len(best_hits) > 0:
            plot_map(np.mean(best_hits, axis=0), title=title, name=name, ax=ax[i],
                 vmin=vmin, vmax=vmax)
        else:
            ax[i].axis('off')

def plot_corr(dfs, experiment_names):
    '''Plots correlations between fearures in multiple cell types'''
    projections = dfs[0].columns[5:]
    n_vecs = len(dfs[0].columns)-5
    figsize = (n_vecs * (len(dfs)+1), len(dfs))
    fig, axs = plt.subplots(nrows=1, ncols=len(projections), figsize=figsize)
    for i, proj in enumerate(projections):
        corr = np.corrcoef([df[proj] for df in dfs])
        im = axs[i].imshow(corr, vmin=-1, vmax=1, cmap='bwr')
        for j in range(corr.shape[1]):
            for k in range(corr.shape[0]):
                txt = axs[i].text(k, j, f"r={corr[j, k]:.2f}",
                       ha="center", va="center", color="w")
                txt.set_path_effects([PathEffects.withStroke(linewidth=1.5,
                                                     foreground='black')])
        ticks = np.arange(len(dfs))
        labels = experiment_names
        axs[i].set_xticks(ticks, labels=labels)
        axs[i].set_yticks(ticks, labels=labels)
        axs[i].set_title(proj)


def _plot_spreares_schema(rs, colors, angle_between):
    '''Plots schematic illustration explaining main figure in plot_spheares'''
    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot()
    points = np.random.normal(0,1,(32,2))
    points /= np.linalg.norm(points, axis=1)[:,None]
    half_angle = angle_between / 2
    y = np.tan(half_angle)
    right_vec = np.array([1, y])
    right_vec /= np.linalg.norm(right_vec)
    left_vec = right_vec.copy()
    left_vec[0] *= -1
    special_vecs = np.stack([left_vec, right_vec])
    for i in range(len(rs)):
        circ = plt.Circle((0, 0), rs[i], color=colors[i], fill=False)
        ax1.add_patch(circ)
        dots = points*rs[i]
        ax1.scatter(*dots.T, color=colors[i], s=30)
        special_dots = special_vecs*rs[i]
        x, y = special_dots[:, 0], special_dots[:, 1]
        ax1.scatter(x[0], y[0], color=colors[i],
                    edgecolors='k', marker='v', s=100)
        ax1.scatter(x[1], y[1], color=colors[i], edgecolors='k', s=100)
    ax1.text(x[0], y[0]*1.15, f'Insulation\nvector',
             ha='center', va='center', fontsize=12)
    ax1.text(x[1], y[1]*1.15, f'Loop\nvector',
             ha='center', va='center', fontsize=12)

    ax1.scatter([0], [0], color='k',s=30)

    a = int(rs.max())
    ax1.set_xlim((-a*1.05, a*1.05))
    ax1.set_ylim((-a*1.05, a*1.05))
    ax1.set_aspect('equal')
    ax1.axis('off')

    canvas = fig.canvas
    canvas.draw()
    plt.close()
    image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape(canvas.get_width_height()[::-1] + (4,))[...,:3]
    x = image.min(axis=(0,2)) < 240
    y = image.min(axis=(1,2)) < 240
    xmesh = np.repeat(x[None,:],len(y),axis=0)
    ymesh = np.repeat(y[None,:],len(x),axis=0).T
    mask = np.logical_and(xmesh, ymesh)
    mask = np.repeat(mask[...,None], 3, axis=2)
    image = image[mask].reshape(y.sum(), x.sum(), 3)
    h,w,_ = image.shape
    alpha = np.ones((h,w,1))
    for i in range(h//2):
        alpha[i+h//2] = (1-i/(h/2)-0.02)**8
    image = np.concatenate([image/255,alpha],axis=2)
    return image

def _angle_between(vec1, vec2):
    return np.arccos(np.clip(np.dot(vec1.flat, vec2.flat), -1.0, 1.0))

def plot_spheares(rs, corrs, corrs_special, vecs_special, ax1, ax2):
    '''Latent space illustration'''
    colors = plt.cm.coolwarm(rs/rs.max()*0.75)

    angle_between = _angle_between(*vecs_special)
    image = _plot_spreares_schema(rs, colors, angle_between)
    ax1.imshow(image, interpolation='bilinear')
    ax1.set_aspect('equal')
    ax1.spines[["left", "right", "top"]].set_visible(False)

    a = int(rs.max())
    xticks = np.linspace(0, 1, 2*int(rs.max())+1)
    labels = ((xticks - xticks.max()/2)*rs.max()*2).astype(int)
    extent = image.shape[1]
    ax1.set_xticks(xticks*extent)
    ax1.set_xticklabels(labels)
    ax1.set_yticks(())
    ax1.spines.bottom.set_position(('axes',0.5))
    ax1.tick_params(axis='x', labelsize=11)

    bplot = ax2.boxplot(corrs.T,
                        vert=True,
                        patch_artist=True,
                        medianprops={'linewidth':1, 'color':'k'},
                        boxprops={'linewidth':0},
                        showfliers=False)

    ax2.scatter(np.arange(len(rs))+1,
                corrs_special[...,0].mean(axis=1),
                zorder=100,
                marker='v',
                c=colors,
                edgecolors='k',
                label='Means for insulation vector'
                )
    ax2.scatter(np.arange(len(rs))+1,
                corrs_special[...,1].mean(axis=1),
                zorder=100,
                c=colors,
                edgecolors='k',
                label='Means for loop vector'
                )

    ax2.set_xticklabels([f'{r:.1f}' for r in rs])
    ax2.set_xlabel('Distance from a map representation in the latent space',
                    fontsize=10)
    ax2.set_ylabel('Pearson r with the map\npredicted from the central point',
                    fontsize=10)
    ax2.legend()
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

def plot_latent_walk(vecs_special, maps_special, rs, axs_ins, axs_loop):
    '''Plots maps predicted from points along the insulation and loop vectors in
the latent space'''
    axs_ins[0].set_title('Insulation vector')
    axs_loop[0].set_title('Loop vector')
    vmax = max(np.abs(maps_special.max()), np.abs(maps_special.min()))
    vmin = -vmax
    for j in range(maps_special.shape[1]):
        if not j%2:
            plot_map(maps_special[0, j],
                     ax=axs_ins[j//2],
                     vmin=vmin,
                     vmax=vmax,
                     name=f'd={rs[j]:.1f}',
                     colorbar=not j)
            plot_map(maps_special[1, j],
                     ax=axs_loop[j//2],
                     vmin=vmin,
                     vmax=vmax,
                     name=f'd={rs[j]:.1f}',
                     colorbar=False)


def plot_latent_space(rs, corrs, corrs_special, vecs_special, maps_special):
    '''Detailed illustration of latent space features'''
    maps_special = np.array(maps_special).mean(axis=1).transpose((1,0,2,3,4))
    fig = plt.figure(figsize=(15,7))
    gs1 = fig.add_gridspec(nrows=50, ncols=80)
    ax1 = fig.add_subplot(gs1[:, :35])
    ax2 = fig.add_subplot(gs1[30:, :35])
    plot_spheares(rs, corrs, corrs_special, vecs_special, ax1, ax2)
    n_axes = int(np.ceil(maps_special.shape[1]/2))
    axs_ins = [fig.add_subplot(gs1[2+i*7:2+(i+1)*6+i, 35:55]) for i in range(1,n_axes)]
    axs_ins = [fig.add_subplot(gs1[2:8, 35:57])] + axs_ins
    axs_loop = [fig.add_subplot(gs1[2+i*7:2+(i+1)*6+i, 58:78]) for i in range(0,n_axes)]
    #axs_loop = [fig.add_subplot(gs1[2:8, 58:80])] + axs_loop
    plot_latent_walk(vecs_special, maps_special, rs, axs_ins, axs_loop)

class LogoPlotter():
    '''Plots motif logo'''
    def __init__(self, a='#228800', c='#2244dd', g='#ffaa00', t='#dd0011'):
        self.letters = ['A','C','G','T']
        self.colors = {'A':a,'C':c,'G':g,'T':t}
        self.w = 130
        self.h = 130
        self.letter_images = self._letter_images()

    def _letter_images(self):
        '''Makes a dict of images for the letters'''
        letters = self.letters
        colors = self.colors
        letter_images = {}
        for letter, color in zip(letters, colors):
            fig, ax = plt.subplots(figsize=(self.w/100, self.h/100))
            color = colors[letter]
            ax.text(-0.32,-0.1,letter,fontsize=120,color=color,fontweight='bold')
            ax.set_xlim(0,2)
            ax.set_ylim(0,1)
            ax.axis('off')
            canvas = fig.canvas
            canvas.draw()
            plt.close()
            image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(canvas.get_width_height()[::-1] + (4,))[...,:3]

            # cut white edges and make letters to have the same sizes
            x = image.min(axis=(0,2)) < 240
            y = image.min(axis=(1,2)) < 240
            xmesh = np.repeat(x[None,:],len(y),axis=0)
            ymesh = np.repeat(y[None,:],len(x),axis=0).T
            mask = np.logical_and(xmesh, ymesh)
            mask = np.repeat(mask[...,None], 3, axis=2)
            image = image[mask].reshape(y.sum(), x.sum(), 3)
            w, h, _ = image.shape
            image = zoom(image, (self.w/w, self.h/h, 1))

            letter_images[letter] = image
        return letter_images

    def plot_logo(self, ic, ax=None):
        logo = []
        l = len(ic)
        for col in ic:
            column = []
            for i in np.argsort(col)[::-1]:
                letter, h = self.letters[i], col[i]
                im = self.letter_images[letter]
                im = zoom(im, (h * 2, 1, 1), order=1)
                column.append(im)
            column = np.concatenate(column, axis=0)
            column_h = column.shape[0]
            max_h = self.h * 4
            top_white = np.full((max_h - column_h, self.w, 3), 255)
            column = np.concatenate([top_white, column])
            logo.append(column)
        if ax is None:
            fig, ax = plt.subplots(figsize=(len(ic)/5, 2))
        ax.imshow(np.concatenate(logo, axis=1))
        ax.set_yticks([0, self.h * 2, self.h * 4])
        ax.set_yticklabels([2, 1, 0])
        ax.set_xticks(np.arange(l) * self.w + self.w // 2)
        ax.set_xticklabels(np.arange(l) + 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)



