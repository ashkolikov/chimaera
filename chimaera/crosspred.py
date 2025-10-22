import os
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio import Phylo
from scipy.ndimage import zoom, gaussian_filter
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, false_discovery_control
from .plot_utils import plot_map


def save_pred(y, org1, org2=None, mask=False, root=''):
    if org2:
        saving_path = root+org1+'/'+'predicted_by_'+org2+'_model.npy'
    else:
        if not mask:
            saving_path = root+org1+'/groundtruth.npy'
        else:
            saving_path = root+org1+'/mask.npy'
    np.save(saving_path, y)


def predict_by_all_models(chis, data, region, data_org, plot=True):
    ys = {}
    seq = data.get_dna(region, seq=True)
    param_names = ['dna_len', 'mapped_len', 'offset', 'binsize']
    # binsize here is not an actual binsize of Hi-C map but 1/128 of mapped_len
    data_params = {i:getattr(data, i) for i in param_names}
    for org, chi in chis.items():
        actual_params = {i:getattr(chi.data, i) for i in param_names}
        actual_model_data = chi.data
        current_model_org = chi.data.organism
        chi.data = data # replace data for model (DNA, Hi-C)
        for param, value in actual_params.items(): # change its params for this model
            setattr(chi.data, param, value)
        #ys_chi = []
        try:
            y = chi.predict_region(
                    region,
                    plot=False,
                    edge_policy='cyclic',
                    mutations=None,
                    shifts=4
                    )
            save_pred(y, data_org, org)
            if plot:
                _, ax = plt.subplots(figsize=(20,5))
                ax.set_title(chi.data.organism+' predicted by '+current_model_org+' model')
                plot_map(y, ax=ax)
                plt.show()
        except:
            print(f"!!! Couldnt make a prediction of {data_org} by {org}")
        #ys[org] = ys_chi
        for param, value in data_params.items(): # set params back
            setattr(chi.data, param, value)
        chi.data = actual_model_data # set dataset back
    #return ys


def cross_predict(datasets, saving_path, regions, plot=False):
    for i, (org, dataset) in enumerate(datasets.items()):
        if not os.path.exists(saving_path+org):
            os.mkdir(saving_path+org)
        data = dataset.data
        print(org)
        chrom = data.chromnames[0]
        region = regions[org]
        #region = f'{chrom}:{start}-{end}'
        hic = data.get_hic(region, edge_policy='cyclic')
        mask = data.get_mask(region, edge_policy='cyclic')
        if plot:
            y_plot = hic.copy()
            y_plot[mask>0.2] = np.nan
            _, ax = plt.subplots(figsize=(20,5))
            ax.set_title('True '+ data.organism)
            plot_map(y_plot, ax=ax)
            plt.show()
        save_pred(hic, org, root=saving_path)
        save_pred(mask, org, mask=True, root=saving_path)
        predict_by_all_models(datasets, data, region, org, plot)
        torch.cuda.empty_cache()
        gc.collect()
    
def cross_correlate(y, y_correct, mask, n_shifts, one_diag, max_size_ratio):
    rs_control = []
    target_len = y_correct.shape[1] # shape (n diagonals, taget map length)
    zoom_rate = target_len / y.shape[1]
    if max_size_ratio:
        if (zoom_rate < 1/max_size_ratio) or (max_size_ratio > zoom_rate):
            return np.nan, np.nan
    y = zoom(y, (zoom_rate, zoom_rate, 1), order=1)
    step = target_len // n_shifts
    n_diag = min(y.shape[0], y_correct.shape[0])
    if one_diag: # compares only one diagonal, the farest present in the both maps
        y_correct_slice = y_correct[-n_diag:-n_diag+1]
        y_slice = y[-n_diag-n_diag+1]
        mask_slice = mask[-n_diag-n_diag+1] < 0.2 # map region with <20% interpolated pixels
    else:
        y_correct_slice = y_correct[-n_diag:]
        y_slice = y[-n_diag:]
        mask_slice = mask[-n_diag:] < 0.2
    ys = [y_slice[:, (i*step):(i+1)*step] for i in range(n_shifts)]
    ys_correct = [y_correct_slice[:, (i*step):(i+1)*step] for i in range(n_shifts)]
    masks = [mask_slice[:, (i*step):(i+1)*step] for i in range(n_shifts)]
    rs = [spearmanr(i[m].flat, j[m].flat).statistic for i,j,m in zip(ys, ys_correct, masks)]
    ys_shuffled = np.random.permutation(ys)
    rs_control = [spearmanr(i[m].flat, j[m].flat).statistic for i,j,m in zip(ys_shuffled, ys_correct, masks)]
    r = np.median(rs)
    p = mannwhitneyu(rs, rs_control).pvalue
    return r, p

def make_matrix(datasets, folder, species_list, names, use_groundtruth=False, n_controls=20, one_diag=False, max_size_ratio=None):
    all_rs = []
    all_ps = []
    significance = []
    if not folder.endswith('/'):
        folder = folder + '/'
    for org in species_list:
        folder = folder+org+'/'
        groundtruth = np.load(folder+'groundtruth.npy')#datasets[org].data.get_hic(regions[org], edge_policy='cyclic')#
        data_quality_mask = np.load(folder+'mask.npy')
        #data_quality_mask = datasets[org].data.get_mask(regions[org], edge_policy='cyclic')
        proper_pred = np.load(folder+f'predicted_by_{org}_model.npy')

        length1 = groundtruth.shape[1]
        length2 = proper_pred.shape[1]
        if not (0.99 < length1 / length2 < 1.01):
            print('Warning: shapes are different')
        length = min(length1, length2)
        groundtruth = groundtruth[:, :length]
        proper_pred = proper_pred[:, :length]
        data_quality_mask = data_quality_mask[:, :length]
        mask = data_quality_mask
        for org2 in species_list:
            try:
                cross_pred = np.load(folder+f'predicted_by_{org2}_model.npy')
                standard = groundtruth if use_groundtruth else proper_pred
                r, p = cross_correlate(cross_pred, standard, mask, n_controls, one_diag, max_size_ratio)
            except FileNotFoundError:
                print(f'file for {org} predicted by {org2} not found')
                r, p = np.nan, np.nan
            all_rs.append(r)
            all_ps.append(p)

    all_rs = np.array(all_rs)
    all_ps = np.array(all_ps)
    all_ps[~np.isnan(all_ps)] = false_discovery_control(all_ps[~np.isnan(all_ps)]) # all_ps*sum(~np.isnan(all_ps)) #bonferroni correction #
    all_ps[np.isnan(all_ps)] = 0 # not to convert nan rs into zeros at the next step
    all_rs[all_ps > 0.05] = 0
    all_rs[np.isnan(all_rs)] = 0
    matrix = all_rs.reshape(len(species_list), len(species_list)).T
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.xticks(np.arange(len(species_list)), names, rotation=90)
    plt.yticks(np.arange(len(species_list)), names)
    for i in range(len(species_list)):
        for j in range(len(species_list)):
            text = f'{matrix[j,i]:.2f}' if not (np.isnan(matrix[j,i]) or matrix[j,i]==0) else ''
            plt.text(i, j, text, ha='center', va='center')
    return matrix
  
def tree(mtx, names, saving_folder='')
    mtx[np.diag_indices(len(names))]=1
    dissimilarity = 1 - np.nanmean([mtx, mtx.T], axis=0)
    matrix = [list(row)[:i+1] for i,row in enumerate(dissimilarity)]
    dm = DistanceMatrix(names, matrix)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)
    Phylo.write(tree, os.path.join(saving_folder, 'tree.xml'), "phyloxml")
