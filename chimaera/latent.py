import numpy as np
from pandas import DataFrame
from collections import defaultdict
from scipy.spatial.distance import cdist

from . import plot_utils
from . import data_utils
import matplotlib.pyplot as plt

def insulation_mask(h=1):
    mask = np.zeros((32,128))
    h = int(32*h)
    for i in range(h):
        mask[-i-1, 64-i-1:64+i+1] = -0.4
    return mask

def loop_mask(h=0.5):
    mask = np.zeros((32,128))
    h = int(32*(1-h))
    for i in range(4):
        mask[max(h-i,0), 64-8+i:64+8-i] = 1
        mask[min(h+i,31), 64-8+i:64+8-i] = 1
    return mask

def fountain_mask(h=0.75):
    mask = np.zeros((32,128))
    h = int(32*h)
    for i in range(h):
        mask[-i-1, 64-i//2-1:64+i//2+1] = 1/3
    return mask

def tad_mask(h=1):
    mask = np.zeros((32,128))
    h = int(32*(1-h))
    for i in range(32-h):
        mask[i+h, 64-i-1:64+i+1] = 1/3
    return mask

def stripe_mask(h=1):
    mask = np.zeros((32,128))
    h = int(32*h)
    for i in range(h):
        mask[-i-1, 64-i-1:64-i+1] = 0.5
    return mask

def mask_to_vec(model, mask):
    if isinstance(mask, str):
        if mask == 'insulation':
            mask = insulation_mask()
        elif mask == 'loop':
            mask = loop_mask()
        elif mask == 'fountain':
            mask = fountain_mask()
        else:
            raise ValueError('Mask type not recognized')
    if mask.ndim == 3:
        mask = mask[...,None]
    elif mask.ndim == 2:
        mask = mask[None,...,None]
    vec = model.hic_to_latent(mask)
    #mask_recon = model.latent_to_hic(vec_raw)
    #vec = model.hic_to_latent(mask_recon)
    vec /= np.linalg.norm(vec)
    return vec

def vec_to_mask(model, vec):
    if vec.ndim == 1:
        vec = vec[None,:]
    mask = model.latent_to_hic(vec)
    plot_utils.plot_map(mask)
    return mask

def proj(model, maps, vecs):
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    vecs /= np.linalg.norm(vecs, axis=1)[:, None]
    if maps.ndim == 3:
        if maps.shape[2] <= model.data.out_channels:
            maps = maps[None,...]
        else:
            maps = maps[...,None]
    elif maps.ndim == 2:
        maps = maps[None,...,None]
    map_vecs = model.hic_to_latent(maps)

def _make_basic_vecs(model, vector):
    masks=[]
    if vector == 'insulation':
        m = latent.insulation_mask()
    elif vector == 'fountain':
        m = latent.fountain_mask()
    elif vector == 'loop':
        m = latent.loop_mask()
    for i in range(64):
        masks.append(np.concatenate([m[:,i:],np.zeros((32,i))], axis=1))
    for i in range(64):
        masks.append(np.concatenate([np.zeros((32,i+1)),m[:,:-i-1]], axis=1))
        return model.hic_to_latent(np.array(masks)[...,None])

def check_mask_reconstruction(model, mask):
    vec = mask_to_vec(model, mask)
    mask_recon = vec_to_mask(model, vec)
    return np.corrcoef(mask.flat, mask_recon.flat)[0,1]

def proj(model, maps, vecs):
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    vecs /= np.linalg.norm(vecs, axis=1)[:, None]
    if maps.ndim == 3:
        if maps.shape[2] <= model.data.out_channels:
            maps = maps[None,...]
        else:
            maps = maps[...,None]
    elif maps.ndim == 2:
        maps = maps[None,...,None]
    map_vecs = model.hic_to_latent(maps)
    return map_vecs.dot(vecs.T)

def scan_projections(model, vecs, central_bins_only=True,
                     vec_names=None, step=1,
                     nan_threshold=0.1, skip_empty_center=True,
                     empty_bins_offset=1,
                     metric='projection',
                     chroms=None,
                     return_latent=False):
    outputs = []
    for channel in range(model.data.out_channels):
        if isinstance(vecs, list) or isinstance(vecs, tuple):
            vecs = np.concatenate(vecs)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        vecs /= np.linalg.norm(vecs, axis=1)[:, None]
        if vec_names is None:
            vec_names = ['vec'+str(i) for i in range(len(vecs))]
        elif isinstance(vec_names, str):
            vec_names = [vec_names]
        df_dict = defaultdict(list)
        w = model.data.map_size
        if chroms is None:
            chroms = model.data.chromnames
        for chrom in chroms:
            print(chrom+':')
            gen = data_utils.HiCCustomGenerator(
                model.data.HIC[chrom],
                model.data.MASKS[chrom],
                threshold=nan_threshold,
                skip_empty_center=skip_empty_center,
                empty_bins_offset=empty_bins_offset,
                w=w,
                channel=channel,
                step=step,
            )
            pred_vecs = model.hic_to_latent(gen, verbose=True)
            valid_predictions = pred_vecs[gen.valid_list]
            if metric=='projection':
                scores = valid_predictions.dot(vecs.T)
            else:
                scores = cdist(valid_predictions, vecs, metric)
            map_starts = [i*step for i in gen.valid_list]
            map_ends = [i+w for i in map_starts]

            translate = model.data._hic_coord_to_genomic
            starts = np.array([translate(i) for i in map_starts])
            ends = np.array([translate(i) for i in map_ends])
            if central_bins_only:
                starts = (starts + ends) // 2 - model.data.binsize
                ends = starts + model.data.binsize
            df_dict['chrom'] += [chrom] * len(scores)
            df_dict['map_start'] += map_starts
            df_dict['map_end'] += map_ends
            df_dict['start'] += list(starts)
            df_dict['end'] += list(ends)
            if return_latent:
                df_dict['latent'] += list(valid_predictions)
            for score, name in zip(scores.T, vec_names):
                df_dict[name] += list(score)
        outputs.append(DataFrame(df_dict))
    if len(outputs) == 1:
        return outputs[0]
    return outputs

def analyze_projections(data, dfs):
    n_vecs = len(dfs[0].columns)-5
    figsize = (n_vecs * 13, len(dfs) * 2)
    fig, axs = plt.subplots(nrows=len(dfs), ncols=n_vecs*2, figsize=figsize)
    if len(dfs) == 1:
        plot_utils.show_min_max_projections(data, 0, dfs[0], ax=axs)
    else:
        for i, df in enumerate(dfs):
            plot_utils.show_min_max_projections(data, i, df, ax=axs[i])
        plot_utils.plot_corr(dfs, data.experiment_names)

