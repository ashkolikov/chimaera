import numpy as np
import os
import sys
import gc
import cooler
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import torch

from . import plot_utils

def get_region(matrix, chrom_name, start, end):
    '''Get Hi-C map by genome position'''
    block = matrix.fetch(f'{chrom_name}:{start}-{end}')
    if block.dtype == 'int32':
        block = block.astype('float32')
    return block

def analyze_data(cool_file):
    if cool_file.endswith('.cool'):
        cool = cooler.Cooler(cool_file)
        resolutions = [cool.binsize]
    elif cool_file.endswith('.mcool'):
        cool = []
        resolutions = cooler.fileops.list_coolers(cool_file)
        resolutions = [i.split('/')[-1] for i in resolutions]
        cool = cooler.Cooler(cool_file + f'::resolutions/{resolutions[0]}')
    matrix = _get_matrix(cool)
    genome_size = cool.chromsizes.sum()
    supposed_max_fragment_length = genome_size // 150
    supposed_min_fragment_length = cool.binsize * 50
    pows = list(range(int(np.log2(supposed_min_fragment_length)),
                      int(np.log2(supposed_max_fragment_length))+1))
    resolutions = [int(i) for i in resolutions if supposed_max_fragment_length/int(i) > 20]
    fig, ax = plt.subplots(nrows=len(resolutions),
                           ncols=len(pows),
                           figsize=(len(pows)*2,
                                    len(resolutions)*2))
    if not isinstance(ax, np.ndarray):
        ax = np.array([[ax]])
    if len(ax.shape) == 1:
        if len(resolutions) == 1:
            ax = ax[None, :]
        else:
            ax = ax[:, None]

    found_good_chrom = False
    chrom_index = 0
    while not found_good_chrom:
        res = np.median(resolutions)
        pow = np.median(pows)
        fragment_len = 2**int(pow)
        chrom = cool.chromnames[chrom_index]
        chrom_middle = cool.chromsizes[chrom] // 2
        start = max(chrom_middle - fragment_len // 2, 0)
        end = min(chrom_middle + fragment_len // 2, cool.chromsizes[chrom])
        region = f'{chrom}:{start}-{end}'
        example = matrix.fetch(region)
        if np.isnan(example).mean() > 0.25:
            chrom_index += 1
        else:
            found_good_chrom = True
    chrom = cool.chromnames[chrom_index]
    chrom_middle = cool.chromsizes[chrom] // 2

    for i, res in enumerate(resolutions):
        if len(resolutions) == 1:
            cool = cooler.Cooler(cool_file)
        else:
            cool = cooler.Cooler(cool_file + f'::resolutions/{res}')
        matrix = _get_matrix(cool)
        for j, pow in enumerate(pows):
            fragment_len = 2**int(pow)
            ax[i,j].set_title(f'Res = {res}\nfragment len = \n{fragment_len}')
            size = fragment_len // res
            if size < 16 or size > 800:
                if size > 800:
                    text = 'Too large'
                else:
                    text = 'Too small'
                ax[i,j].imshow(np.array([[np.nan]]))
                ax[i,j].text(0,0,text, ha='center', fontsize=11)
                ax[i,j].set_xticks(())
                ax[i,j].set_yticks(())
                [i.set_color('red') for i in ax[i,j].spines.values()]
                [i.set_linewidth(2) for i in ax[i,j].spines.values()]
                continue

            start = max(chrom_middle - fragment_len // 2, 0)
            end = min(chrom_middle + fragment_len // 2, cool.chromsizes[chrom])
            region = f'{chrom}:{start}-{end}'
            example = matrix.fetch(region)
            ax[i,j].imshow(np.log(example+0.001), cmap='Reds', interpolation='none')
            w = example.shape[1]
            ax[i,j].set_xticks((0, w//2, w))
            ax[i,j].set_yticks((0, w//2, w))
            if fragment_len > 1500000:
                 t = ax[i,j].annotate('?', xy=(size/2,size/2),
                                      xytext=(size/2,size/2),
                                      ha='center', va='center',
                                      fontsize=70, color='grey')
                 t.set_alpha(.4)
            [i.set_linewidth(2) for i in ax[i,j].spines.values()]
            contact_score = min((example>0).mean()**2, 1)
            number_score = min(genome_size / fragment_len / 800, 1)
            size_score = min(example.shape[0] / 32, 1)
            score = contact_score * size_score * number_score
            #print(f'Res={res}, len={fragment_len}, score={score:.3f}')
            if score < 0.25:
                [i.set_color('red') for i in ax[i,j].spines.values()]
            elif score < 0.5:
                [i.set_color('yellow') for i in ax[i,j].spines.values()]
            else:
                [i.set_color('green') for i in ax[i,j].spines.values()]
    print('''Green outline - recommended (good training forecast), yellow -
possibly model won't train, red - model hardly will train.
Score includes map sparsity, genome size and single window size, thresholds
selected empirically.
Maps with side < 20 are not shown because they are not informative, with > 800 -
redundantly large because all maps are then brought to the same size.
Grey '?' means dna fragment is >1.5M b.p. and if you dont't have a GPU with
large RAM, maximal possible mini-batch size may be not sufficient for correct
training.''')
    plt.tight_layout()
    plt.show()

def _prepare_dir(saving_path, rewrite):
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)
    else:
        files_in_saving_dir = os.listdir(saving_path)
        if files_in_saving_dir:
            npy_files = [i for i in files_in_saving_dir if i.endswith('.npy')]
            if npy_files:
                if not rewrite:
                    raise ValueError("Saving directory already has .npy files \
and 'rewrite' arg is False. Set it True if you want rewriting these files")
                else:
                    for i in npy_files:
                        os.remove(os.path.join(saving_path, i))

def _with_resolution(cool_file, resolution):
    if resolution is not None:
        if cool_file.endswith('.mcool'):
            cool_file = cool_file + f'::resolutions/{resolution}'
        elif cool_file.endswith('.cool'):
            res = cooler.Cooler(cool_file).binsize
            if res != int(resolution):
                print(f'WARNING: file {cool_file} has only one resolution -\
{res}. It differs from required ({resolution})')
    else:
        if cool_file.endswith('.mcool'):
            raise ValueError(f'file {cool_file} has many resolutions - \
select one')
    return cool_file

def _select_chromnames(clr, chroms, cool_file):
    chromnames = clr.chromnames

    if chroms is not None:
        chromnames = [i for i in chromnames if i in chroms]
        if len(chromnames) == 0:
            chrs = ', '.join(clr.chromnames)
            raise ValueError(f"None of listed chroms is found.\
Found chroms are: {chrs}. If names are incorrect, \
rename them with correct names using 'rename_dict' argument")
        elif len(chromnames) < len(chroms):
            print(f'WARNING: some of listed chroms are not found in {cool_file}')
    return chromnames

def _get_matrix(clr):
    if "weight" in clr.bins().columns:
        return clr.matrix(balance=True)
    else:
        print('WARNING: Unable to open balenced matrix - using unbalanced. \
It may lead to incorrect results')
        return clr.matrix(balance=False)

def rough_balance(mtx):
    q = np.quantile(mtx, 0.8, axis=1) == 0
    mtx = mtx/mtx.sum(axis=1)[:,None] + mtx/mtx.sum(axis=1)[None,:]
    mtx[np.isinf(mtx)] = np.nan
    mtx[:, q] = np.nan
    mtx[q, :] = np.nan
    return mtx

def pool(mtx, rate=2):
    mtx[np.isnan(mtx)] = 0
    mtx = torch.Tensor(mtx[None, None, ...])
    pooled_mtx = torch.nn.AvgPool2d((rate,rate))(mtx).numpy()[0,0]
    return pooled_mtx


def preprocess(
        cool_files,
        saving_path,
        fragment_length,
        resolution=None,
        chroms=None,
        rename_dict=None,
        rewrite=False,
        autobalance=False,
        pooling_rate=1
    ):
    if not isinstance(cool_files, list) or isinstance(cool_files, tuple):
        cool_files = [cool_files]
    if not isinstance(cool_files[0], str):
        raise ValueError('cool_files must be strings')
    if isinstance(chroms, str):
        chroms = [chroms]

    # save fragment length and resolution in directory name
    saving_path += '__len=' + str(fragment_length)
    if resolution is not None:
        saving_path += '_res=' + str(resolution)
    _prepare_dir(saving_path, rewrite)

    # get availible chromosomes from the fist file
    # (considered other files also contain them)
    clr0 = cooler.Cooler(_with_resolution(cool_files[0], resolution))
    chromnames = _select_chromnames(clr0, chroms, cool_files[0])
    chromsizes = {name:clr0.chromsizes[name] for name in chromnames}

    for name_in_file in chromnames:
        chrom_slices = []
        chrom_len = chromsizes[name_in_file]
        if rename_dict is not None:
            if name_in_file in rename_dict.keys():
                name = rename_dict[name_in_file]
            else:
                name = name_in_file
        else:
            name = name_in_file
        print('Chrom '+name+' processing:')
        if chrom_len < fragment_length*2:
            print(f'Chromosome {name} is too short')
            continue
        for cool_file in cool_files:
            cool_file = _with_resolution(cool_file, resolution)
            clr = cooler.Cooler(cool_file)
            if name_in_file not in clr.chromnames:
                raise ValueError(f'Chrom {name} is absent in at least one file')
            matrix = _get_matrix(clr)

            w = fragment_length * 2 // clr.binsize
            arange = range(0, chrom_len - fragment_length * 2, fragment_length)
            n = len(arange)
            chrom_slices_single_cooler = np.zeros((n,
                                                   w//pooling_rate,
                                                   w//pooling_rate))
            for i, start in enumerate(arange):
                new_block = get_region(
                    matrix,
                    name_in_file,
                    start,
                    start + fragment_length * 2
                )
                if len(new_block) != w:
                    zoom_rate = w / len(new_block)
                    mask = np.isnan(new_block)
                    new_block[mask] = 0
                    new_block = zoom(new_block, zoom_rate, order=1)
                    mask = zoom(mask.astype(float), zoom_rate, order=1)
                    new_block[mask>0] = np.nan
                if pooling_rate > 1:
                    new_block = pool(new_block, pooling_rate)
                if autobalance or pooling_rate > 1:
                    new_block = rough_balance(new_block)
                chrom_slices_single_cooler[i] = new_block
                del new_block
            chrom_slices.append(chrom_slices_single_cooler)
        filename = name + '_' + str(chrom_len) + '.npy'
        full_filename = os.path.join(saving_path, filename)
        chrom_slices = np.stack(chrom_slices, axis=-1)
        np.save(full_filename, chrom_slices)
        del chrom_slices, chrom_slices_single_cooler
        gc.collect()
        print(f'{name} is loaded')