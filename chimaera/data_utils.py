# -*- coding: utf-8 -*-
import os
import gc

import numpy as np
from tensorflow.keras.utils import to_categorical
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import interpn
from collections import defaultdict

def _parse_index(i, max_i):
    if isinstance(i, tuple):
        if len(i) == 2 and isinstance(i[1], int):
            i, shift = i
    else:
        shift = 0
    if isinstance(i, slice):
        start = i.start if i.start else 0
        start = start if start >= 0 else max_i - start
        stop = i.stop if i.stop else max_i
        stop = stop if stop >= 0 else max_i - stop
        stop = min(stop, max_i)
        iterator = range(start, stop)
    elif isinstance(i, int):
        iterator = [i]
    else:
        iterator = i
    return iterator

def _revcomp(dna):
    revcomp_site = []
    pairs = {'a':'t','c':'g','g':'c','t':'a','n':'n'}
    return ''.join(map(lambda base: pairs[base], dna))[::-1]

def _parse_coord(coord):
    split = coord.split(':')
    if len(split) == 2:
        chrom, coord = split
    else:
        chrom, coord = split[0], ''
    chrom = chrom.strip()
    coord = coord.split('-')
    if len(coord) == 2:
        start, end = coord
        start = _parse_coord_digit(start)
        if start is None:
            start = 0
        end = _parse_coord_digit(end)
        return (chrom, start, end)
    elif len(coord) == 1:
        mid = _parse_coord_digit(coord[0])
        if mid is None:
            return (chrom, 0, None)
        else:
            return (chrom, mid)
    else:
        start, end = coord[1], coord[2]
        start = _parse_coord_digit(start)
        start *= -1
        end = _parse_coord_digit(end)
        return (chrom, start, end)

def _parse_coord_digit(digit):
    if digit == '':
        return None
    else:
        return int(digit.replace(',', '').strip())

def one_hot(seqs):
    alphabet = {'a' : 0, 'c' : 1, 'g' : 2, 't' : 3, 'n' : 4}
    if isinstance(seqs, str):
        seqs = [seqs]
    seqs = [[alphabet[i] for i in seq] for seq in seqs]
    return to_categorical(seqs)[...,:4]

'''def one_hot(seq):
    mapping = dict(zip("acgtn", range(5)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(5)[seq2][:,:4]'''

def get_region(matrix, chrom_name, start, end):
        '''Get Hi-C map by genome position'''
        block = matrix.fetch(f'{chrom_name}:{start}-{end}')
        if block.dtype == 'int32':
            block = block.astype('float32')
        return block

def normalize_by_diag_means(chrom_slice):
    w = chrom_slice.shape[1]
    for diag in range(w):
        diag_means = [np.nanmean(chrom_slice[:, diag+i, i]) for i in range(w-diag)]
        diag_mean = np.nanmean(diag_means)
        for i in range(w-diag):
            chrom_slice[:, diag+i, i] -= diag_mean
            chrom_slice[:, i, diag+i] -= diag_mean
            chrom_slice[:, i, i] = 1
    return chrom_slice

def interp_nan(a_init, pad_zeros=True, method="linear"):
    '''Interpolates NaNs. Code from cooltools https://cooltools.readthedocs.io/en/latest/'''
    shape = np.shape(a_init)
    if pad_zeros:
        a = np.zeros(tuple(s + 2 for s in shape))
        a[tuple(slice(1, -1) for _ in shape)] = a_init
    else:
        a = np.array(a_init)

    isnan = np.isnan(a)
    if np.sum(isnan) == 0:
        return a_init
    if np.any(isnan[:, 0] | isnan[:, -1]) or np.any(isnan[0, :] | isnan[-1, :]):
        raise ValueError("Edges must not have NaNs")
    # Rows/cols to be considered fully null may have non-NaN diagonals
    # so we'll take the maximum NaN count to identify them
    n_nans_by_row = np.sum(isnan, axis=1)
    n_nans_by_col = np.sum(isnan, axis=0)
    i_inds = np.where(n_nans_by_row < np.max(n_nans_by_row))[0]
    j_inds = np.where(n_nans_by_col < np.max(n_nans_by_col))[0]
    if np.sum(isnan[np.ix_(i_inds, j_inds)]) > 0:
        raise AssertionError("Found additional NaNs")

    loc = np.where(isnan)
    a[loc] = interpn(
        (i_inds, j_inds),
        a[np.ix_(i_inds, j_inds)],
        loc,
        method=method,
        bounds_error=False
    )

    if pad_zeros:
        a = a[tuple(slice(1, -1) for _ in shape)]

    return a

def make_chimeric_dna(data, table, length, edge_policy='error'):
    '''Concatenate DNA from regions in some table'''
    dna_list = []
    for _, line in table.iterrows():
        region = f'{line.chrom}:{line.start}-{line.end}'
        dna_list.append(data.get_dna(region, seq=True, edge_policy=edge_policy))
    lens = [len(i) for i in dna_list]
    while True:
        l = 0
        index = []
        while l < length:
            i = np.random.randint(len(lens))
            index.append(i)
            l += lens[i]
        dna = ''.join([dna_list[j] for j in index])[:length]
        yield dna

##########################
### Data slicing tools ###
##########################

class DataLoader():
    def __init__(self, data, regions, check=None, edge_policy='error'):
        self.data = data
        self.regions = regions
        self.edge_policy = edge_policy
        self.shape = (None,)
        self.fun = None
        self.check = check
        self.step = None
        if check:
            self._check_regions()

    def _check_regions(self):
        self.regions = [i for i in self.regions if self._check(i)]


    def _check(self, region):
        if self.check == 'train':
            return self.data._check_train(*region)
        elif self.check == 'not-train':
            return self.data._check_test(*region)

    def __getitem__(self, i):
        iterator = _parse_index(i, len(self.regions))
        batch = np.zeros((len(iterator), *self.shape[1:]))
        for i,j in enumerate(iterator):
            region = self.regions[j]
            batch[i] = self.fun(*region, edge_policy=self.edge_policy)
        return batch

    def __len__(self):
        return len(self.regions)



class DNALoader(DataLoader):
    '''Slices DNA'''
    def __init__(self, data, regions, check=None, edge_policy='error'):
        super(DNALoader, self).__init__(data, regions, check=check, edge_policy=edge_policy)
        self.shape = (None, data.dna_len, 4)
        self.fun = data._slice_dna

class MutantDNALoader(DataLoader):
    '''Slices DNA from mutant genome (used for multiple mutations)'''
    def __init__(self, data, mutant_genome, regions, check=None, edge_policy='error'):
        super(MutantDNALoader, self).__init__(data, regions, check=check, edge_policy=edge_policy)
        self.shape = (None, data.dna_len, 4)
        self.DNA = mutant_genome
        self.fun = self._slice_dna

    def _slice_dna(self, chrom, start, end, seq=False, edge_policy='error', fixed_size=True):
        '''returns one-hot encoded dna from the given region'''
        rev_comp = False
        if start > end:
            rev_comp = True
            start, end = end, start
        # if fixed_size:
        #     end = start + self.shape[1]
        chrom_len = len(self.DNA[chrom])
        if start < 0:
            dna = self.DNA[chrom][0:end]
        else:
            dna = self.DNA[chrom][start:end]
        if start < 0 or end > chrom_len:
            if edge_policy == 'error': # raises error
                raise ValueError(f'Coordinates {start, end} exсeed chrom {chrom} \
chrom size ({chrom_len})')
            elif edge_policy == 'empty':  # adds masked bins
                if start < 0:  
                    left_empty = 'n'*(-start)
                    dna = left_empty + dna
                if end > chrom_len:
                    right_empty = 'n'*(end-chrom_len)
                    dna = dna + right_empty
            elif edge_policy == 'cyclic': # adds bins from the other end
                if start < 0:  
                    left_empty = dna[start:]
                    dna = left_empty + dna
                if end > chrom_len:
                    right_empty = dna[:end-chrom_len]
                    dna = dna + right_empty
            else:
                raise ValueError("edge_policy should be 'error', 'empty' or 'cyclic'") 
        if rev_comp: #
            dna = _revcomp(dna) # !!! revcomp in sequence
        if seq:
            return dna
        else:
            dna = one_hot(dna)
            #if rev_comp:
            #   dna = np.flip(dna)
            return dna

class ChimericDNALoader():
    '''Slices DNA from concatenated fragments (e.g. all intergenic regions)'''
    def __init__(self, data, table, length, size, edge_policy='error'):
        self.length = length
        self.size = size
        self.gen = make_chimeric_dna(data, table, length, edge_policy=edge_policy)

    def __getitem__(self, i):
        iterator = _parse_index(i, None)
        batch = np.zeros((len(iterator), self.length, 4))
        for i, _ in enumerate(iterator):
            batch[i] = one_hot(next(self.gen))
        return batch

    def __len__(self):
        return self.size

class MapDataLoader(DataLoader):
    '''Slices Hi-C maps and their masks of interpolated bins'''
    def __init__(self, data, regions, check):
        super(MapDataLoader, self).__init__(data, regions, check)
        if data.square_maps:
            w = data.square_w//2
            self.shape = (None, w, w, data.out_channels)
        else:
            self.shape = (None, data.h, data.map_size, data.out_channels)
        self.regions = self.translate_regions(self.regions)
        self.initial_regions = self.regions

    def translate_regions(self, dna_regions):
        hic_regions = []
        for region in dna_regions:
            chrom, start, end = region
            hic_start, hic_end = self.data._corresp_map_for_a_dna(start, end)
            hic_region = (chrom, hic_start, hic_end)
            hic_regions.append(hic_region)
        return hic_regions


class MaskLoader(MapDataLoader):
    '''Slices masks of interpolated bins'''
    def __init__(self, data, regions, check=None):
        super(MaskLoader, self).__init__(data, regions, check)
        self.fun = data._slice_mask

class HiCLoader(MapDataLoader):
    '''Slices Hi-C maps'''
    def __init__(self, data, regions, check=None):
        super(HiCLoader, self).__init__(data, regions, check)
        self.fun = data._slice_map
        self.mask = MaskLoader(data, regions)

#############################
### Mini-batch generators ###
#############################

class DataGenerator():
    '''Generates data batches'''
    def __init__(self, x, y=None, batch_size=1, mask=None):
        self.x = x
        self.y = y
        self.mask = mask
        self.batch_size = batch_size
        self.indices = np.arange(len(x))

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        x = self.x[indices]
        if self.y is not None:
            y = self.y[indices]
            if self.mask is not None:
                mask = self.mask[indices]
                x, y, mask = self.transform(x, y, mask)
                return x, y, mask
            else:
                return x, y
        else:
            x = self.transform(x)
            #print(x.shape)
            return x

    def shuffle(self):
        np.random.shuffle(self.indices)

    def transform(self, *args):
        return args


class DNATrainValGenerator(DataGenerator):
    '''Generates batches of dna and hic pairs'''
    def __init__(self, x, y, mask=None, batch_size=1, rc=0, sigma=0):
        super(DNATrainValGenerator, self).__init__(
            x=x,
            y=y,
            mask=mask,
            batch_size=batch_size
        )
        self.rc = float(rc)
        self.sigma = sigma

    def transform(self, x, y, mask=None):
        if self.sigma:
            y = gaussian_filter(y, sigma=(0,self.sigma,self.sigma,0))
        if np.random.rand() < self.rc:
            return self.rev_comp(x, y, mask)
        else:
            return x, y, mask

    def rev_comp(self, x, y, mask):
        x = np.flip(x, axis=(1, 2))
        y = np.flip(y, axis=2)
        if mask is not None:
            mask = np.flip(mask, axis=2)
        return x, y, mask


class DNAPredictGenerator(DataGenerator):
    '''Generates DNA batches for prediction'''
    def __init__(self, x, batch_size=1, rc=False):
        super(DNAPredictGenerator, self).__init__(
            x=x,
            y=None,
            batch_size=batch_size
        )
        self.rc = rc

    def transform(self, x):
        if self.rc:
            return self.rev_comp(x)
        else:
            return x

    def rev_comp(self, x):
        return np.flip(x, axis=(1, 2))

class HiCTrainValGenerator(DataGenerator):
    '''Generates Hi-C batches for autoencoder training'''
    def __init__(self, x, mask=None, batch_size=1, random_flip=0):
        super(HiCTrainValGenerator, self).__init__(
            x=x,
            y=x,
            mask=mask,
            batch_size=batch_size
        )
        self.random_flip = float(random_flip)

    def transform(self, x, y, mask=None):
        if x.shape[-1] > 1:
            # DNA model predicts all cell types in the dataset simultaneously
            # but autoencoder works with single maps. To train it we select
            # cell type randomly for each map in a batch
            rand_inds = np.random.choice(x.shape[-1], x.shape[0])
            x = np.array([x[i,...,j] for i,j in enumerate(rand_inds)])[...,None]
        if np.random.rand() < self.random_flip:
            x = self.flip(x, axis=2)
            if mask is not None:
                mask = self.flip(mask, axis=2)
        return x, x, mask

    def flip(self, x):
        return np.flip(x, axis=2)

class HiCCustomGenerator(DataGenerator):
    '''Generates Hi-C batches along the chromosome'''
    def __init__(self,
                 hic,
                 mask,
                 threshold=1,
                 step=1,
                 channel=0,
                 w=128,
                 batch_size = 8,
                 skip_empty_center=False,
                 empty_bins_offset=0,
                 square_maps=False
                 ):
        self.hic = hic
        self.mask = mask
        self.batch_size = batch_size
        self.skip_empty_center = skip_empty_center
        self.threshold = threshold
        self.step = step
        self.channel = channel
        self.offset = empty_bins_offset
        self.w = w
        self.valid_list = []
        self.view_mask = self._get_view_mask(square_maps)

    def _get_view_mask(self, square_maps):
        if square_maps:
            w = self.hic.shape[1]
            a = np.zeros((w,w), dtype=bool)
            a[:w//2, w//2:] = True
            a[w//2:, :w//2] = True
        else:
            h = self.hic.shape[0]
            a = np.zeros((h,self.w), dtype=bool)
            for i in range(h):
                a[-i-1, self.w//2-i-1:self.w//2+i+1] = True
        return a

    def __len__(self):
        return (self.hic.shape[1] - self.w) // self.step // self.batch_size

    def __getitem__(self, index):
        batch = []
        for i in range(index * self.batch_size * self.step,
                    (index + 1) * self.batch_size * self.step,
                    self.step):
            # check if region is valid by NaN score (overall and in cental bins)
            # if not it will be predicted but not included in valid_list
            good = True
            if self.skip_empty_center:
                center = (i+self.w//2-self.offset, i+self.w//2+self.offset+1)
                left, right = center
                if np.all(self.mask[-1, left:right] > 0.1):
                    good = False
            nans_in_view_field = self.mask[:, i:i+self.w, self.channel][self.view_mask]
            if nans_in_view_field.mean() >= self.threshold:
                good = False
            if good:
                self.valid_list.append(i//self.step)
            batch.append(self.hic[:, i:i+self.w, self.channel][...,None])

        x = np.array(batch)
        return x

class MultiSpeciesHiCGenerator(HiCTrainValGenerator):
    '''Generates Hi-C batches for autoencoder training
on many datasets at the same time'''
    def __init__(self, loaders, batch_size=1, random_flip=0):
        super(MultiSpeciesHiCGenerator, self).__init__(
            x=loaders,
            batch_size=batch_size,
            random_flip=random_flip
        )
        self.total_len = sum([len(i) for i in self.x])
        self.cumsum_len = np.cumsum([len(i) for i in self.x])
        self.indices = np.arange(self.total_len)

    def __len__(self):
        return int(np.ceil(self.total_len / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        index_dict = defaultdict(list)
        for ind in indices:
            loader_index = np.where(self.cumsum_len > ind)[0][0]
            if loader_index > 0:
                ind -= self.cumsum_len[loader_index-1]
            index_dict[loader_index].append(ind)
        x = []
        for loader_index, map_indices in index_dict.items():
            maps = self.x[loader_index][map_indices]
            x.append(maps)
        x = np.concatenate(x)
        x, _ = self.transform(x, x)
        return x, x
