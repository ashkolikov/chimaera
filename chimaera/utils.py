import gc
import os
import numpy as np
from sklearn.model_selection import train_test_split
import bioframe
from cooltools.lib.numutils import observed_over_expected, adaptive_coarsegrain, interp_nan, set_diag
from scipy.ndimage import rotate, zoom, gaussian_filter
from tensorflow.keras.utils import to_categorical # for one-hot encoding of DNA
from Bio import SeqIO

def rescale_ys(y, scale=(0, 1), min_max=(None, None), norm_regime='global'):
    '''
    min_max scaling that normalized Hi-C maps to the given scale interval.
    Note that "norm_regime" does nothing if the min_max is default. Can be "each" or "global".
    '''

    if min_max==(None, None) and norm_regime=='each':
        # Normalizing each snippet separately if there is no default min-max:
        y_min = np.min(y, axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
        y_max = np.max(y, axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
        total_min = total_max = (None, None)
    else:
        if min_max!=(None, None):
            # Default min-max provided, no difference between 'global' and 'each':
            total_min = min_min[0]
            total_max = min_max[1]
        elif norm_regime == 'global':
            # Normalize the whole dataset
            total_min = y.min()
            total_max = y.max()
        else:
            raise ValueError(f"Normalization regime '{norm_regime}' not supported. Available: 'each' and 'global'")

        y_min = total_min
        y_max = total_max

    y -= y_min
    y /= y_max

    if scale != (0, 1):
        y *= scale[1] - scale[0]
        y += scale[0]

    gc.collect() # TODO: do we need it?
    return y, (total_min, total_max)

def split_data(bin_table, method, params={}):
    """
    Split bin_table into coordinates for train and test.
    bin_table: pandas dataframe with genomic bins
    method: string with the name of the splitting method. Available: 'test', 'chr', 'random', 'first', 'last'
    params: dictionary of splitting parameters:
        'val_split' is the percentage of data used for validation (for 'random', 'first' and 'last')
        'random_seed' is for 'random'
        'chroms' is for 'chr' ! Not implemented yet!
    Output:
        _x_train, _x_val, _y_train, _y_val indexes in the bin table

    _x_train==to _y_train and _x_val==_y_val
    to follow sklearn-style train-test split, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    if method == 'test':
        # Return unchanged set:
        idxs = bin_table.index.values
        return idxs, idxs, idxs, idxs
    elif method.startswith('chr'): # Any usecase?
        raise ValueError('Method "chr" is not implemented yet!')
        # if val_split < 1:
        #     val_split = int(len(hic_list) * val_split)
        # val_split += 2 # because then two elenents will be thrown out
        #
        # chrom, pos = method.split()
        # chrom_inds = np.where(dna_list[:, 0] == self.names.index(chrom))
        # ind_in_chrom = np.where(np.all(((dna_list[chrom_inds][:, 1] < pos), (dna_list[chrom_inds][:, 2] >= pos)), axis = 0))
        # mid = chrom_inds[0][0] + ind_in_chrom[0][0]
        # x_val = [dna_list.pop(mid - val_split // 2) for i in range(val_split)]
        # y_val = [hic_list.pop(mid - val_split // 2) for i in range(val_split)]
        # # cut edge elements to avoid intersection with train dataset
        # data = dna_list, x_val[1:-1], hic_list, y_val[1:-1]
        # del hic_list
        # return data
    elif method == 'random':
        val_split = params["val_split"]
        random_state = params.get("random_state", 0) # TODO: default to None?
        if val_split < 1:
            val_split = int(len(bin_table) * val_split)

        data = train_test_split(bin_table.index.values, bin_table.index.values,
                                test_size=val_split,
                                train_size=len(bin_table)-val_split,
                                random_state=random_state)
        return data

    elif method=='first' or method=='last':
        val_split = params["val_split"]
        if val_split < 1:
            val_split = int(len(bin_table) * val_split)
        idx = bin_table.index.values
        if method == 'first':
            data = idx[val_split:], idx[:val_split], idx[val_split:], idx[:val_split]
        elif method == 'last':
            data = idx[:-val_split], idx[-val_split:], idx[:-val_split], idx[-val_split:]
        return data

    else:
        raise ValueError(f"Split method '{method}' is not supported. Available: 'first', 'chr', 'random', 'first', 'last'")

# def fetch_regions(clr, window_table, viewframe):
#     '''Get Hi-C snippets for the positions from the window_table.
#     clr: input cooler
#     window_table: dataframe with selected genomic windows
#     regions: viewframe with chromosome sizes
#     '''
#
#     # Assign genomic regions to windows:
#     windows = snipping.assign_regions(window_table, viewframe)
#
#     snipper = cooltools.snipping.CoolerSnipper(clr, regions=viewframe)
#     stack = cooltools.snipping.pileup(
#         windows,
#         snipper.select,
#         snipper.snip)
#
#     return
#
#     mtx_raw = self.balanced.fetch(f'{name}:{start}-{end}')
#     mtx_balanced = self.not_balanced.fetch(f'{name}:{start}-{end}')
#     return self.transform_hic(mtx_raw, mtx_balanced)


def transform_hic_old(hic_matrix_raw, hic_matrix):
    '''Transformations on Hi-C maps; deprecated'''
    transformed_arr = adaptive_coarsegrain(hic_matrix_raw, hic_matrix)
    nan_mask = np.isnan(transformed_arr)
    transformed_arr, _,_,_ = observed_over_expected(transformed_arr, mask = ~nan_mask)
    transformed_arr = np.log(transformed_arr)
    transformed_arr = interp_nan(transformed_arr)
    return transformed_arr, np.mean(nan_mask)


def chimaera_transform(snippet_raw, snippet_bal,
                min_frac_valid_pixels=0.75,
                remove_diag=2,
                fill_diag=0,
                fill_missing=0,
                sigma=None,
                params_coarsegrain={},
                params_ooe={},
                params_interp={},
                ):
    """
    Transform raw and balanced snippets.
    Applies (1) adaptive coarsegrain,
            (2) OOE,
            (3) interpolate nans,
            (4) log,
            (5) fill in first diagonals, and
            (6) fill in missing pixels (e.g. log(0) ).
    Forced return array full of nans in two cases:
    F1) Ratio of valid pixels below threshold
    F2) nan interpolations filled it (probably due to singletons)

    Available parameters:
    :param min_frac_valid_pixels: float, fraction of the pixels valid in snippet for interpolation
    :param remove_diag: integer, number of diagonals to fill in; set 0 for diagonals untouched
    :param fill_diag: integer or np.float32 or np.nan, value to fill in the diagonals
    :param fill_missing: int or np.float32 or np.nan, value to fill in missing data (e.g. log(0))
    :param params_corasegrain: dict with default cutoff=5, max_levels=8, min_shape=8, see
https://github.com/open2c/cooltools/blob/210f6b94b4aecfb821eb9553d2708839130f73a7/cooltools/lib/numutils.py#L1198
    :param params_ooe: dict with default dist_bin_edge_ratio=1.03, see
https://github.com/open2c/cooltools/blob/210f6b94b4aecfb821eb9553d2708839130f73a7/cooltools/lib/numutils.py#L564
    :param params_interp: dict with default pad_zeros=True, method="linear", verbose=False, see
https://github.com/open2c/cooltools/blob/210f6b94b4aecfb821eb9553d2708839130f73a7/cooltools/lib/numutils.py#L201
    :return: transformed snippet
    """

    # (1) adaptive coarsegrain
    snippet = adaptive_coarsegrain(ar=snippet_bal, countar=snippet_raw, **params_coarsegrain)

    # F1) check number of valid pixels
    nan_mask = np.isnan(snippet)
    if np.sum(~nan_mask) < min_frac_valid_pixels * snippet.shape[0] * snippet.shape[1]:
        # print('Small number of valid pixels')
        snippet = np.full(snippet.shape, np.nan)
        return snippet

    # (2) OOE
    snippet, _, _, _ = observed_over_expected(snippet, mask=~nan_mask, **params_ooe)

    # (3) interpolates nans
    # F2) check number of valid pixels
    try:
        snippet = interp_nan(snippet, **params_interp)  # TODO: replace this interpolation?
    except AssertionError as e: # non-recoverable snippet
        print('Failed interpolation', e)
        snippet = np.full(snippet.shape, np.nan)
        return snippet

    # (4) log
    snippet = np.log(snippet)

    # (5) fill in first diagonals
    for diag in range(remove_diag):
        set_diag(snippet, fill_diag, i=diag, copy=False)

    # fill in missing pixels (e.g. log(0) )
    snippet[~np.isfinite(snippet)] = fill_missing

    # TODO: think of moving this out of the transforming function:
    if sigma is not None:
        snippet = gaussian_filter(snippet, sigma=sigma)

    return snippet


###### Snipper class that performes transformation of snippets on a fly.
###### Inherited from cooltools.snipping:
class TransformingSnipper:
    def __init__(self, clr, transform, view_df=None, padding=2, params_transform={}):
        """
        Snipper for snipping the area with padding (measured in pixels)
         and applying transformations of transform handler to it.
        """

        # get chromosomes from bins, if view_df not specified:
        if view_df is None:
            view_df = bioframe.make_viewframe(
                [(chrom, 0, l, chrom) for chrom, l in clr.chromsizes.items()]
            )
        else:
            # appropriate viewframe checks:
            if not bioframe.is_viewframe(view_df):
                raise ValueError("view_df is not a valid viewframe.")
            if not bioframe.is_contained(
                view_df, bioframe.make_viewframe(clr.chromsizes)
            ):
                raise ValueError(
                    "view_df is out of the bounds of chromosomes in cooler."
                )

        self.view_df = view_df.set_index("name")

        self.clr = clr
        self.transform = transform
        self.params_transform = params_transform
        self.binsize = self.clr.binsize
        self.offsets = {}
        self.padding = padding

    def select(self, region1, region2):
        region1_coords = self.view_df.loc[region1]
        region2_coords = self.view_df.loc[region2]
        self.offsets[region1] = self.clr.offset(region1_coords) - self.clr.offset(
            region1_coords[0]
        )
        self.offsets[region2] = self.clr.offset(region2_coords) - self.clr.offset(
            region2_coords[0]
        )
        matrix_raw = self.clr.matrix(balance=False, sparse=True).fetch(region1_coords, region2_coords).tocsr()
        matrix_bal = self.clr.matrix(balance=True,  sparse=True).fetch(region1_coords, region2_coords).tocsr()
        self._isnan1 = np.isnan(self.clr.bins()["weight"].fetch(region1_coords).values)
        self._isnan2 = np.isnan(self.clr.bins()["weight"].fetch(region2_coords).values)
        return (matrix_raw, matrix_bal)

    def snip(self, matrices, region1, region2, tup):
        matrix_raw, matrix_bal = matrices
        s1, e1, s2, e2 = tup
        offset1 = self.offsets[region1]
        offset2 = self.offsets[region2]
        binsize = self.binsize
        lo1, hi1 = (s1 // binsize) - offset1, (e1 // binsize) - offset1
        lo2, hi2 = (s2 // binsize) - offset2, (e2 // binsize) - offset2
        dm, dn = hi1 - lo1, hi2 - lo2

        # Introduce padding:
        lo1 -= self.padding
        hi1 += self.padding
        lo2 -= self.padding
        hi2 += self.padding

        assert hi1 >= 0
        assert hi2 >= 0

        m, n = matrix_raw.shape
        out_of_bounds = False
        if lo1 < 0:
            out_of_bounds = True
        if lo2 < 0:
            out_of_bounds = True
        if hi1 > m:
            out_of_bounds = True
        if hi2 > n:
            out_of_bounds = True

        if out_of_bounds:
            snippet = np.full((dm, dn), np.nan)
        else:
            snippet_raw = matrix_raw[lo1:hi1, lo2:hi2].toarray().astype("int")
            snippet_bal = matrix_bal[lo1:hi1, lo2:hi2].toarray().astype("float")
            snippet_bal[self._isnan1[lo1:hi1], :] = np.nan
            snippet_bal[:, self._isnan2[lo2:hi2]] = np.nan

            snippet = self.transform(snippet_raw, snippet_bal, **self.params_transform)
            # Remove padding after all the transformations:
            snippet = snippet[self.padding:-self.padding, self.padding:-self.padding]

        return snippet

def load_fasta(genome_path, clr, exclude_chroms, include_chroms):
    '''Constructor of a dict of chromosomes' sequences'''
    dna = dict()
    names = []
    if os.path.isdir(genome_path):
        files_in_dir = os.listdir(genome_path)
        if include_chroms != []:
            availible_files = [i for i in files_in_dir if
                               (i.split('.')[0] in include_chroms and i.split('.')[0] in clr.chromnames)]
        elif exclude_chroms != []:
            availible_files = [i for i in files_in_dir if
                               (i.split('.')[0] not in exclude_chroms and i.split('.')[0] in clr.chromnames)]
        else:
            print("If it's possible you should select not all chromosomes for test&val samples")
            availible_files = [i for i in files_in_dir if i.split('.')[0] in clr.chromnames]

        for file in availible_files:
            fasta = next(SeqIO.parse(os.path.join(genome_path, file), "fasta"))
            dna[fasta.name] = str(fasta.seq).lower()
            names.append(fasta.name)
            print(f'dna data for {fasta.name} is loaded')
            del fasta
    else:
        gen = SeqIO.parse(genome_path, "fasta")
        for fasta in gen:
            name = fasta.name
            if include_chroms != []:
                if name not in include_chroms or name not in clr.chromnames:
                    del fasta
                    continue
            elif exclude_chroms != []:
                if name in exclude_chroms or name not in clr.chromnames:
                    del fasta
                    continue
            # else:
            #     print('You should select not all chromosomes for test&val samples')
            dna[name] = str(fasta.seq)
            names.append(name)
            print(f'dna data for {fasta.name} is loaded')
            del fasta
    gc.collect()
    return dna, names


def cast_samples(idx_x_train, idx_x_val, idx_y_train, idx_y_val, nsamples=9, seed=None):
    '''Make a small subsample of the validation sample for visulization'''
    if len(idx_y_val)<nsamples:
        return idx_x_train, idx_x_val, idx_y_train, idx_y_val
    # if seed is None:
    #     np.random.seed(self.sample_seed)
    # elif
    if seed == 'random':
        pass
    else:
        np.random.seed(seed)
    train_inds = np.random.choice(len(idx_y_train), nsamples, replace = False)
    val_inds = np.random.choice(len(idx_y_val), nsamples, replace = False)
    x_train_sample, y_train_sample = idx_x_train[train_inds], idx_y_train[train_inds]
    x_val_sample, y_val_sample = idx_x_val[val_inds], idx_y_val[val_inds]

    return x_train_sample, x_val_sample, y_train_sample, y_val_sample


def one_hot(seq, alphabet):
    """
    One-hot encoded sequence of DNA
    :param seq: input DNA sequence
    :param alphabet: alphabet where each letter from seq corresponds to numbers
    :return: one-hot encoded array
    """
    return to_categorical([alphabet[i] for i in list(seq)])[:,:4]
