import os
import gc

import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate, zoom, gaussian_filter
from scipy import interpolate
from itertools import product

from IPython.display import clear_output

from cooler import Cooler
from cooltools.lib.numutils import interp_nan#, observed_over_expected, adaptive_coarsegrain
from Bio import SeqIO
import bioframe

from .plot import *

class Dataset(object):
    """Loads data for training

hic_file: path to file with Hi-c maps;
genome_file_or_dir: path to genome fasta file or to folder
    with chromosomes' files;
fragment_length: length of a DNA fragment;
sigma: sigma in gaussian filter for maps;
chroms_to_exclude: chomosomes not used in training;
chroms_to_include: chomosomes used in training - if both args
    are not None chroms_to_include is a priority;
scale: None (no scaling) or tuple of ints, scaling values in Hi-C maps.
    Note that it affects the decoder last layer activation function;
normalize: scale using global min&max ('gloabal') or each scal map
    separately ('each');
min_max: min and max for scaling, is used when adding smth to
    the dataset, technical arguement;
map_size: map size in pixels;
nan_threshold: highest permissible percentage of missing values in
    a map;
rev_comp: stochastic reverse complement while training;
stochastic_sampling: sample with random shifts along the genome while
    training (works not good in most cases);
shift_repeats: sample with fixed shifts along the genome while
    training (e.g. if shift_repeats is 4, each position in the genome will be
    presented in the sample for times in fragments with a shift of 1/4 of their
    length). Note that this argument increases the size of the dataset;
expand_dna: use DNA context for prediction (a half of a fragment length
    at both sides). Note that this argument increases the size of the dataset twice;
psi: pseudocount being added to normalized Hi-C map before logarithm - you
    may change it to find perfect contrast of maps but default value is suitable for
    most maps we used;
cross_threshold: it is threshhold of column mean in normalized map to 
    consider the column (and corresponding row i.e a cross) as unmapped region and 
    interpolate it using its neighbours. If you see strange blue 'V' pattern in maps
    after transformations they may be artifactы of purely mapped regions but 
    processed without interpolation - then you should enlarge this arguement;
val_split: describing validation sample generation. Is a tuple of a string:
        'first' - first n in sample
        'last' - last n in sample
        'random'
        chromosome name + position of center,
    and a number:
        integer - number of objects
        float < 1 - proportion of the dataset
    examples: ('first', 32) - first 32 objects
              ('last', 0.1) - last 1/10 of the dataset
              ('chr5 2451000', 20) - 20 objects around 2451000 position at chr5;
processed_hic: you may save processed hic on your disc and load it if 
    processing spends a lot of time;
h: higth of rotated map. 32 is preferable.
"""
    def __init__(self,
                 hic_file, 
                 genome_file_or_dir, 
                 fragment_length,
                 sigma = 0,
                 chroms_to_exclude = [],
                 chroms_to_include = [],
                 scale = None,
                 normalize = 'standart',
                 min_max = (None, None),
                 nan_threshold = 0.2,
                 stochastic_sampling = False,
                 shift_repeats = 1,
                 interpolator = None,
                 expand_dna = True,
                 psi = .001,
                 cross_threshold = 0.05,
                 val_split = ('first', 32),
                 processed_hic = None,
                 h = 32,
                 organism_name = None):

        if stochastic_sampling and (shift_repeats > 1):
            raise ValueError("Stochastic sampling and shift_repeats can't be used together")
        if (val_split[0] == 'random') and (stochastic_sampling or (shift_repeats > 1)):
            raise ValueError("Random split is incorrect in not fixed dataset")
        self.nan_threshold = nan_threshold
        self.stochastic_sampling = stochastic_sampling
        self.shift_repeats = shift_repeats
        self.expand_dna = expand_dna
        self.val_split = val_split
        self.chroms_to_exclude = chroms_to_exclude
        self.chroms_to_include = chroms_to_include
        self.min_max = min_max
        self.cross_threshold = cross_threshold
        self.h = h
        self.psi = psi
        self.sd = None

        self.scale = scale if scale is not None else (0,1)
        self.normalize = normalize
        self.sigma = sigma
        self.interpolator = interpolator

        self.mapped_len = fragment_length
        self.map_size = 128
        self.offset = 0 if not self.expand_dna else self.mapped_len // 2
        self.dna_len = self.mapped_len + self.offset * 2
        self.resolution = self.mapped_len // self.map_size
        self.max_dist = self.mapped_len // (self.map_size // self.h // 2)

        self.cooler = Cooler(hic_file)
        self.matrix = self.cooler.matrix(balance=True)
        self.organism = organism_name
        self.assembly = genome_file_or_dir.split('/')[-1].split('.')[0]

        self.names = self.choose_chroms(self.cooler.chromnames, chroms_to_include, chroms_to_exclude)
        self.DNA = self.make_dna(genome_file_or_dir, self.names)
        self.gc_content, self.trans_mtx = 0,0#self.calc_stats()
        if processed_hic is None:
            self.HIC = self.make_hic()
            #self.save_hic('./')
            #print('Making Hi-c dataset is long, you may save it by .save_hic method to use it after')
        else:
            self.HIC = {name: np.load(processed_hic + name + '.npy') for name in self.names}
        X, y = self.make_dataset()
        initial_samples = self.train_val_split(X, y)
        initial_samples = self.expand_dataset(initial_samples)
        self._x_train, self._x_val, self._y_train, self._y_val = self.cleanup(*initial_samples)
        self.x_train, self.y_train  = DNALoader(self, self._x_train), HiCLoader(self, self._y_train, normalize)
        self.x_val, self.y_val  = DNALoader(self, self._x_val), HiCLoader(self, self._y_val, normalize)
        self.scale_hic()

        '''self.names, self.DNA = self.make_dna(genome_file_or_dir, chroms_to_exclude, chroms_to_include)
        self.gc_content, self.trans_mtx = self.calc_stats()
        if processed_hic is None:
            self.HIC = self.make_hic()
            #self.save_hic('./')
            #print('Making Hi-c dataset is long, you may save it by .save_hic method to use it after')
        else:
            self.HIC = {name: np.load(processed_hic + name + '.npy') for name in self.names}
        X, y = self.make_dataset()
        initial_samples = self.train_val_split(X, y)
        initial_samples = self.expand_dataset(initial_samples)
        self._x_train, self._x_val, self._y_train, self._y_val = self.cleanup(*initial_samples)
        self.x_train, self.y_train  = DNALoader(self, self._x_train), HiCLoader(self, self._y_train)
        self.x_val, self.y_val  = DNALoader(self, self._x_val), HiCLoader(self, self._y_val)'''
        

        self.params =  {'hic_file': hic_file, 
                        'genome_file_or_dir': genome_file_or_dir,
                        'fragment_length': fragment_length,
                        'chroms_to_exclude': self.chroms_to_exclude,
                        'chroms_to_include': self.chroms_to_include,
                        'sigma': sigma,
                        'scale': scale,
                        'nan_threshold': nan_threshold,
                        'stochastic_sampling': stochastic_sampling,
                        'shift_repeats': shift_repeats,
                        'h': h,
                        'normalize': normalize,
                        'psi': psi,
                        'expand_dna': expand_dna,
                        'val_split': val_split,}
    
    def choose_chroms(self, available, include, exclude):
        if include:
            return [i for i in available if i in include]
        if exclude:
            return [i for i in available if i not in exclude]
        else:
            return available

    def make_dna(self, genome_file_or_dir, chroms):
        '''Makes a dict of chromosomes' sequences'''
        DNA = dict()
        if os.path.isdir(genome_file_or_dir):
            available_files = os.listdir(genome_file_or_dir)
            extension = available_files[0].split('.')[1]
            for chrom in chroms:
                file_name = chrom + '.' + extension
                if file_name in available_files:
                    if self.cooler.chromsizes[chrom] < self.dna_len * 4:
                        print(f"Chromosome {chrom} is too short so it can't be used")
                        self.names.remove(chrom)
                    else:
                        full_path = os.path.join(genome_file_or_dir, file_name)
                        fasta = next(SeqIO.parse(full_path, "fasta"))
                        DNA[chrom] = str(fasta.seq).lower()
                        print(f'DNA data for {fasta.name} is loaded')
                        del fasta
        else:
            gen = SeqIO.parse(genome_file_or_dir, "fasta")
            n_found = 0
            print('The order of chromosomes loading will be as in genome file but in sample the order will be as in .cool file')
            for fasta in gen:
                chrom = fasta.name
                if chrom in chroms:
                    if self.cooler.chromsizes[chrom] < self.dna_len * 4:
                        print(f"Chromosome {chrom} is too short so it can't be used")
                        self.names.remove(chrom)
                    n_found += 1
                    DNA[chrom] = str(fasta.seq).lower()
                    print(f'DNA data for {chrom} is loaded')
                del fasta
                if len(chroms) == n_found:
                    break
        for chrom in chroms:
            if not (chrom in DNA.keys()):
                self.names.remove(chrom)
                print(f'No DNA found for {chrom}')
        gc.collect()
        print()
        return DNA
    
    def calc_stats(self):
        l = 0
        gc = 0
        transition_mtx = np.zeros(16).astype(int)
        for chrom in self.DNA.values():
            gc += chrom.count('g') + chrom.count('c')
            l += len(chrom)
            '''for j, dinucl in enumerate(product('acgt', repeat=2)):
                transition_mtx[j] += chrom.count(''.join(dinucl))
        transition_mtx /= transition_mtx.sum()
        transition_mtx = transition_mtx.reshape(4,4)'''
        gc_content = gc / l
        return gc_content, None

    
    def interpolate_nan(self, array):
        '''Interpolating pixels with zero contacts. The slowest step'''
        x = np.arange(0, array.shape[1])
        y = np.arange(0, array.shape[0])
        array = np.ma.masked_invalid(array)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]
        return interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')

    def get_region(self, chrom_name, start, end):
        '''Get Hi-C map by genome position'''
        return self.matrix.fetch(f'{chrom_name}:{start}-{end}')
    
    def rotate_and_cut(self, square_map):
        '''

        ▓▓▓░░░░░░░░░░░░
        ░░▓▓▓▓▓▓░░░░░░░        ░░░░░░░░░░░░░░░▓░
        ░░▓▓▓▓▓▓░░░░░░░   ──>  ░░░▓░░░░░░░░░▓▓▓▓
        ░░░░░░░▓▓▓▓▓▓▓▓        ░▓▓▓▓▓░░░░░▓▓▓▓▓▓
        ░░░░░░░▓▓▓▓▓▓▓▓        ▓▓▓▓▓▓▓▓░▓▓▓▓▓▓▓▓
        ░░░░░░░▓▓▓▓▓▓▓▓

        '''
        square_map = zoom(square_map, 181 / len(square_map), order=1) # 181 ~ 128 * sqrt(2)
        rotated_map = rotate(square_map, 45, order=1)                      
        cut_map = rotated_map[len(rotated_map) // 2 - self.h : len(rotated_map) // 2,
                                len(rotated_map) // 4 : -len(rotated_map) // 4]
        return cut_map
    
    def calculate_insulation_score(self):
        pass

    def make_hic(self):
        '''Makes continuous Hi-C datasets for each chromosome. It is a stripe
along the main diagonal.'''
        HIC = dict.fromkeys(self.names)
        for name in self.names:
            chrom_len = self.cooler.chromsizes[name]
            assert chrom_len == len(self.DNA[name])
            chrom_hic = []
            quality = []
            for start in range(0, chrom_len-self.mapped_len*2, self.mapped_len):
                new_block = self.get_region(name, start, start + self.mapped_len * 2)
                new_block[np.isnan(new_block)] = 0

                # We have some genome bins with zero or too little mapped reads
                # but it doesn't mean they realy have no contacts in cell.
                # We will work with them separately (as not zeros but nans)
                # and interpolate them using their neighbours
                cross = np.mean(new_block != 0, axis=0) < self.cross_threshold
                new_block[cross] = np.nan
                new_block[:,cross] = np.nan

                new_block += self.psi

                result = np.log(new_block)
                if self.interpolator is None:
                    result = interp_nan(result)
                    result = self.rotate_and_cut(result)[-self.h:]
                else:
                    result[cross] = -2
                    result[:,cross] = -2
                    result = zoom(result, 181 / len(result), order=1) # 181 ~ 128 * sqrt(2)
                

                # Tracing nan crosses after rotation and count their projections:
                new_block[cross] = -5
                new_block[:, cross] = -5
                new_block = self.rotate_and_cut(new_block)#[-self.h:]
                
                new_block = np.log(new_block)
                block_quality = np.isnan(new_block).mean(axis=0)
                # Now we have information how many pixels were interpolated in 
                # each column this data can be used as quality - if a map 
                # contains too many interpolated pixels, we will trow it out

                chrom_hic.append(result)
                quality.append(block_quality)
            
            quality = np.hstack(quality)
            if self.interpolator is not None:
                chrom_hic = self.interpolator.predict(np.array(chrom_hic))[..., 0]
                chrom_hic = [self.rotate_and_cut(i, a=False)[-self.h:] for i in chrom_hic]
            chrom_hic = np.hstack(chrom_hic)
            chrom_hic -= np.mean(chrom_hic, axis=1)[...,None]
            if self.sigma:
                chrom_hic = gaussian_filter(chrom_hic, sigma = self.sigma)

            HIC[name] = np.vstack([quality, chrom_hic])
            del chrom_hic, quality
            print(f'Hi-C data for {name} is loaded')
        gc.collect()
        return HIC
    
    def save_hic(self, folder):
        for name,array in self.HIC.items():
            np.save(os.path.join(folder, name), array)
        print('Saved at ' + folder)

    def make_dataset(self):
        '''Makes arrays of corresponded positions in the DNA and in stripe of
the Hi-C map. Doesn't do any operations with data itself.'''
        hic_list = []
        dna_list = []
        for n, name in enumerate(self.names):
            chrom_len = self.cooler.chromsizes[name]
            start_pos = self.offset
            end_pos = chrom_len - self.offset - self.mapped_len * 3 # with a margin
            for start in range(start_pos, end_pos, self.mapped_len):
                map_pos = int(start / self.mapped_len * self.map_size)
                hic_list.append(np.array([n, map_pos, map_pos + self.map_size]))
                dna_list.append(np.array([n, start - self.offset + self.mapped_len // 2,
                                          start + self.mapped_len + self.offset + self.mapped_len // 2]))
        return np.array(dna_list), np.array(hic_list)
    
    def train_val_split(self, dna_list, hic_list):
        '''Splits arrays of positions intto train and val samples'''
        assert len(dna_list) == len(hic_list)
        if self.val_split == 'test':
            return dna_list, dna_list, hic_list, hic_list
        method, val_split = self.val_split
        if method.startswith('chr'):
            if val_split < 1:
                val_split = int(len(hic_list) * val_split)
            val_split += 4 # because later 4 elenents will be thrown out

            chrom, pos = method.split()
            chrom_inds = np.where(dna_list[:, 0] == self.names.index(chrom))
            ind_in_chrom = np.where(np.all(((dna_list[chrom_inds][:, 1] < pos),
                                            (dna_list[chrom_inds][:, 2] >= pos)),
                                            axis = 0))
            mid = chrom_inds[0][0] + ind_in_chrom[0][0]
            x_val = [dna_list.pop(mid - val_split // 2) for i in range(val_split)]
            y_val = [hic_list.pop(mid - val_split // 2) for i in range(val_split)]
            # throw out 2 edge elements to avoid intersection with train sample
            data = dna_list, x_val[2:-2], hic_list, y_val[2:-2]
            del hic_list
            return data
        else:
            if method == 'random':
                if val_split > 1:
                    val_split /= len(hic_list)

                data = train_test_split(dna_list, hic_list, val_size = val_split, random_state = 0)
                del hic_list
                return data

            else:
                if val_split < 1:
                    val_split = int(len(hic_list) * val_split)
                # throw out 2 edge elements of train sample to avoid intersection with validation
                if method == 'first':
                    return dna_list[val_split+2:], dna_list[:val_split], hic_list[val_split+2:], hic_list[:val_split]
                elif method == 'last':
                    return dna_list[:-val_split-2], dna_list[-val_split:], hic_list[:-val_split-2], hic_list[-val_split:]


    def expand_dataset(self, initial_samples):
        '''Makes additional objects in each sample by sliding along the genome. 
E.g. if initial sample is [..., 10000-20000, 20000-30000, ...] and shift_repeats
is 3, it will become [..., 10000-20000, 20000-30000, ..., 13333-23333, 
23333-33333, ..., 16667-26667, 26667-26667]. So each region is represented
multiple times but in various context. '''

        if self.shift_repeats > 1:
            _x_train, _x_val, _y_train, _y_val = initial_samples
            x_train, x_val, y_train, y_val = [], [], [], []
            y_shift = self.map_size // self.shift_repeats
            x_shift = int(y_shift * self.mapped_len / self.map_size)
            for i in range(self.shift_repeats):
                x_shift_array = np.array([[0, x_shift*i, x_shift*i]])
                y_shift_array = np.array([[0, y_shift*i, y_shift*i]])
                y_train.append(_y_train + y_shift_array)
                x_train.append(_x_train + x_shift_array)
                y_val.append(_y_val + y_shift_array)
                x_val.append(_x_val + x_shift_array)

            x_train = np.concatenate(x_train)
            x_val = np.concatenate(x_val)
            y_train = np.concatenate(y_train)
            y_val = np.concatenate(y_val)
            return x_train, x_val, y_train, y_val
        else:
            return initial_samples

    def cleanup(self, x_train, x_val, y_train, y_val):
        '''This function removes maps with too many interpolated pixels (based on
nan_threshold) from samples.'''
        good = bad = 0
        good_train = []
        expected_train_size = len(y_train)
        for i in range(len(y_train)):
            n, start, end = y_train[i]
            name = self.names[n]
            # nan percentage for a map is written in the first raw of Hi-C stripe:
            if self.HIC[name][0, start : end].mean() > self.nan_threshold:
                bad += 1
            else:
                good += 1
                good_train.append(i)
        x_train, y_train = x_train[good_train], y_train[good_train]

        good_val = []
        expected_val_size = len(y_val)
        for i in range(len(y_val)):
            n, start, end = y_val[i]
            name = self.names[n]
            # nan percentage for a map is written in the first raw of Hi-C stripe:
            if self.HIC[name][0, start : end].mean() > self.nan_threshold:
                bad += 1
            else:
                good += 1
                good_val.append(i)
        x_val, y_val = x_val[good_val], y_val[good_val]
        clear_output()            
        print(f'{bad/(bad+good)*100:.2f}% of maps were excluded by NaN threshold')
        print(f'Train sample has {len(good_train)} pairs now (of \
{expected_train_size} possible)')
        print(f'Validation sample has {len(good_val)} pairs now \
(of {expected_val_size} possible)')
        return x_train, x_val, y_train, y_val

    def scale_hic(self):
        '''Scale Hi-C maps' values to some interval'''
        if self.normalize == 'minmax':
            chrom_mins = []
            if self.min_max == (None, None):
                y_train = self.y_train[:]
                total_min = y_train.min()
                total_max = y_train.max()
            for i in self.HIC.values():
                i[1:] = (i[1:] - total_min) / (total_max - total_min)

            if self.scale != (0, 1):
                for i in self.HIC.values():
                    i[1:] *= self.scale[1] - self.scale[0]
                    i[1:] += self.scale[0]
            gc.collect()
        else:
            self.q005 = 0
            self.q995 = 1
            self.sd = self.y_train[:].std()
            y = self.y_train[:]
            q005 = np.quantile(y, 0.005)
            q995 = np.quantile(y, 0.995)
            self.q005, self.q995 = q005, q995

    def calculate_insulation_score(self, window=10):
        resolution = self.cooler.binsize
        insulation_table = insulation(self.cooler, [window * resolution], 
                                      ignore_diags=None, min_dist_bad_bin=0,
                                      verbose=True)
        y_train_is = []
        y_val_is = []
        l = self.mapped_len // resolution
        for bin in self._x_train:
            chrom = self.names[bin[0]]
            start = bin[1] + self.offset
            end = bin[2] - self.offset
            region = (chrom, start, end)
            column = [f'log2_insulation_score_{window * resolution}']
            insul_region = bioframe.select(insulation_table, region)[column]
            y_train_is.append(insul_region[:l])
        for bin in self._x_val:
            chrom = self.names[bin[0]]
            start = bin[1] + self.offset
            end = bin[2] - self.offset
            region = (chrom, start, end)
            column = [f'log2_insulation_score_{window * resolution}']
            insul_region = bioframe.select(insulation_table, region)[column]
            y_val_is.append(insul_region[:l])
        y_train_is = np.array(y_train_is)
        y_val_is = np.array(y_val_is) 
        for arr in [y_val_is, y_train_is]:
            arr[np.isnan(arr)] = 0
            arr[arr > 2] = 2
            arr[arr < -2] = -2
        return y_train_is[...,0], y_val_is[...,0]

    def load_bed(self, bed_file, bins, name):
        bed = pd.read_csv(bed_file)

        x_train = self._x_train.copy()
        x_train[:, 0] = [self.names[i] for i in x_train[:, 0]]
        x_train = pd.DataFrame(self._x_train)

        x_val = self._x_val.copy()
        x_val[:, 0] = [self.names[i] for i in x_val[:, 0]]
        x_val = pd.DataFrame(self._x_val)
        print(f'Data loaded into .{name}_train and .{name}_val attributes')

    def coord(self,
            x_chrom_start_end=None,
            y_chrom_start_end=None,
            x_pos=None,
            y_pos=None):
        '''
Translate coordinates between .HIC and .DNA and between single fragments of them.
Use ONE arguement of possible:
   x_chrom_start_end: [chrom NUMBER, start, end] in .DNA (list or array) -> 
[chrom NAME, start, end] in .HIC (list)
   y_chrom_start_end: [chrom NUMBER, start, end] in .HIC (list or array) -> 
[chrom NAME, start, end] in .DNA (list)
   x_pos: position in dna fragment (int) -> position in corresponding map (int)
   y_pos: position in map (int) -> position in corresponding dna fragment (int)
'''
        binsize = self.resolution
        y_offset = self.offset // binsize
        if x_chrom_start_end is not None:
            n, start, end = x_chrom_start_end
            chrom = self.names[n]
            y_start = (start + self.offset - self.mapped_len // 2) // binsize
            y_end = (end - self.offset - self.mapped_len // 2) // binsize
            return chrom, y_start, y_end
        elif y_chrom_start_end is not None:
            n, start, end = y_chrom_start_end
            chrom = self.names[n]
            x_start = start * binsize + self.mapped_len // 2 - self.offset
            x_end = end * binsize + self.offset + self.mapped_len // 2
            return chrom, x_start, x_end
        elif x_pos is not None:
            return (x_pos - self.offset) // binsize
        elif y_pos is not None:
            return y_pos * binsize + self.offset
    
    def plot_quality(self):
        for name, chrom in self.HIC.items():
            plt.figure(figsize=(15,4))
            print(name)
            plt.plot(chrom[0])
            plt.show()
    
    def plot_annotated_map(self,
                        sample,
                        number,
                        ax=None,
                        y=None,
                        name=None,
                        axis='both',
                        x_position='bottom',
                        show_position=True,
                        full_name=True,
                        mutations=None,
                        mutation_names=None,
                        genes=None,
                        gene_names=None,
                        motifs=None,
                        motif_names=None,
                        chrom=None,
                        start=None,
                        end=None,
                        colorbar=True,
                        show=True,
                        vmin=None,
                        vmax=None,
                        **kwargs):
        
        if ax is None:
            _, ax = plt.subplots()
        
        if y is None:
            if sample=='train':
                y = self.y_train[number]
            elif sample=='val':
                y = self.y_val[number]
                
        y = get_2d(y)
        im = ax.imshow(y,
                       extent=[0, y.shape[1], y.shape[0], 0],
                       cmap="hic_cmap",
                       interpolation='none',
                       vmin=vmin,
                       vmax=vmax,
                       **kwargs)

        if colorbar:
            annotate_colorbar(im, ax, vmin, vmax)
        
        if start is None:
            if sample=='train':
                chrom, start, end = self.coord(y_chrom_start_end=self._y_train[number])
            elif sample=='val':
                chrom, start, end = self.coord(y_chrom_start_end=self._y_val[number])
            start += self.offset
            end -= self.offset

        
        cbottom, ctop = annotate(ax, self.offset, self.mapped_len+self.offset,
                                 axis, h=self.h,
                                 w=self.map_size, position=x_position)
        if show_position:
            if full_name:
                ctop = annotate_coord(ax, chrom, start, end,
                                      organism=self.organism,
                                      assembly=self.assembly,
                                      position='top', constant=ctop)
            else:
                ctop = annotate_coord(ax, chrom, start, end,
                                      position='top', constant=ctop)
        if mutations is not None:
            map_positions = ((mutations - self.offset) / self.resolution).astype(int)
            cbottom = annotate_mutations(ax, positions=map_positions,
                                         dna_positions=mutations,
                                         names=mutation_names, constant=cbottom)
        if genes is not None:
            map_positions = ((genes - self.offset) / self.resolution).astype(int)
            cbottom = annotate_genes(ax, positions=genes, names=gene_names,
                                     constant=cbottom)
        if name:
            ax.set_ylabel(name)
            
        if show:
            plt.show()


class DNALoader():

    """Saves memory and helps to make array of one-hot-encoded DNA batch
    only when it is needed. Input data of model is stored in DNA string 
    and array of indices"""

    def __init__(self, data, input_data):
        self.DNA = data.DNA
        self.names = data.names
        self.dna_len = data.dna_len
        self.input_data = input_data
        self.len = len(self.input_data)
        self.alphabet = {'a' : 0, 'c' : 1, 'g' : 2, 't' : 3, 'n' : 4}

    def one_hot(self, seq):
        return to_categorical([self.alphabet[i] for i in list(seq)])[:,:4]
        
    def __getitem__(self, i):
        if isinstance(i, tuple):
            if len(i) == 2 and isinstance(i[1], int):
                i, shift = i
        else:
            shift = 0
        if isinstance(i, slice):
            start = i.start if i.start else 0
            start = start if start >= 0 else self.len - start
            stop = i.stop if i.stop else self.len    
            stop = stop if stop >= 0 else self.len - stop    
            stop = min(stop, len(self))
            iterator = range(start, stop)
        elif isinstance(i, int):
            iterator = [i]
        else:
            iterator = i
        batch = []
        for j in iterator:
            chrom, start, _ = self.input_data[j]
            start += shift
            end = start + self.dna_len
            seq = self.DNA[self.names[chrom]][start : end]
            batch.append(self.one_hot(seq))
        return np.array(batch)

    def __len__(self):
        return self.len


class HiCLoader():
    '''Works similar to DNALoader, returns fragments of the map stripe by positions'''
    def __init__(self, data, input_data, normalize='minmax'):
        self.data = data
        self.HIC = data.HIC
        self.normalize = normalize
        self.names = data.names
        self.input_data = input_data
        self.len = len(self.input_data)
    
    def show(self, i):
        plot_map(self[i])
    
    def show_orig(self, n):
        a = self.data.coord(y_chrom_start_end=self.input_data[n])
        orig_map = np.log(self.data.matrix.fetch(f'{a[0]}:{a[1]}-{a[2]}')+self.data.psi)
        plt.imshow(orig_map, cmap='Reds')
        c = len(orig_map)
        a=c//4
        plt.plot([a,a+a//2, c-a//2, c-a, a],[a, a//2,c-a-a//2, c-a, a], c='k')
        plt.xticks([a, c-a])
        plt.yticks([a, c-a])
        plt.grid(True)
        plt.show()

    def __getitem__(self, i):
        if isinstance(i, tuple):
            if len(i) == 2 and isinstance(i[1], int):
                i, shift = i
        else:
            shift = 0
        if isinstance(i, slice):
            start = i.start if i.start else 0
            start = start if start >= 0 else self.len - start
            stop = i.stop if i.stop else self.len    
            stop = stop if stop >= 0 else self.len - stop    
            stop = min(stop, len(self))
            iterator = range(start, stop)
        elif isinstance(i, int):
            iterator = [i]
        else:
            iterator = i
        batch = []
        for j in iterator:
            chrom, start, end = self.input_data[j]
            start += shift
            end += shift
            hic_map = self.HIC[self.names[chrom]][1:, start : end]
            batch.append(hic_map)
        batch = np.array(batch)[...,None]
        if self.normalize == 'standart' or self.normalize == 'both':
            mu = batch.mean(axis=(1,2,3))[:,None,None,None]
            sd = self.data.sd 
            if sd is not None:
                batch = (batch - mu) / sd
            else: 
                batch = batch - mu
            if self.normalize == 'both':
                batch = (batch - self.data.q005) / (self.data.q995 - self.data.q005)
                if self.data.q005 != 0:
                    batch[batch > 0.99] = 0.99
                    batch[batch < 0.01] = 0.01
        return batch

    def __len__(self):
        return self.len
