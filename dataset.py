import os
import gc

import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate, zoom, gaussian_filter
from scipy import interpolate
from itertools import product

from cooler import Cooler
from cooltools.lib.numutils import interp_nan#, adaptive_coarsegrain, observed_over_expected,
from Bio import SeqIO
import bioframe

#from .plot import *

def get_region(matrix, chrom_name, start, end):
        '''Get Hi-C map by genome position'''
        block = matrix.fetch(f'{chrom_name}:{start}-{end}')
        if block.dtype == 'int32':
            block = block.astype('float32')
        return block

def make_arr_from_cool(cool_file,
                       fragment_length,
                       resolution=None,
                       saving_path=None):
    if resolution is not None:
        cool_file = cool_file + f'::resolutions/{resolution}'
    cooler = Cooler(cool_file)
    if saving_path is not None:
        if not os.path.isdir(saving_path):
            os.mkdir(saving_path)
    try:
        matrix = cooler.matrix(balance=True)
        matrix.fetch(f'{cooler.chromnames[0]}:0-{cooler.chromsizes[0]//1000}')
    except:
        print('Unable to open balenced matrix - using unbalanced. It may lead to incorrect results')
        matrix = cooler.matrix(balance=False)
    
    chromnames = cooler.chromnames
    hic_cuts = dict()
    for name in chromnames:
        chrom_len = cooler.chromsizes[name]
        if chrom_len < fragment_length*2:
            print(f'Chromosome {name} is too short')
            continue
        chrom_hic = []
        w = fragment_length*2 // cooler.binsize
        arange = range(0, chrom_len - fragment_length * 2, fragment_length)
        n = len(arange)
        hic_cuts[name] = np.zeros((n,w,w))
        for i, start in enumerate(arange):
            new_block = get_region(matrix,
                                   name,
                                   start,
                                   start + fragment_length * 2)
            if len(new_block > w):
                new_block = new_block[:w, :w]
            hic_cuts[name][i] = new_block
        filename = name + '_' + str(chrom_len) + '.npy'
        full_filename = os.path.join(saving_path, filename)
        np.save(full_filename, hic_cuts[name])
        print(f'{name} is loaded')
    gc.collect()
    return cooler, hic_cuts

class DataMaster(object):
    """Loads data for training

hic_data_path: path to file with Hi-c maps;
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
                 hic_data_path, 
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
                 remove_first_diag = 2,
                 interpolator = None,
                 expand_dna = True,
                 psi = .001,
                 cross_threshold = 0.05,
                 use_adaptive_coarsegrain = False,
                 val_split = 'default',
                 processed_hic = None,
                 h = 32,
                 organism_name = None):

        if stochastic_sampling and (shift_repeats > 1):
            raise ValueError("Stochastic sampling and shift_repeats can't be used together")
        self.nan_threshold = nan_threshold
        self.stochastic_sampling = stochastic_sampling
        self.shift_repeats = shift_repeats
        self.expand_dna = expand_dna
        self.val_split = val_split
        self.remove_first_diag = remove_first_diag
        self.chroms_to_exclude = chroms_to_exclude
        self.chroms_to_include = chroms_to_include
        self.use_adaptive_coarsegrain = use_adaptive_coarsegrain
        self.min_max = min_max
        self.cross_threshold = cross_threshold
        self.h = h
        self.psi = psi
        self.sd = None
        self.mean = None

        self.cmap = "RdBu_r"

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

        self.hic_data_path = hic_data_path
        if hic_data_path.endswith('cool'):
            self.cooler, self.hic_cuts = make_arr_from_cool(hic_data_path, 
                                                            fragment_length)
            self.chromsizes = self.cooler.chromsizes
            self.names = self.choose_chroms(self.cooler.chromnames,
                                            chroms_to_include,
                                            chroms_to_exclude)
        else:
            self.cooler = None
            self.hic_cuts = dict()
            self.chromsizes = dict()
            all_files = os.listdir(hic_data_path)
            hic_files = [i for i in all_files if i.endswith('npy')]
            self.names = self.choose_chroms([i.split('_')[0] for i in hic_files],
                                            chroms_to_include,
                                            chroms_to_exclude)
            for i in hic_files:
                filename = os.path.join(hic_data_path, i)
                chrom_name_and_size = i.split('.')[0]
                chromname, chromsize = chrom_name_and_size.split('_')
                if chromname in self.names:
                    self.hic_cuts[chromname] = filename
                    self.chromsizes[chromname] = int(chromsize)

        
        self.organism = organism_name
        self.assembly = genome_file_or_dir.split('/')[-1].split('.')[0]

        
        self.DNA = self.make_dna(genome_file_or_dir, self.names)
        dna_lens = {i:len(j) for i,j in self.DNA.items()}
        for name in self.names:
            if self.chromsizes[name] != dna_lens[name]:
                raise ValueError('Chrom sizes in fasta and cool are different')

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
        self.mask_train, self.mask_val = HiCLoader(self, self._y_train, normalize, mask=True), HiCLoader(self, self._y_val, normalize, mask=True)
        #self.scale_hic()

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
        

        self.params =  {'hic_data_path': hic_data_path, 
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
                    if self.chromsizes[chrom] < self.dna_len * 4:
                        print(f"Chromosome {chrom} is too short")
                        self.names.remove(chrom)
                    else:
                        full_path = os.path.join(genome_file_or_dir, file_name)
                        fasta = next(SeqIO.parse(full_path, "fasta"))
                        DNA[chrom] = str(fasta.seq).lower()
                        print(f'DNA data for {fasta.name} is loaded')
                        del fasta
                        gc.collect()
        else:
            gen = SeqIO.parse(genome_file_or_dir, "fasta")
            n_found = 0
            for fasta in gen:
                chrom = fasta.name
                if chrom in chroms:
                    if self.chromsizes[chrom] < self.dna_len * 4:
                        print(f"Chromosome {chrom} is too short")
                        self.names.remove(chrom)
                    n_found += 1
                    DNA[chrom] = str(fasta.seq).lower()
                    print(f'DNA data for {chrom} is loaded')
                del fasta
                gc.collect()
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

    
    '''def interpolate_nan(self, array):
        x = np.arange(0, array.shape[1])
        y = np.arange(0, array.shape[0])
        array = np.ma.masked_invalid(array)
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]
        return interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')'''

    '''def get_region(self, chrom_name, start, end):
        if not self.use_adaptive_coarsegrain:
            block = self.matrix.fetch(f'{chrom_name}:{start}-{end}')
        else:
            block = self.matrix.fetch(f'{chrom_name}:{start}-{end}')
            block_raw = self.unbalanced_matrix.fetch(f'{chrom_name}:{start}-{end}')
            block = adaptive_coarsegrain(ar=block, countar=block_raw)
        if block.dtype == 'int32':
            block = block.astype('float32')
        return block'''
    
    def rotate_and_cut(self, square_maps):
        '''

        ▓▓▓░░░░░░░░░░░░
        ░░▓▓▓▓▓▓░░░░░░░        ░░░░░░░░░░░░░░░▓░
        ░░▓▓▓▓▓▓░░░░░░░   ──>  ░░░▓░░░░░░░░░▓▓▓▓
        ░░░░░░░▓▓▓▓▓▓▓▓        ░▓▓▓▓▓░░░░░▓▓▓▓▓▓
        ░░░░░░░▓▓▓▓▓▓▓▓        ▓▓▓▓▓▓▓▓░▓▓▓▓▓▓▓▓
        ░░░░░░░▓▓▓▓▓▓▓▓

        '''
        rate = 181 / square_maps.shape[1] # 181 ~ 128 * sqrt(2)
        square_maps = zoom(square_maps, (1, rate, rate, 1), order=1)
        rotated_maps = rotate(square_maps, 45, order=1, axes=(1, 2))
        w = rotated_maps.shape[1]                    
        cut_maps = rotated_maps[:, w//2-self.h:w//2, w//4:-w//4]
        return cut_maps
    

    def make_hic(self):
        '''Makes continuous Hi-C datasets for each chromosome. It is a stripe
along the main diagonal.'''
        HIC = dict.fromkeys(self.names)
        for name in self.names:
            chrom_cut = np.load(self.hic_cuts[name])
            chrom_cut[np.isnan(chrom_cut)] = 0
            crosses = np.mean(chrom_cut != 0, axis=1) < self.cross_threshold
            crosses = np.repeat(crosses[:, None, :], chrom_cut.shape[1], axis=1)
            masks = np.zeros(chrom_cut.shape, dtype=bool)
            masks[crosses] = 1
            masks[crosses.transpose(0,2,1)] = 1

            chrom_cut[masks]=np.nan
            chrom_cut = np.array([interp_nan(i) for i in chrom_cut])
            
            chrom_hic = np.stack([chrom_cut, masks], axis=-1)
            chrom_hic = self.rotate_and_cut(chrom_hic)


            '''for start in range(0, chrom_len-self.mapped_len*2, self.mapped_len):
                new_block = self.get_region(name, start, start + self.mapped_len * 2)
                new_block[np.isnan(new_block)] = 0

                # We have some genome bins with zero or too little mapped reads
                # but it doesn't mean they realy have no contacts in cell.
                # We will work with them separately
                cross = np.mean(new_block != 0, axis=0) < self.cross_threshold

                mask = np.zeros(new_block.shape)
                mask[cross] = 1
                mask[:,cross] = 1
                if self.interpolator is None:
                    #new_block[cross] = np.nan
                    #new_block[:,cross] = np.nan
                    #new_block = interp_nan(new_block)
                    new_block = np.stack([new_block, mask], axis=-1)
                    new_block = self.rotate_and_cut(new_block)[-self.h:]
                    chrom_hic.append(new_block)
                else:
                    chrom_hic.append(np.stack([new_block, mask], axis=-1))
            
            if self.interpolator is not None:
                chrom_hic = np.array([zoom(i, (128/i.shape[0], 128/i.shape[0], 1), order=1) for i in chrom_hic])
                chrom_hic[..., 0] *= 5
                chrom_hic[..., :1] = self.interpolator.predict(chrom_hic, verbose=0)
                chrom_hic = [self.rotate_and_cut(i)[-self.h:] for i in chrom_hic]

            '''
            chrom_hic = np.hstack(chrom_hic)
            to_transform = chrom_hic[...,0]
            to_transform += self.psi
            to_transform = np.log(to_transform)

            valid = chrom_hic[...,1] < 0.1
            valid_by_diag = [to_transform[i][valid[i]] for i in range(self.h)]
            means = [i.mean() for i in valid_by_diag]
            stds = [i.std() for i in valid_by_diag]
            to_transform -= np.array(means)[:, None]
            to_transform /= np.array(stds)[:, None]*2
 
            #to_transform[~valid] = 0
            
            if self.sigma:
                to_transform = gaussian_filter(to_transform, sigma=self.sigma)
            
            if self.remove_first_diag:
                to_transform[-self.remove_first_diag:] = 0

            chrom_hic[..., 0] = to_transform

            del chrom_cut, masks, crosses, to_transform, valid, valid_by_diag
            gc.collect()
            #chrom_hic[...,1] = chrom_hic[...,1] > 0.5 # make masks discrete after interpolations
            HIC[name] = chrom_hic
            print(f'Hi-C data for {name} is loaded')
        
        return HIC
    
    def save_hic(self, folder):
        for name,array in self.HIC.items():
            np.save(os.path.join(folder, name), array)
        print('Saved at ' + folder)

    def make_dataset(self):
        '''Makes arrays of corresponded positions in the DNA and in stripe of
the Hi-C map. Doesn't make any operations with data itself.'''
        hic_index_list = []
        dna_index_list = []
        for n, name in enumerate(self.names):
            chrom_len = self.chromsizes[name]
            start_pos = 0#self.offset ###
            end_pos = chrom_len - self.offset - self.mapped_len * 3 # with a margin
            current_chrom_hic_index_list = []
            current_chrom_dna_index_list = []
            for start in range(start_pos, end_pos, self.mapped_len):
                map_pos = int(start / self.mapped_len * self.map_size)
                current_chrom_hic_index_list.append(np.array([n, map_pos, map_pos + self.map_size]))
                current_chrom_dna_index_list.append(np.array([n, start - self.offset + self.mapped_len // 2,
                                          start + self.mapped_len + self.offset + self.mapped_len // 2]))
            hic_index_list.append(np.array(current_chrom_hic_index_list))
            dna_index_list.append(np.array(current_chrom_dna_index_list))
        return dna_index_list, hic_index_list

    def train_val_split(self, dna_index_list, hic_index_list):
        n_chroms = len(dna_index_list)
        dna_index_array = np.concatenate(dna_index_list)
        hic_index_array = np.concatenate(hic_index_list)
        assert len(dna_index_array) == len(hic_index_array)
        x_train, x_val, y_train, y_val = [], [], [], []
        val_sample_contig_list = []
        if self.val_split == 'default': # val sample contains contigous part of DNA with length ~ 64 * fragment_size (starting from the first chrom)
            chroms_used_for_val_sample = list(set([self.names[i] for i in dna_index_array[:64, 0]]))
            for name in chroms_used_for_val_sample[:-1]:
                size = self.chromsizes[name]
                val_sample_contig_list.append((name, size))
            val_sample_contig_list.append((chroms_used_for_val_sample[-1], dna_index_array[63, 2]))
        else:
            chroms_used_for_val_sample = []
            chroms = self.val_split.split(',')
            for chrom in chroms:
                if ':' in chrom:
                    name, end = chrom.split(':')
                    end = int(end)
                else:
                    name = chrom
                    end = self.chromsizes[name]
                chroms_used_for_val_sample.append(name)
                val_sample_contig_list.append((name, end))
        max_chrom_n = max([len(i) for i in dna_index_list])
        visualization_array = np.full((n_chroms, max_chrom_n), np.nan)
        for contig in val_sample_contig_list:
            name, end = contig
            chrom_index = self.names.index(name)
            end_after_shifts = end + self.dna_len
            for j, pair in enumerate(zip(dna_index_list[chrom_index], hic_index_list[chrom_index])):
                x, y = pair
                if x[2] <= end:
                    x_val.append(x)
                    y_val.append(y)
                    visualization_array[chrom_index, j] = 1
                elif x[1] > end_after_shifts:
                    x_train.append(x)
                    y_train.append(y)
                    visualization_array[chrom_index, j] = -1
        
        train_only_chrom_indices = [i for i in range(len(self.names)) if self.names[i] not in chroms_used_for_val_sample]
        for chrom_index in train_only_chrom_indices:
            for j, pair in enumerate(zip(dna_index_list[chrom_index], hic_index_list[chrom_index])):
                x, y = pair
                x_train.append(x)
                y_train.append(y)
                visualization_array[chrom_index, j] = -1
        #clear_output()
        print('Validation sample is red, train sample is blue:')
        plot_samples(visualization_array, self.names)
        x_train, x_val, y_train, y_val = map(np.array, [x_train, x_val, y_train, y_val])
        #self.val_sample_contig_list = val_sample_contig_list
        return x_train, x_val, y_train, y_val       


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
            samples = x_train, x_val, y_train, y_val
        else:
            x_train, x_val, y_train, y_val = initial_samples

        train_sample_contig_list = []
        train_sample_chrom_indices = list(set(x_train[:,0]))
        for chrom_index in train_sample_chrom_indices:
            sample_subset_from_current_chrom = x_train[x_train[:,0]==chrom_index]
            start = sample_subset_from_current_chrom[:,1:].min()
            end = sample_subset_from_current_chrom[:,1:].max()
            train_sample_contig_list.append((self.names[chrom_index], start, end))

        val_sample_contig_list = []
        val_sample_chrom_indices = list(set(x_val[:,0]))
        for chrom_index in val_sample_chrom_indices:
            sample_subset_from_current_chrom = x_val[x_val[:,0]==chrom_index]
            start = sample_subset_from_current_chrom[:,1:].min()
            end = sample_subset_from_current_chrom[:,1:].max()
            val_sample_contig_list.append((self.names[chrom_index], start, end))


        val_sample_contig_list_string = ', '.join([f'{name}:{start:,}-{end:,}' for name,start,end in val_sample_contig_list])
        train_sample_contig_list_string = ', '.join([f'{name}:{start:,}-{end:,}' for name,start,end in train_sample_contig_list])

        print('Validation sample includes fragments from the following contigs: ' + val_sample_contig_list_string)
        print('Train sample includes fragments from the following contigs: ' + train_sample_contig_list_string)

        return x_train, x_val, y_train, y_val
        
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
            if self.HIC[name][:, start : end, 1].mean() > self.nan_threshold:
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
            if self.HIC[name][:, start : end, 1].mean() > self.nan_threshold:
                bad += 1
            else:
                good += 1
                good_val.append(i)
        x_val, y_val = x_val[good_val], y_val[good_val]
                    
        print(f'{bad/(bad+good)*100:.2f}% of maps were excluded by NaN threshold')
        print(f'Train sample has {len(good_train)} pairs now (of \
{expected_train_size} possible)')
        print(f'Validation sample has {len(good_val)} pairs now \
(of {expected_val_size} possible)')
        return x_train, x_val, y_train, y_val

    '''def scale_hic(self):
            
        # individual maps normalization
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
            y = self.y_train[:]
            self.sd = y.std()
            self.mean = y.mean()'''

    '''def calculate_insulation_score(self, window=10):
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
        return y_train_is[...,0], y_val_is[...,0]'''
    
    def make_bed_of_samples(self, dir=None):
        if dir is None:
            dir = './'
        train_name = os.path.join(dir, 'train.bed')
        val_name = os.path.join(dir, 'val.bed')
        df = pd.DataFrame({
            'chrom': [self.names[i] for i in self._x_train[:, 0]],
            'dna_start': self._x_train[:, 1],
            'dna_end': self._x_train[:, 2],
            'hic_start': self._y_train[:, 1],
            'hic_end': self._y_train[:, 2],
            'sample': ['train']*len(self._x_train)
        })
        df.to_csv(train_name, sep='\t')

    def load_bed(self, bed_file, bins, name):
        bed = pd.read_csv(bed_file)

        x_train = self._x_train.copy()
        x_train[:, 0] = [self.names[i] for i in x_train[:, 0]]
        x_train = pd.DataFrame(self._x_train)

        x_val = self._x_val.copy()
        x_val[:, 0] = [self.names[i] for i in x_val[:, 0]]
        x_val = pd.DataFrame(self._x_val)
        print(f'Data loaded into .{name}_train and .{name}_val attributes')

    def coord(self, x_chrom_start_end=None, y_chrom_start_end=None, x_pos=None, y_pos=None):
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
            plt.plot(chrom[...,1].mean(axis=0))
            plt.show()
    
    def plot_annotated_map(self,
                        sample,
                        number,
                        ax=None,
                        y=None,
                        name=None,
                        axis='both',
                        y_label_shift=False,
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
                       cmap=self.cmap,
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
                                 start, axis, h=self.h,
                                 w=self.map_size, y_label_shift=y_label_shift)
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
            txt = ax.text(1,5,f'{name}', fontsize=14, color='white')
            txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='black')])
            
        if show:
            plt.show()


class DNALoader():

    """Saves memory and helps to make array of one-hot-encoded DNA batch
    only when it is needed. Input data of model is stored in DNA string 
    and array of indices"""

    def __init__(self, data, input_data):
        self.data = data
        self.names = data.names
        self.dna_len = data.dna_len
        self.input_data = input_data
        self.len = len(self.input_data)
        self.alphabet = {'a' : 0, 'c' : 1, 'g' : 2, 't' : 3, 'n' : 4}

    def one_hot(self, seq):
        return to_categorical([self.alphabet[i] for i in list(seq)])[:,:4]
    
    def gen(self):
        return DataGenerator(self, batch_size=1)
        
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
            seq = self.data.DNA[self.names[chrom]][start : end]
            batch.append(self.one_hot(seq))
        return np.array(batch)

    def __len__(self):
        return self.len


class HiCLoader():
    '''Works similar to DNALoader, returns fragments of the map stripe by positions'''
    def __init__(self, data, input_data, normalize='minmax', mask=False):
        self.data = data
        self.normalize = normalize
        self.mask = mask
        self.names = data.names
        self.input_data = input_data
        self.len = len(self.input_data)
    
    def show(self, i):
        plot_map(self[i])
    
    def show_orig(self, n):
        a = self.data.coord(y_chrom_start_end=self.input_data[n])
        hic = self.data.get_region(*a)
        orig_map = np.log(hic + self.data.psi)
        plt.imshow(orig_map, cmap='Reds')
        c = len(orig_map)
        a = c//4
        plt.plot([a,a+a//2, c-a//2, c-a, a],[a, a//2,c-a-a//2, c-a, a], c='k')
        plt.axis('off')
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
            mask = self.data.HIC[self.names[chrom]][:, start : end]
            hic_map = self.data.HIC[self.names[chrom]][:, start : end]
            batch.append(hic_map)
        batch = np.array(batch)
        batch, mask_batch = batch[...,:1], batch[...,1:]
        if not self.mask:
            '''if self.normalize == 'standart':
                mu = self.data.mean
                sd = self.data.sd
                if sd is not None:
                    batch = (batch - mu) / sd
                    for i in range(len(batch)):
                        batch[i][mask_batch[i] > 0.2] = batch[i][mask_batch[i] < 0.2].mean()
                    noise = np.random.normal(0,1,batch.shape)
                    batch[mask_batch > 0.2] += noise[mask_batch > 0.2]
                else: 
                    batch = batch'''
            return batch
        else:
            return mask_batch

    def __len__(self):
        return self.len
