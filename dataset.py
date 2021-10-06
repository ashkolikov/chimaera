import os
import gc

import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate, zoom, gaussian_filter


from cooler import Cooler
from cooltools.lib.numutils import interp_nan, observed_over_expected, adaptive_coarsegrain
from Bio import SeqIO

from .plot import *

class DataMaster(object):
    """docstring for DataMaster"""
    def __init__(self,
                 hic_file, 
                 genome_file_or_dir, 
                 fragment_length,
                 sigma,
                 chroms_to_exclude = [],
                 chroms_to_include = [],
                 scale = (0, 1),
                 normalize = 'global',
                 min_max = (None, None),
                 map_size = 64,
                 nan_threshold = 0.2,
                 rev_comp = False,
                 stochastic_sampling = False,
                 shift_repeats = 1,
                 expand_dna = True,
                 dna_encoding = 'one-hot',
                 val_split = ('first', 32),
                 cut_chromosome_ends = 0,
                 sample_seed = 0):

        if stochastic_sampling and (shift_repeats > 1):
            raise ValueError("Stochastic sampling and shift_repeats can't be used together")
        if (val_split[0] == 'random') and (stochastic_sampling or (shift_repeats > 1)):
            raise ValueError("Random sampling is incorrect in not fixed dataset")
        self.mapped_len = fragment_length
        self.map_size = map_size
        self.nan_threshold = nan_threshold
        self.stochastic_sampling = stochastic_sampling
        self.shift_repeats = shift_repeats
        self.expand_dna = expand_dna
        self.cut_chromosome_ends = cut_chromosome_ends
        self.val_split = val_split
        self.rev_comp = rev_comp
        self.dna_encoding = dna_encoding
        self.sample_seed = sample_seed
        self.chroms_to_exclude = chroms_to_exclude
        self.chroms_to_include = chroms_to_include
        self.min_max = min_max

        self.scale = scale
        self.normalize = normalize
        self.sigma = sigma

        self.cooler = Cooler(hic_file)
        self.make_dna(genome_file_or_dir, chroms_to_exclude, chroms_to_include)
        self._x_train, self._x_val, self._y_train, self._y_val = self.split_data()
        self.scale_y()
        self.x_train, self.y_train  = DNALoader(self, self._x_train), HiCLoader(self, self._y_train)
        self.x_val, self.y_val  = DNALoader(self, self._x_val), HiCLoader(self, self._y_val)
        self.make_sample()
        #self.x_train, self.x_val, self.y_train, self.y_val = extra_data
        
        self.params = {'hic_file': hic_file, 
                       'genome_file_or_dir': genome_file_or_dir, 
                       'fragment_length': fragment_length,
                       'chroms_to_exclude': self.chroms_to_exclude,
                       'chroms_to_include': self.chroms_to_include,
                       'sigma': sigma,
                        'scale': scale,
                        'normalize': normalize,
                        'map_size': map_size,
                        'nan_threshold': nan_threshold,
                        'stochastic_sampling': stochastic_sampling,
                        'shift_repeats': shift_repeats,
                        'rev_comp': rev_comp,
                        'expand_dna': expand_dna,
                        'val_split': val_split,
                        'cut_chromosome_ends': cut_chromosome_ends,
                        'sample_seed': sample_seed}

    def make_dna(self, genome_file_or_dir, exclude, include):
        self.DNA = dict()
        self.names = []
        if os.path.isdir(genome_file_or_dir):
            files_in_dir = os.listdir(genome_file_or_dir)
            if include != []:
                availible_files = [i for i in files_in_dir if (i.split('.')[0] in include and i.split('.')[0] in self.cooler.chromnames)]
            elif exclude != []:
                availible_files = [i for i in files_in_dir if (i.split('.')[0] not in exclude and i.split('.')[0] in self.cooler.chromnames)]
            else:
                raise ValueError('You should select not all chromosomes for test&val samples')
            
            for file in availible_files:
                fasta = next(SeqIO.parse(os.path.join(genome_file_or_dir, file), "fasta"))
                self.DNA[fasta.name] = str(fasta.seq).lower()
                self.names.append(fasta.name)
                print(f'DNA data for {fasta.name} is loaded')
                del fasta
        else:
            gen = SeqIO.parse(genome_file_or_dir, "fasta")
            for fasta in gen:
                name = fasta.name
                if include != []:
                    if name not in include or name not in self.cooler.chromnames:
                        del fasta
                        continue 
                elif exclude != []:
                    if name in exclude or name not in self.cooler.chromnames:
                        del fasta
                        continue
                else:
                    raise ValueError('You should select not all chromosomes for test&val samples')        
                self.DNA[name] = str(fasta.seq).lower()
                self.names.append(name)
                print(f'DNA data for {fasta.name} is loaded')
                del fasta
        gc.collect()
        print()

    def transform_hic(self, hic_matrix_raw, hic_matrix):
        transformed_arr = adaptive_coarsegrain(hic_matrix_raw, hic_matrix)
        nan_mask = np.isnan(transformed_arr)
        transformed_arr, _,_,_ = observed_over_expected(transformed_arr, mask = ~nan_mask)
        transformed_arr = np.log(transformed_arr)
        transformed_arr = interp_nan(transformed_arr)
        return transformed_arr, np.mean(nan_mask)

    def get_region(self, name, start, end):
        mtx_raw = self.balanced.fetch(f'{name}:{start}-{end}')
        mtx_balanced = self.not_balanced.fetch(f'{name}:{start}-{end}')
        #print(mtx_raw.shape)
        return self.transform_hic(mtx_raw, mtx_balanced)

    def split_data(self):
        resolution = self.cooler.binsize
        brim_len = 0 if not self.expand_dna else self.mapped_len // 2
        self.dna_len = self.mapped_len + brim_len * 2
        initial_map_size = self.mapped_len // resolution
        zoom_rate = self.map_size / initial_map_size
        self.zoom_rate = zoom_rate
        fragment_length_str = f'{brim_len} + {self.mapped_len} + {brim_len}' if brim_len else str(self.mapped_len)
        if self.zoom_rate != 1:
            print(f'Maps are zoomed {zoom_rate} times')
        print(f'For {self.map_size}x{self.map_size} map used {fragment_length_str} nucleotide fragments')

        use_big_map = self.stochastic_sampling or (self.shift_repeats > 1)
        self.slice_mapped_len = self.mapped_len if not use_big_map else self.mapped_len * 2
        real_map_size = self.map_size if not use_big_map else self.map_size * 2
        if use_big_map:
            print(f'Initial dataset contains {real_map_size}x{real_map_size} maps, overlooping in {self.map_size} pixels')
            print(f'{self.map_size}x{self.map_size} maps for training will be sampled from them, maps for testing are their top left fragments')

        self.balanced, self.not_balanced = self.cooler.matrix(balance=True), self.cooler.matrix(balance=False)
        hic_list = []
        dna_list = []
        for n_, name in enumerate(self.names):
            chrom_len = self.cooler.chromsizes[name]
            assert chrom_len == len(self.DNA[name])
            start_pos = max(self.cut_chromosome_ends, brim_len)
            end_pos = chrom_len - start_pos - self.slice_mapped_len
            for start in range(start_pos, end_pos, self.mapped_len):
                new_block, nan_percentage = self.get_region(name, start, start + self.slice_mapped_len)
                #new_block2, nan_mask2 = get_region(balanced, not_balanced, name, start, start + fragment_size)
                if nan_percentage > self.nan_threshold:
                    continue
                #new_block = np.stack((new_block, new_block2), axis=-1)
                new_block = np.nan_to_num(new_block, nan = -1.0, posinf = 1.0, neginf = -1.0)
                assert np.any(~np.isinf(new_block))
                if zoom_rate != 1:
                    new_block = zoom(new_block, (zoom_rate, zoom_rate))[:real_map_size, :real_map_size, None]
                else:
                    new_block = new_block[:real_map_size, :real_map_size, None]
                if self.sigma:
                    new_block = gaussian_filter(new_block, sigma = self.sigma)

                hic_list.append(new_block)
                dna_list.append(np.array([n_, start - brim_len, start + self.slice_mapped_len + brim_len]))
                del new_block
            print(f'Hi-C data for {name} is loaded')

        data  = self.train_val_split(np.array(dna_list), np.array(hic_list))
        if self.shift_repeats > 1:
            _x_train, _x_val, _y_train, _y_val = data
            x_train, x_val, y_train, y_val = [], [], [], []
            shift = self.mapped_len // self.shift_repeats
            y_shift = int(shift / self.mapped_len * self.map_size)
            for i in range(self.shift_repeats):
                x_shift_array = np.concatenate((np.zeros((1, 1), dtype=int), np.full((2, 1), shift * i))).T
                y_shift_start, y_shift_end = y_shift * i, y_shift * i + self.map_size
                y_train.append(_y_train[:, y_shift_start:y_shift_end, y_shift_start:y_shift_end])
                x_train.append(_x_train + x_shift_array)
                y_val.append(_y_val[:, y_shift_start:y_shift_end, y_shift_start:y_shift_end])
                x_val.append(_x_val + x_shift_array)
            del data
            data = np.concatenate(x_train), np.concatenate(x_val), np.concatenate(y_train), np.concatenate(y_val)
            del _x_train, _x_val, _y_train, _y_val, x_train, x_val, y_train, y_val

        del dna_list, hic_list
        gc.collect()
        
        return data

    def train_val_split(self, dna_list, hic_list):
        assert len(dna_list) == len(hic_list)
        if self.val_split == 'test':
            return dna_list, dna_list, hic_list, hic_list
        method, val_split = self.val_split
        if method.startswith('chr'):
            if val_split < 1:
                val_split = int(len(hic_list) * val_split)
            val_split += 2 # because then two elenents will be thrown out

            chrom, pos = method.split()
            chrom_inds = np.where(dna_list[:, 0] == self.names.index(chrom))
            ind_in_chrom = np.where(np.all(((dna_list[chrom_inds][:, 1] < pos), (dna_list[chrom_inds][:, 2] >= pos)), axis = 0))
            mid = chrom_inds[0][0] + ind_in_chrom[0][0]
            x_val = [dna_list.pop(mid - val_split // 2) for i in range(val_split)]
            y_val = [hic_list.pop(mid - val_split // 2) for i in range(val_split)]
            # cut edge elements to avoid intersection with train dataset
            data = dna_list, x_val[1:-1], hic_list, y_val[1:-1]
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
                val_split += 2 # because then two elenents will be thrown out
                if method == 'first':
                    data = dna_list[val_split:], dna_list[:val_split], hic_list[val_split:], hic_list[:val_split]
                elif method == 'last':
                    data = dna_list[:-val_split], dna_list[-val_split:], hic_list[:-val_split], hic_list[-val_split:]
                del hic_list
                # cut edge elements to avoid intersection with train dataset or cut a chromosome end
                return data[0], data[1][1:-1], data[2], data[3][1:-1]


    def scale_y(self):
        if self.scale is None:
            return
        else:
            if self.normalize == 'global':
                if self.min_max == (None, None):
                    total_min = min(self._y_train.min(), self._y_val.min())
                else:
                    total_min = self.min_max[0]
                self._y_train -= total_min
                self._y_val -= total_min

                
                if self.min_max == (None, None):
                    total_max = max(self._y_train.max(), self._y_val.max())
                else:
                    total_max = self.min_max[1]
                
                self._y_train /= total_max
                self._y_val /= total_max

                self.min_max = (total_min, total_max)

            elif self.normalize == 'each':
                self._y_train -= np.min(self._y_train, axis=(1,2,3)).reshape(-1,1,1,1)
                self._y_train /= np.max(self._y_train, axis=(1,2,3)).reshape(-1,1,1,1)
                self._y_val -= np.min(self._y_val, axis=(1,2,3)).reshape(-1,1,1,1)
                self._y_val /= np.max(self._y_val, axis=(1,2,3)).reshape(-1,1,1,1)
            else:
                raise ValueError("normalize arguement should be 'global' or 'each'")

            if self.scale != (0, 1):
                self._y_train *= self.scale[1] - self.scale[0]
                self._y_train += self.scale[0]
                self._y_val *= self.scale[1] - self.scale[0]
                self._y_val += self.scale[0]
        
        gc.collect()

    def make_sample(self, seed=None):
        if len(self.y_val)<9:
            return
        if seed is None:
            np.random.seed(self.sample_seed)
        elif seed == 'random':
            pass
        else:
            np.random.seed(seed)
        train_inds = np.random.choice(len(self.y_train), 9, replace = False)
        val_inds = np.random.choice(len(self.y_val), 9, replace = False)
        self.x_train_sample, self.y_train_sample = self.x_train[train_inds], self.y_train[train_inds]
        self.x_val_sample, self.y_val_sample = self.x_val[val_inds], self.y_val[val_inds]




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
        # if isinstance(self.input_data, np.ndarray):

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
    def __init__(self, data, input_data):
        self.map_size = data.map_size
        self.input_data = input_data
        self.len = len(self.input_data)
    
    def show(self, i):
        plot_map(self[i])

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
            ind = list(range(start, stop))
        elif isinstance(i, int):
            ind = [i]
        else:
            ind = i
        batch = self.input_data[ind]
        batch = batch[:, shift : shift + self.map_size, shift : shift + self.map_size]
        return np.array(batch)

    def __len__(self):
        return self.len



class DataGenerator(Sequence):
    def __init__(self, data, train, batch_size = 4, shuffle = True, encoder = None):
        if train:
            self.X = data.x_train
            if data.stochastic_sampling:
                self.y = data.y_train
            else:
                self.y = data.y_latent_train
        else:
            self.X = data.x_val
            if data.stochastic_sampling or data.val_split == 'test':
                self.y = data.y_val                
            else:
                self.y = data.y_latent_val

        if data.stochastic_sampling and not encoder:
            raise ValueError('For training on dataset using stochastic sampling pass your encoder model to the encoder arguement')
        self.data = data
        self.batch_size = batch_size
        self.train = train
        self.shuffle = shuffle
        self.encoder = encoder
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        if not self.data.stochastic_sampling:
            X = self.X[indexes]
            y = self.y[indexes]
        else:
            if self.train:
                shift = np.random.choice(self.data.mapped_len)
                y_shift = int(shift / self.data.mapped_len * self.data.map_size)
                X = self.X[indexes, shift]
                y = self.y[indexes, y_shift]
                if self.data.rev_comp:
                    X, y = self.stochastic_rev_comp(X, y)
                y = self.encoder.predict(y)
            else:
                X = self.X[indexes]
                y = self.y[indexes]
                y = self.encoder.predict(y)
        return X, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def stochastic_rev_comp(self, X, y):
        ind = np.random.random(len(X)) > 0.5
        if self.data.dna_encoding == 'one_hot':
            X[ind] = np.flip(X[ind], axis=(1, 2))
        else:
            X[ind] = 3 - X[ind]
            X[X==-1] = 4
        return X, y


class HiCDataGenerator(Sequence):
    def __init__(self, data, rotate = True, shuffle = True):
        self.X = data.y_train
        self.batch_size = 64
        self.rotate = rotate
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X = self.X[indexes]
        if self.rotate:
            X = self.stochastic_rotate(X)
        return X, X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def stochastic_rotate(self, a):
        ind = np.random.random(len(a)) > 0.5
        a[ind] = np.flip(a[ind], axis=(1, 2))
        return a
