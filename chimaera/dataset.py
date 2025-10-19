# -*- coding: utf-8 -*-
import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom, gaussian_filter
from collections import defaultdict
from Bio import SeqIO
from torch import masked_select
import json
import warnings


from . import data_utils
from . import plot_utils


def dataset_from_params(
        params_path,
        hic_data_path,
        genome,
        test_chroms_only=False,
        show=True,
        **kwargs
    ):
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            data_params = json.load(f)
    else:
        raise FileNotFoundError('Params file not found')
    data_params.pop('batch_size')
    if test_chroms_only:
        chroms = [i.split(':')[0] for i in data_params['test_regions'].split(';')]
        data_params['chroms_to_include'] = chroms
    data_params.update(kwargs)
    data = Dataset(hic_data_path=hic_data_path,
                    genome=genome,
                    show=show,
                    **data_params)
    return data



class Dataset():
    '''Data loading, processing and storing'''
    def __init__(self,
                 hic_data_path,
                 genome=None,
                 chroms_to_exclude = [],
                 chroms_to_include = [],
                 test_regions = 'default',
                 val_fraction = 0.33,
                 h = 32,
                 sigma = 0,
                 nan_threshold = 0.2,
                 cross_threshold = 0.75,
                 enlarge_crosses = 0,
                 skip_quadrants = False,
                 remove_first_diag = 2,
                 offset = 'default',
                 psi = .001,
                 interpolation_order=0,
                 scale_by=1,
                 fragment_length=None,
                 organism_name = '',
                 show=True,
                 experiment_names=None,
                 square_maps=False,
                 plot_maps=False):

        self.hic_data_path = hic_data_path
        self.nan_threshold = nan_threshold
        self.remove_first_diag = remove_first_diag
        self.chroms_to_exclude = chroms_to_exclude
        self.val_fraction = val_fraction
        self.chroms_to_include = chroms_to_include
        self.cross_threshold = cross_threshold
        self.enlarge_crosses = enlarge_crosses
        self.skip_quadrants = skip_quadrants
        self.h = h
        self.psi = psi
        self.sigma = sigma
        self.scale_by = scale_by
        self.square_maps = square_maps
        self.interpolation_order = interpolation_order
        self.plot_samples = plot_samples
        len_and_res = hic_data_path.split('len=')[1]
        self.mapped_len = int(len_and_res.split('_')[0])
        if fragment_length is not None:
            if fragment_length != self.mapped_len:
                raise ValueError(f"Fragment length should be {fragment_length}")

        # not flexible now
        self.map_size = 128
        self.cmap = "RdBu_r"

        self.offset = self.mapped_len // 2 if offset=='default' else offset
        if self.offset > self.mapped_len // 2:
            raise ValueError(f"offset shouldn't be more than fragment_lenght/2 \
({self.mapped_len // 2})")
        self.dna_len = self.mapped_len + self.offset * 2
        self.resolution = self.mapped_len // self.map_size
        self.binsize = self.resolution # just a synonym
        h_binsize = self.mapped_len // (self.map_size // 2)
        # nearest contacts on maps:
        self.min_dist = self.remove_first_diag * h_binsize
        # farest contacts on maps:
        self.max_dist = (self.h + self.remove_first_diag) * h_binsize

        # find preloaded hi-c
        self.hic_files, available_chroms_with_sizes = self._find_available_hic(
            hic_data_path
        )
        # select chromosomes
        self.chromsizes = self._choose_chroms(
            available_chroms_with_sizes,
            chroms_to_include,
            chroms_to_exclude
        )
        self.chromnames = list(self.chromsizes.keys())

        self.test_regions = self._parse_test_split(test_regions)

        self.genome_size =  sum(self.chromsizes.values())
        self.organism = organism_name
        self.experiment_names = experiment_names

        # load DNA as a dict of stings for each chromosome
        if genome is not None:
            self.DNA = self._make_dna(genome, self.chromnames)
            self._check_chromsizes()

        # load Hi-C as a dict of contigous chromosome slices along the main diag
        if self.square_maps:
            self.HIC, self.MASKS, self.square_hic, self.square_masks = self._make_hic()
        else:
            self.HIC, self.MASKS = self._make_hic()
        # last dim in Hi-C slices is for multiple experiments
        # (different cell lines, developmental stages, etc)
        self.out_channels = self.HIC[self.chromnames[0]].shape[-1]
        if self.experiment_names:
            if len(self.experiment_names) != self.out_channels:
                raise ValueError(f'Loaded data has maps from {self.out_channels}\
 Hi-C experiments but you listed names for {len(self.experiment_names)}')
        else:
            if self.out_channels > 1:
                self.experiment_names = [f'Experim. {i}' for i in range(self.out_channels)]
            else:
                self.experiment_names = ['']

        # select regions for each sample
        if genome is not None:
            self.inintial_train_sample, self.val_sample, self.test_sample = self._make_samples(
                val_fraction=val_fraction
            )
            self.train_sample = self.inintial_train_sample.copy() # will be modified while training

            # make loaders generating slices of selected regions
            self.x_train = data_utils.DNALoader(self, self.inintial_train_sample, check='train')
            self.y_train = data_utils.HiCLoader(self, self.inintial_train_sample, check='train')
            self.x_val = data_utils.DNALoader(self, self.val_sample, check='not-train')
            self.y_val = data_utils.HiCLoader(self, self.val_sample, check='not-train')
            self.x_test = data_utils.DNALoader(self, self.test_sample, check='not-train')
            self.y_test = data_utils.HiCLoader(self, self.test_sample, check='not-train')

            if show:
                print('Test and val regions:')
                self.show(exclude_imputed=True)

    def shift_train_sample(self, len_fraction):
        shift = int(len_fraction * self.mapped_len)
        self.train_sample = [(chrom, start+shift, end+shift) for chrom, start, end in self.train_sample]

    def reset_train_sample(self):
        self.train_sample = self.inintial_train_sample.copy()

    def _find_available_hic(self, hic_data_path):
        '''Returns names and sizes of chromosomes with availible hi-c maps'''
        if os.path.isdir(hic_data_path):
            all_files = os.listdir(hic_data_path)
            hic_files = [i for i in all_files if i.endswith('npy')]
            hic_files_dict = dict()
            if len(hic_files) == 0:
                raise ValueError(f'No .npy files found in {hic_data_path}')
            chromsizes = dict()
            for hic_file in hic_files:
                filename = os.path.join(hic_data_path, hic_file)
                splited_name = hic_file.split('.')
                chrom_name_and_size = '.'.join(splited_name[:-1])
                chrom_name_and_size = chrom_name_and_size.split('_')
                if len(chrom_name_and_size) == 2:
                    chromname, chromsize = chrom_name_and_size
                elif len(chrom_name_and_size) > 2:
                    chromname = '_'.join(chrom_name_and_size[:-1])
                    chromsize = chrom_name_and_size[-1]
                else:
                    raise ValueError('filename for chromosome is incorrect')

                if chromname in chromsizes.keys():
                    raise ValueError(f'Chrom {chromname} has more than 1 files')
                chromsizes[chromname] = int(chromsize)
                hic_files_dict[chromname] = filename
            return hic_files_dict, chromsizes
        else:
            raise ValueError('hic_data_path should be a directory with \
preprocessed .npy files')

    def _check_chromsizes(self):
        '''Checks if chrom sizes in genome and hi-c files are the same'''
        dna_lens = {i:len(j) for i,j in self.DNA.items()}
        for name in self.chromnames:
            if self.chromsizes[name] != dna_lens[name]:
                raise ValueError('Chrom sizes in fasta and cool are different')

    def _choose_chroms(self, available, include, exclude):
        '''Makes list of used chromosomes'''
        if include:
            chroms = {i:j for i,j in available.items() if i in include}
        elif exclude:
            chroms = {i:j for i,j in available.items() if i not in exclude}
        else:
            chroms = available
        return chroms

    def _make_dna(self, genome, chroms):
        '''Makes a dict of chromosomes' sequences'''
        chroms = chroms.copy()
        DNA = dict()
        short = []
        correct_chrom_order = []
        if os.path.isdir(genome): # chroms are separete files
            available_files = os.listdir(genome)
            for file_name in available_files:
                chrom = file_name.split('.')[0]
                if chrom in chroms:
                    if self.chromsizes[chrom] < self.dna_len * 4:
                        print(f"Chromosome {chrom} is too short")
                        short.append(chrom)
                    else:
                        full_path = os.path.join(genome, file_name)
                        fasta = next(SeqIO.parse(full_path, "fasta"))
                        DNA[chrom] = str(fasta.seq).lower()
                        print(f'DNA data for {chrom} is loaded')
                        correct_chrom_order.append(chrom)
                        del fasta
                        #gc.collect()
        else: # all chroms in one file
            gen = SeqIO.parse(genome, "fasta")
            n_found = 0
            for fasta in gen:
                chrom = fasta.name
                if chrom in chroms :
                    if self.chromsizes[chrom] < self.dna_len * 4:
                        print(f"Chromosome {chrom} is too short")
                        short.append(chrom)
                    else:
                        DNA[chrom] = str(fasta.seq).lower()
                        print(f'DNA data for {chrom} is loaded')
                        correct_chrom_order.append(chrom)
                    n_found += 1
                del fasta
                #gc.collect()
                if len(chroms) == n_found:
                    break
        for chrom in chroms:
            if chrom in short:
                self.chromnames.remove(chrom)
            elif not (chrom in DNA.keys()):
                self.chromnames.remove(chrom)
                print(f'No DNA found for {chrom}')
        # set the correct order of chromosomes as in the genome file:
        self.chromnames.sort(key=lambda x:correct_chrom_order.index(x))
        self.chromsizes={i:self.chromsizes[i] for i in self.chromnames}


        gc.collect()
        print()
        if len(DNA) == 0:
            if self.chroms_to_include:
                raise ValueError("No DNA found. Probably you passed wrong \
chrom names or genome file doesn't contain these chromosomes. Check chrom names\
 in .fasta and .cool files - may be they differ")
            else:
                raise ValueError("No DNA found. Seems fasta file doesn't \
contain chroms listed in the Hi-C map files. Check chrom names in .fasta and \
.cool files - may be they differ")
        return DNA


    def _rotate_and_cut(self, square_maps):
        '''

        ▓▓▓░░░░░░░░░░░░
        ░░▓▓▓▓▓▓░░░░░░░        ░░░░░░░░░░░░░░░▓░
        ░░▓▓▓▓▓▓░░░░░░░   ──>  ░░░▓░░░░░░░░░▓▓▓▓
        ░░░░░░░▓▓▓▓▓▓▓▓        ░▓▓▓▓▓░░░░░▓▓▓▓▓▓
        ░░░░░░░▓▓▓▓▓▓▓▓        ▓▓▓▓▓▓▓▓░▓▓▓▓▓▓▓▓
        ░░░░░░░▓▓▓▓▓▓▓▓

        '''
        rate = 181 / square_maps.shape[1] # 181 ~ 128 * sqrt(2)
        square_maps = zoom(
            square_maps,
            (1, rate, rate, 1),
            order=self.interpolation_order,
            grid_mode=True,
            mode='grid-constant'
        )
        rotated_maps = rotate(square_maps, 45, order=0, axes=(1, 2))
        w = rotated_maps.shape[1]
        d = self.remove_first_diag
        cut_maps = rotated_maps[:, w//2-self.h-d:w//2-d, w//4:-w//4]
        return cut_maps


    def _update_masks(self, masks, bins):
        bins = np.repeat(
                bins[:, None, :],
                masks.shape[1],
                axis=1
            )
        # make crosses
        masks[bins] = 1
        masks[bins.transpose(0,2,1,3)] = 1
        return masks


    def _make_hic(self):
        '''Makes continuous Hi-C datasets for each chromosome. It is a stripe
along the main diagonal.'''
        HIC = dict.fromkeys(self.chromnames)
        MASKS = dict.fromkeys(self.chromnames)
        if self.square_maps:
            square_hic = dict.fromkeys(self.chromnames)
            square_masks = dict.fromkeys(self.chromnames)
        for name in self.chromnames:
            chrom_hic = np.load(self.hic_files[name])

            # make masks for not mapped genomic bins (crosses of NaNs on maps)
            # crosses with too little contacts (< cross_threshold) are treated same
            masks = np.zeros(chrom_hic.shape)
            chrom_hic[np.isnan(chrom_hic)] = 0
            empty_bins = np.all(chrom_hic == 0, axis=1)
            # make crosses wider because close bins are often bad
            for step in range(self.enlarge_crosses):
                for j, fragment in enumerate(empty_bins):
                    fragment = fragment.copy()
                    for i in range(step, len(fragment)-1-step):
                        if fragment[i+1, 0]:
                            empty_bins[j][i] = True
                        if i > 0:
                            if fragment[i-1, 0]:
                                empty_bins[j][i] = True
            masks = self._update_masks(masks, empty_bins)

            # diag normalization 
            chrom_hic[masks==1] = np.nan
    
            
            # strange bins handling
            valid = chrom_hic[masks==0]
            valid[np.isnan(valid)] = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = np.nanmedian(chrom_hic)
                strange_bins = np.nanmedian(chrom_hic, axis=1) < self.cross_threshold * m
            masks = self._update_masks(masks, strange_bins)
            chrom_hic[masks==1] = np.nan

            chrom_hic = np.log(chrom_hic + self.psi)
            chrom_hic = data_utils.normalize_by_diag_means(chrom_hic)
            chrom_hic /= np.nanstd(chrom_hic)

            # crosses interpolation
            n_maps = chrom_hic.shape[0]
            n_experim = chrom_hic.shape[-1]
            chrom_hic = np.array([[data_utils.interp_nan(chrom_hic[i, ..., j]) for j in range(n_experim)] for i in range(n_maps)])
            chrom_hic = chrom_hic.transpose((0,2,3,1))

            if self.square_maps:
                square_hic[name] = chrom_hic
                square_masks[name] = masks
                self.square_w = chrom_hic.shape[1] # !!!
            # maps rotation - diagonals become rows
            chrom_hic = self._rotate_and_cut(chrom_hic)
            chrom_mask = self._rotate_and_cut(masks)

            # manually normalizing empty fragments
            empty_fragments = np.where(np.all(chrom_mask>0, axis=(1,2,3)))
            chrom_hic[empty_fragments] = 0

            chrom_hic = np.hstack(chrom_hic)
            chrom_mask = np.hstack(chrom_mask)

            '''valid = chrom_mask < 0.1
            valid_by_diag = [chrom_hic[i][valid[i]] for i in range(self.h)]
            means_by_diag = np.array([i.mean() for i in valid_by_diag])[:, None, None]
            std = np.std(chrom_hic[valid]) # not for each diagonal
            chrom_hic = (chrom_hic - means_by_diag) / std'''

            # to avoid possible deletions and isertions:
            if self.skip_quadrants:
                ectopic_quadrants = np.zeros(chrom_mask.shape, dtype=bool)
                for column_index in range(chrom_mask.shape[1]):
                    if np.any(chrom_mask[:, column_index] > 0):
                        a = np.flip(chrom_mask[:, column_index], axis=0)
                        a = np.cumsum(a, axis=0, dtype=bool)
                        ectopic_quadrants[:, column_index] = np.flip(a, axis=0)
                chrom_mask[ectopic_quadrants] = 1

            # gaussian blur
            if self.sigma:
                chrom_hic = gaussian_filter(chrom_hic, sigma=self.sigma)

            del masks#, valid
            gc.collect()
            HIC[name] = chrom_hic * self.scale_by
            MASKS[name] = chrom_mask
            print(f'Hi-C data for {name} is loaded')
        print()
        if self.square_maps:
            return HIC, MASKS, square_hic, square_masks
        return HIC, MASKS

    def scale_hic(self, multiplier):
        for chrom in self.chromnames:
            self.HIC[chrom] *= multiplier
        self.scale_by *= multiplier


    def _genomic_coord_to_hic(self, start, end=None):
        start = (start - self.mapped_len // 2) // self.binsize
        if end is None:
            return start
        else:
            end = (end - self.mapped_len // 2) // self.binsize
            return start, end

    def _hic_coord_to_genomic(self, start, end=None, interval=False):
        start = start * self.binsize + self.mapped_len // 2
        if interval:
            start = (start - self.binsize, start + self.binsize)
        if end is None:
            return start
        else:
            end = (end - self.binsize, end + self.binsize)
            if interval:
                end = (end - self.binsize, end + self.binsize)
            return start, end

    def _corresp_map_for_a_dna(self, start, end):
        '''Coordinates in the Hi-C map for DNA coordinates with an offset'''
        start += self.offset
        end -= self.offset
        start, end = self._genomic_coord_to_hic(start, end)
        return start, end

    def _corresp_dna_for_a_map(self, start, end):
        '''Coordinates with an offset of dna in the genome for given \
coordinates of the corresponding Hi-C map'''
        start, end = self._hic_coord_to_genomic(start, end, interval=False)
        return start - self.offset, end + self.offset


    def _check_train(self, chrom, start, end):
        '''Checks if the given region belongs to the train region'''
        for test_chrom, test_start, test_end in self.test_regions:
            if chrom == test_chrom:
                if any([test_start < start < test_end,
                        test_start < end < test_end,
                        start < test_start < end,
                        start < test_end < end]):
                    return False
        return True

    def _check_test(self, chrom, start, end):
        '''Checks if the given region DOESN'T belong to the train region'''
        for test_chrom, test_start, test_end in self.test_regions:
            if chrom == test_chrom:
                if test_start < start < test_end and test_start < end < test_end:
                    return True
        return False

    def _parse_test_split(self, test_split):
        '''parses 'test_split' argument'''
        if test_split == 'default':
            target_len = self.mapped_len * 101
            i = 0
            while self.chromsizes[self.chromnames[i]] < target_len:
                i += 1
                if i == len(self.chromsizes):
                    raise ValueError("Default train-test split failed - all chroms \
are too short. Type chroms for test sample manually in 'test_split' argument")
            return [(self.chromnames[i], 0, target_len)]
        else:
            regions = [data_utils._parse_coord(i) for i in test_split.split(';')]
            for i in range(len(regions)):
                end = regions[i][2]
                if end is None:
                    chrom = regions[i][0]
                    regions[i] = (chrom,
                                  regions[i][1],
                                  self.chromsizes[chrom])
            return regions



    def _slice_map(self, chrom, start, end, fixed_size=True, edge_policy='error'):
        '''returns mask of interpolated hi-c map pixels for the given region'''
        if self.square_maps: # !!!
            w = end-start
            ind = (start+64)//128
            first_bin = (start+64)%128
            return self.square_hic[chrom][ind,
                                        first_bin:first_bin+w,
                                        first_bin:first_bin+w]
        rev = False
        if start > end:
            rev = True
            start, end = end, start
        if start < 0:
            hic_map = self.HIC[chrom][:, 0:end]
        else:
            hic_map = self.HIC[chrom][:, start:end]
        chrom_len  = self.HIC[chrom].shape[1]
        h = hic_map.shape[0]
        c = hic_map.shape[2]
        if start < 0 or end > chrom_len:
            if edge_policy == 'error': # raises error
                raise ValueError(f'Coordinates {start, end} exсeed chrom {chrom} \
map size ({chrom_len})')
            elif edge_policy == 'empty':  # adds zero bins
                if start < 0:  
                    left_empty = np.zeros((h, -start, c))
                    hic_map = np.concatenate([left_empty, hic_map], axis=1)
                if end > chrom_len:
                    right_empty = np.zeros((h, end-chrom_len, c))
                    hic_map = np.concatenate([hic_map, right_empty], axis=1)
            elif edge_policy == 'cyclic': # adds bins from the other end
                if start < 0:  
                    left_empty = hic_map[:, start:]
                    hic_map = np.concatenate([left_empty, hic_map], axis=1)
                if end > chrom_len:
                    right_empty = hic_map[:, :end-chrom_len]
                    hic_map = np.concatenate([hic_map, right_empty], axis=1)
            else:
                raise ValueError("edge_policy should be 'error', 'empty' or 'cyclic'") 
        if rev:
            hic_map = np.flip(hic_map, axis=1)
        if fixed_size:
            return hic_map[:, :self.map_size]
        else:
            return hic_map


    def _slice_mask(self, chrom, start, end, fixed_size=True, edge_policy='error'):
        '''returns mask of interpolated hi-c map pixels for the given region'''
        if self.square_maps: # !!!
            w = end-start
            ind = (start+64)//128
            first_bin = (start+64)%128
            return self.square_masks[chrom][ind,
                                        first_bin:first_bin+w,
                                        first_bin:first_bin+w]
        rev = False
        if start > end:
            rev = True
            start, end = end, start
        if start < 0:
            hic_map = self.MASKS[chrom][:, 0:end]
        else:
            hic_map = self.MASKS[chrom][:, start:end]
        chrom_len  = self.MASKS[chrom].shape[1]
        h = hic_map.shape[0]
        c = hic_map.shape[2]
        if start < 0 or end > chrom_len:
            if edge_policy == 'error': # raises error
                raise ValueError(f'Coordinates {start, end} exсeed chrom {chrom} \
map size ({chrom_len})')
            elif edge_policy == 'empty':  # adds masked bins
                if start < 0:  
                    left_empty = np.ones((h, -start, c))
                    hic_map = np.concatenate([left_empty, hic_map], axis=1)
                if end > chrom_len:
                    right_empty = np.ones((h, end-chrom_len, c))
                    hic_map = np.concatenate([hic_map, right_empty], axis=1)
            elif edge_policy == 'cyclic': # adds bins from the other end
                if start < 0:  
                    left_empty = hic_map[:, start:]
                    hic_map = np.concatenate([left_empty, hic_map], axis=1)
                if end > chrom_len:
                    right_empty = hic_map[:, :end-chrom_len]
                    hic_map = np.concatenate([hic_map, right_empty], axis=1)
            else:
                raise ValueError("edge_policy should be 'error', 'empty' or 'cyclic'") 
        if rev:
            hic_map = np.flip(hic_map, axis=1)
        if fixed_size:
            return hic_map[:, :self.map_size]
        else:
            return hic_map

    def _slice_dna(self, chrom, start, end, seq=False, edge_policy='error', fixed_size=False):
        '''returns one-hot encoded dna from the given region'''
        rev_comp = False
        if start > end:
            rev_comp = True
            start, end = end, start
        if fixed_size:
            end = start + self.dna_len
        chrom_len = self.chromsizes[chrom]
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
            dna = data_utils._revcomp(dna) # !!! revcomp in sequence
        if seq:
            return dna
        else:
            dna = data_utils.one_hot(dna)
            #if rev_comp:
            #   dna = np.flip(dna)
            return dna

    def _find_good_regions(self, check=None, nan_threshold=None, step=1):
        '''Find all regions in Hi-C map with appropriate quality'''
        good_regions = []
        for chrom in self.chromnames:
            for i in range(0, self.HIC[chrom].shape[1]-self.map_size, step):
                region = (chrom, i, i + self.map_size)
                nan_check = self._check_nan(*region, nan_threshold)
                if nan_check: # nan percentage is not more than nan_threshold
                    if check == 'train':
                        if self._check_train(*region):
                            good_regions.append(region)
                    elif check == 'not-train':
                        if self._check_test(*region):
                            good_regions.append(region)
                    else:
                        good_regions.append(region)
        return good_regions

    def _check_nan(self, chrom, start, end, nan_threshold=None):
        '''Check if the percentage of interpolated NaN-bins in the given \
region is less than some threshold'''
        if nan_threshold is None:
            nan_threshold = self.nan_threshold
        nan_score = self.MASKS[chrom][:, start:end].mean()
        return nan_score <= nan_threshold


    def _make_regular_sample(self, check=None):
        '''Fragments are sampled with regular steps from the train or test \
region (or the whole genome if 'check' is None)'''
        step = self.mapped_len
        sample = []
        good = bad = 0
        for chrom in self.chromnames:
            max_start = self.chromsizes[chrom] - 4 * self.mapped_len - self.mapped_len // 2
            for start in range(self.mapped_len//2, max_start + 1, step):
                dna_start = start - self.offset
                dna_end = start + self.mapped_len + self.offset
                region = (chrom, dna_start, dna_end)
                if check == 'train':
                    if not self._check_train(*region):
                        continue
                elif check == 'not-train':
                    if not self._check_test(*region):
                        continue
                hic_start, hic_end = self._corresp_map_for_a_dna(dna_start, dna_end)
                hic_region = (chrom, hic_start, hic_end)
                if self._check_nan(*hic_region):
                    good += 1
                    sample.append(region)
                else:
                    bad +=1
        if len(sample) == 0:
            print('WARNING: ' + check.capitalize() + ' sample is empty')
            return []
        sample_name = check.capitalize() + ' sample: ' if check else 'Totally: '
        info = f'{bad/(bad+good)*100:.2f}% of maps excluded by NaN threshold'
        print(sample_name + info)
        return sample

    def _analyze_sample_sizes(self, train_sample, val_sample, test_sample):
        info = f'Train sample has {len(train_sample)} objects, \
validation sample - {len(val_sample)}, \
test sample - {len(test_sample)}'
        print(info)
        if len(train_sample) < 10000:
            if len(train_sample) < 1000:
                print('Train sample is very small, enlarge it')
            elif len(train_sample) < 5000:
                print('Train sample is rather small, may be you should enlarge \
it')
            else:
                print('Train sample seems large enough but you may enlarge it')
            print("If you used not all chromosomes, maybe you can use all")
            if self.nan_threshold < 0.75:
                print("If too many maps were excluded by nan_threshold may be \
you can increase it")
        else:
            _, start, end = train_sample[0]
            if end - start > 1000000:
                print('Train sample is large and dna fragments are long. \
Training will be very long, maybe you can reduce train sample')
                if self.offset:
                    print('You may set offset to 0 to shorten fragments')
                if self.nan_threshold:
                    print('You may remove all map fragments with interpolated \
pixels by setting nan_threshold=0')


    def _make_samples(self, val_fraction=0.33):
        '''Train/val/test split'''
        train_sample = self._make_regular_sample(check='train')
        test_sample = self._make_regular_sample(check='not-train')
        # split the test sample on val and test
        val_sample_size = int(len(test_sample) * val_fraction)
        val_sample = test_sample[:val_sample_size]
        test_sample = test_sample[val_sample_size:]
        # check if sample sizes are sufficient, give recommendations if not:
        self._analyze_sample_sizes(train_sample, val_sample, test_sample)
        # plot samples on chomosomes:
        if self.plot_samples:
                df = self._make_dataframe_of_samples(train_sample, val_sample, test_sample)
                print('Train sample is gray, val - orange, test - blue')
                plot_utils.plot_samples(self.chromsizes, df)
        return train_sample, val_sample, test_sample

    def _make_dataframe_of_certain_sample(self, sample, name):
        chroms, starts, ends = [], [], []
        for i in sample:
            chroms.append(i[0])
            starts.append(i[1])
            ends.append(i[2])
        sample = [name] * len(chroms)
        df = pd.DataFrame({
            'chrom': chroms,
            'start': starts,
            'end': ends,
            'sample': sample
        })
        return df

    def _make_dataframe_of_samples(self, train_sample, val_sample, test_sample):
        df = self._make_dataframe_of_certain_sample(train_sample, 'train')
        df = pd.concat([df, self._make_dataframe_of_certain_sample(val_sample, 'val')])
        df = pd.concat([df, self._make_dataframe_of_certain_sample(test_sample, 'test')])
        return df

    def _parse_experiment_name(self, experiment_name, experiment_index):
        if experiment_name:
            if experiment_name in self.experiment_names:
                experiment_index = self.experiment_names.index(experiment_name)
            else:
                raise ValueError("Dataset doesn't contain specified experiment")
        else:
            experiment_name = self.experiment_names[experiment_index]
        return experiment_name, experiment_index


    def plot_annotated_map(
            self,
            region=None,
            hic_map=None,
            chrom=None,
            start=None,
            middle=None,
            end=None,
            name=None,
            sample=None,
            index=None,
            experiment_index=0,
            experiment_name=None,
            ax=None,
            axis='both',
            x_top=False,
            brief=True,
            y_label_shift=False,
            show_position=True,
            full_name=True,
            mutations=None,
            mutation_names=None,
            genes=None,
            gene_names=None,
            motifs=None,
            motif_names=None,
            colorbar=True,
            show=False,
            zero_centred=False,
            vmin=None,
            vmax=None):
        '''Multiple annotations for Hi-C map plotting'''
        if ax is None:
            _, ax = plt.subplots()
        # if map array not privided, obtain it
        if hic_map is None:
            if sample is not None: # from some sample
                if sample=='train':
                    hic_map = self.y_train[index]
                elif sample=='val':
                    hic_map = self.y_val[index]
                elif sample=='test':
                    hic_map = self.y_test[index]
                else:
                    raise ValueError("'sample' argument not recognized")
            else: # by genomic coordinates
                if region is not None:
                    hic_map = self.fetch(region=region)
                elif all([i is not None for i in [chrom, start, end]]):
                    hic_map = self.fetch(chrom=chrom, start=start, end=end)
                elif all([i is not None for i in [chrom, middle]]):
                    hic_map = self.fetch(chrom=chrom, middle=middle)
                else:
                    raise ValueError('not enough info to fetch the fragment')
        if zero_centred: # color scale will be centred at 0
            min_val, max_val = hic_map.min(), hic_map.max()
            max_abs_val = max(np.abs(min_val), max_val)
            vmin, vmax = -max_abs_val, max_abs_val

        # if one of index or name is missing, restoring it
        experiment_name, experiment_index = self._parse_experiment_name(
            experiment_name,
            experiment_index
        )

        # plot main data
        plot_utils.plot_map(hic_map, ax=ax, vmin=vmin, vmax=vmax,
                            colorbar=colorbar, name=name, hide_axis=False,
                            experiment_index=experiment_index)

        if region is not None:
            chrom, start, end = self._parse_region(region)

        #if map comes from some sample find its genomic location to annotate it:
        if start is None:
            if sample is not None:
                if sample=='train':
                    chrom, start, end = self.x_train.regions[index]
                elif sample=='val':
                    chrom, start, end = self.x_val.regions[index]
                elif sample=='test':
                    chrom, start, end = self.x_test.regions[index]
                else:
                    raise ValueError('incorrect sample argument')
                start, end = start + self.offset, end - self.offset
            else:
                chrom, start, end = 'Unknown', 0, hic_map.shape[1]*self.resolution


        # annotate axes
        cbottom, ctop = plot_utils.annotate(
            ax,
            start,
            end,
            axis,
            h=self.h,
            x_top=x_top,
            brief=brief,
            remove_first_diag=self.remove_first_diag,
            w=hic_map.shape[1],
            y_label_shift=y_label_shift
        )
        # make title with genomic coordinates
        if show_position:
            if full_name:
                ctop = plot_utils.annotate_title(
                    ax,
                    chrom,
                    start,
                    end,
                    organism=self.organism,
                    experiment_name=experiment_name,
                    position='top',
                    vertical_shift=ctop
                )
            else:
                ctop = plot_utils.annotate_title(
                    ax,
                    chrom,
                    start,
                    end,
                    experiment_name=experiment_name,
                    position='top',
                    vertical_shift=ctop
                )
        # point mutation positions if provided
        if mutations is not None:
            map_positions = ((mutations - self.offset) / self.resolution).astype(int)
            cbottom = plot_utils.annotate_mutations(
                ax,
                positions=map_positions,
                dna_positions=mutations,
                names=mutation_names,
                vertical_shift=cbottom
            )
        # draw genes with orientations if provided
        if genes is not None:
            map_positions = ((genes - self.offset) / self.resolution).astype(int)
            cbottom = plot_utils.annotate_boxes(
                ax,
                self,
                positions=genes,
                names=gene_names,
                vertical_shift=cbottom
            )

        if show:
            pass #plt.show()


    def _parse_region(self, region):
        if isinstance(region, str): # convert single coord string into tuple
            region = data_utils._parse_coord(region)
        if len(region) == 3: # chrom, start, end
            chrom, start, end = region
        else: # chrom, single position (a middle of fixed size fragment)
            chrom, middle = region
            start = middle - self.dna_len // 2
            end = middle + self.dna_len - self.dna_len // 2
        return chrom, start, end


    def get_dna(self, region, return_parsed_region=False, seq=False, edge_policy='error'):
        parsed_region = self._parse_region(region)
        chrom, dna_start, dna_end = parsed_region
        dna = self._slice_dna(chrom, dna_start, dna_end, seq=seq, edge_policy=edge_policy)
        if return_parsed_region:
            return dna, parsed_region # returns dna as array and parsed coords
        else:
            return dna


    def get_dna_loader(self, region, special_dna=None, edge_policy='error'):
        parsed_region = self._parse_region(region)
        chrom, dna_start, dna_end = parsed_region
        length = dna_end - dna_start
        n_full_fragments = length // self.mapped_len
        remainder_length = length % self.mapped_len
        # what fraction of last fragment required:
        remainder_frac = remainder_length / self.mapped_len
        n_fragments_total = n_full_fragments + (remainder_length > 0)
        last_nucl =  n_fragments_total * self.mapped_len + self.offset
#         if special_dna is None:
#             if last_nucl > self.chromsizes[chrom] or dna_start < self.offset:
#                 raise ValueError(f'DNA region ({region}) with required offset \
# ({self.offset}) exceeds chrom size ({self.chromsizes[chrom]})')
        regions = []
        for current_dna_start in range(
                dna_start,
                dna_end+1,
                self.mapped_len
                ):
            current_dna_end = current_dna_start + self.mapped_len + self.offset
            current_dna_start = current_dna_start - self.offset
            regions.append((chrom, current_dna_start, current_dna_end))
        if special_dna is None:
            loader = data_utils.DNALoader(self, regions, edge_policy=edge_policy)
        else: # for predicting mutant DNA fragments
            loader = data_utils.MutantDNALoader(self, special_dna, regions, edge_policy=edge_policy)
        return loader, parsed_region, remainder_frac


    def _get_hic_coords(self, region):
        chrom, dna_start, dna_end = self._parse_region(region)
        hic_start, hic_end = self._genomic_coord_to_hic(dna_start, dna_end)
        return chrom, hic_start, hic_end

    def get_hic(self, region, plot=False, experiment_index=0, edge_policy='empty'):
        parsed_region = self._parse_region(region)
        chrom, hic_start, hic_end, = self._get_hic_coords(region)
        hic = self._slice_map(
            chrom,
            hic_start,
            hic_end,
            fixed_size=False,
            edge_policy=edge_policy
            )
        if plot:
            self.plot_annotated_map(
                hic_map=hic,
                region=region,
                axis='x',
                experiment_index=experiment_index
                )
        else:
            return hic

    def get_mask(self, region, plot=False, edge_policy='empty'):
        parsed_region = self._parse_region(region)
        chrom, hic_start, hic_end, = self._get_hic_coords(region)
        hic = self._slice_mask(
            chrom,
            hic_start,
            hic_end,
            fixed_size=False,
            edge_policy=edge_policy
            )
        if plot:
            plot_utils.plot_map(hic, title=region)
        else:
            return hic

    def show(self, experiment_index=0, exclude_imputed=True):
        for region in self.test_regions:
            chrom, start, end = region
            region = f'{chrom}:{start+self.mapped_len}-{end-self.mapped_len}'
            hic = self.get_hic(region)
            if exclude_imputed:
                mask = self.get_mask(region)
                hic = hic.copy()
                hic[mask > 0] = np.nan
            for i in range(0, hic.shape[1]//1000*1000, 1000):
                _, ax = plt.subplots(figsize=(30,1))
                self.plot_annotated_map(hic_map=hic[:, i:i+1000],
                                        chrom=chrom,
                                        start=start+self.mapped_len+i*self.resolution,
                                        end=start+self.mapped_len+(i+1000)*self.resolution,
                                        experiment_index=experiment_index,
                                        axis='x',
                                        ax=ax,
                                        vmin=np.nanmin(hic),
                                        vmax=np.nanmax(hic),
                                        show=True,
                                        )
