import numpy as np
from random import shuffle
from .data_utils import DataGenerator, _revcomp


class SeqMatrix():
    '''Parent class for inheriting. Used for making insertions/substitutions \
dynamically '''
    pass

'''class RandomSeq(SeqMatrix):
    def __init__(self, p, length):
        self.p = p
        self.length = length
        self.onehot = OneHotEncoder(sparse=False)
        self.onehot.fit(np.arange(4).reshape(4,1))

    def __len__(self):
        return self.length

    def spawn(self, **kwargs):
        random_seq = np.random.choice(4, (self.length, 1), p=self.p)
        return self.onehot.transform(random_seq)'''


def parse_composition(
        composition,
        length,
        short_spacer_length,
        long_spacer_length
    ):
    '''Parses 'composition' argument for many other methods'''
    compositions = composition.split(',')
    all_boxes = []
    for composition in compositions:
        start = 0
        boxes = []
        for symbol in composition:
            if symbol == '_':
                start += long_spacer_length
            else:
                end = start + length
                if symbol == '>':
                    orientation = 'f'
                elif symbol == '<':
                    orientation = 'rc'
                elif symbol == '~':
                    orientation = 'shuffle'
                boxes.append((orientation, start, end))
                start = end + short_spacer_length
        all_boxes.append(boxes)
    return all_boxes

def insert_seq(
        dna,
        insertion,
        positions='auto',
        composition='>',
        between_insertions=20,
        long_spacer_length='auto',
        anchor='center',
        offset=0
    ):
    '''Inserts (substitutes) mutation into one-hot dna fragment. May add \
multiple same mutations in specified composition'''
    if not isinstance(insertion, SeqMatrix):
        raise ValueError('Only SeqMatrix inheriting objects accepted')
    if len(dna.shape) == 2:
        dna = dna[None, ...]
    l = dna.shape[1]
    if long_spacer_length == 'auto':
        long_spacer_length = (l - 2*offset) // 4
    all_boxes = parse_composition(composition,
                                  length = len(insertion),
                                  short_spacer_length = between_insertions,
                                  long_spacer_length = long_spacer_length)

    if positions == 'auto':
        if not offset:
            start, end = 0, l
        elif isinstance(offset, int):
            start, end = offset, l - offset
        else:
            raise ValueError('incorrect offset')
        positions = np.linspace(start, end, len(all_boxes)+2)[1:-1].astype(int)
    elif isinstance(positions, int):
        positions = [positions]
    result_boxes = []
    batch_size = dna.shape[0]
    for j in range(batch_size):
        for boxes, position in zip(all_boxes, positions):
            orientations = [i[0] for i in boxes]
            boxes = np.array([i[1:] for i in boxes])
            total_length = boxes.max() - boxes.min()
            if anchor == 'center':
                boxes -= total_length // 2
            elif anchor == 'right':
                boxes -= total_length
            elif anchor != 'left':
                raise ValueError('Anchor should be left, center or right')
            boxes += position
            boxes[boxes < 0] = 0
            boxes[boxes > l] = l
            for box, orientation in zip(boxes, orientations):
                start, end = box
                if orientation == 'rc':
                    site = insertion.spawn(rc=True)
                    box = [box[1], box[0]]
                elif orientation == 'shuffle':
                    site = insertion.spawn(shuffle=True)
                else:
                    site = insertion.spawn()
                #site = np.repeat(site[None, ...], dna.shape[0], axis=0)
                box_len = end - start
                site_length = site.shape[0]
                if box_len > site_length:
                    flank = (box_len - site_length) // 2
                    start += flank
                    end -= (box_len - site_length - flank)
                if start==0:
                    site = site[:, -end:]
                if end==l:
                    site = site[:, :end-start]
                dna[j, start:end] = site
                result_boxes.append(box)
    return dna, result_boxes


class MutGenerator(DataGenerator):
    '''Making new sequences with mutations just for loading into the model'''
    def __init__(
            self,
            x,
            mutations,
            batch_size,
            strategy='auto',
            composition='>',
            positions='auto',
            between_insertions=20,
            long_spacer_length='auto',
            anchor='center',
            return_wt_too=False,
            offset=0
        ):
        self.rc = False
        self.return_wt_too = return_wt_too
        self.batch_size = batch_size
        self.kwargs = {'composition':composition,
                       'positions':positions,
                       'between_insertions':between_insertions,
                       'long_spacer_length':long_spacer_length,
                       'anchor':anchor,
                       'offset':offset}
        self.boxes = []
        if isinstance(x, np.ndarray):
            if len(x.shape) < 3:
                x = x[None, ...]
            self.l = x.shape[1]
        else:
            self.l = x[0].shape[1]
        self.x = x

        if not isinstance(mutations, list):
            mutations = [mutations]
        self.mutations = mutations

        if strategy not in ['auto',
                            'one_to_one',
                            'all_to_one',
                            'one_to_all',
                            'all_to_all']:
            raise ValueError('Strategy should be auto, one_to_one, all_to_one, \
one_to_all or all_to_all')
        if strategy == 'auto':
            if len(x) == len(mutations):
                strategy = 'one_to_one'
            elif len(x) == 1:
                strategy = 'all_to_one'
            elif len(mutations) == 1:
                strategy = 'one_to_all'
            else:
                strategy = 'all_to_all'
        self.strategy = strategy
        if self.strategy == 'all_to_all':
            self.n = len(self.mutations) * len(self.x)
        else:
            self.n = max(len(self.mutations), len(self.x))
        self.indices = np.arange(self.n)
        '''if strategy == 'all_to_one':
            self.x = np.repeat(self.x, batch_size, axis=0)'''

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, index):
        n_mutations = len(self.mutations)
        indices = self.indices[index * self.batch_size :
                               (index + 1) * self.batch_size]
        if self.strategy == 'one_to_all': # one mutation to all fragments
            x_wt = self.x[indices]
            n_fragments = len(x_wt)
            mutation = self.mutations[0]
            if self.return_wt_too:
                x_native = x_wt.copy()
            x, boxes = insert_seq(x_wt, mutation, **self.kwargs)
            self.boxes.append(boxes)
        elif self.strategy == 'all_to_one': # all mutations in one fragment
            mutations = [self.mutations[ind] for ind in indices]
            x_wt = self.x
            n_fragments = len(x_wt)
            if self.return_wt_too:
                x_native = x_wt.copy()
            x = np.zeros((n_fragments, self.l, 4))
            boxes = []
            for i in range(n_fragments):
                x_, boxes_ = insert_seq(x_wt, mutations[i], **self.kwargs)
                boxes.append(boxes_)
                x[i] = x_[0]
            self.boxes.append(boxes)
        elif self.strategy == 'one_to_one': # one mutation to one fragment
            x_wt = self.x[indices]
            n_fragments = len(x_wt)
            if self.return_wt_too:
                x_native = x_wt.copy()
            mutations = [self.mutations[ind] for ind in indices]
            x = np.zeros((n_fragments, self.l, 4))
            boxes = []
            for i in range(n_fragments):
                x_, boxes_ = insert_seq(x_wt[i:i+1], mutations[i],**self.kwargs)
                boxes.append(boxes_)
                x[i] = x_[0]
            self.boxes.append(boxes)
        elif self.strategy == 'all_to_all': # all mutations to all fragments
            x_wt = self.x[indices // n_mutations]
            n_fragments = len(x_wt)
            if self.return_wt_too:
                x_native = x_wt.copy()
            mutations = [self.mutations[ind] for ind in indices % n_mutations]
            x = np.zeros((n_fragments, self.l, 4))
            boxes = []
            for i in range(n_fragments):
                x_, boxes_ = insert_seq(x_wt[i:i+1], mutations[i],**self.kwargs)
                boxes.append(boxes_)
                x[i] = x_[0]
            self.boxes.append(boxes)
        if self.rc:
            x = np.flip(x, axis=(1,2))
        if self.return_wt_too:
            return x, x_native
        else:
            return x

def modify(genome, chrom, start, end, modification):
    '''Makes mutations directly in the genome dict'''
    if modification == '~':
        shuffled = np.random.permutation(genome[chrom][start:end])
        genome[chrom][start:end] = list(shuffled)
    elif modification == '<':
        genome[chrom][start:end] = list(_revcomp(genome[chrom][start:end]))
    else:
        raise ValueError('Modification type not understood')

def modify_genome(data, table, modification, regions=None, threshold=None):
    '''Add mutations to the whole genome'''
    new_genome = {chrom:list(dna) for chrom, dna in data.DNA.items()}
    for _, line in table.iterrows():
        if line.chrom not in data.chromnames:
            continue
        if threshold is not None:
            if line.score > threshold:
                modify(new_genome, line.chrom, line.start, line.end, modification)
        else:
            modify(new_genome, line.chrom, line.start, line.end, modification)
    return {chrom:''.join(dna) for chrom, dna in new_genome.items()}

def substitution(dna, start, end, new_seq):
    if end is None:
        end = start + len(new_seq)
    return dna[:start] + new_seq + dna[end:]

def insertion(dna, start, new_seq):
    return dna[:start] + new_seq + dna[start:]

def deletion(dna, start, end, return_removed=False):
    removed_dna = dna[start:end]
    dna = dna[:start] + dna[end:]
    if return_removed:
        return dna, removed_dna
    else:
        return dna

def permutation(dna, n):
    size = len(dna)
    step = size//n
    fragments = [dna[i*step:(i+1)*step] for i in range(n)]
    shuffle(fragments)
    return ''.join(fragments)

def inversion(dna, start, end):
    return dna[:start] + _revcomp(dna[start:end]) + dna[end:]

def _apply_mutation(
        mutation,
        dna,
        region_start,
        new_seq=None,
        return_removed=False,
    ):
    mutation_class, region = mutation.split(':')
    if mutation_class == 'trans': # combination of deletion and insertion
        initial_region, region = region.split('->')
        mutation1 = 'del:' + initial_region
        dna, new_seq = _apply_mutation(mutation1, dna, region_start,
                                       return_removed=True)
        mutation_class = 'ins'

    region = region.replace(',', '')
    region = region.split('-')
    if len(region) == 2:
        start, end = region
        start, end = int(start.strip()), int(end.strip())
        start, end = start - region_start, end - region_start
    elif len(region) == 1:
        if mutation_class == 'perm':
            start = 0
            end = None
            n = int(region[0].strip())
        else:
            start = int(region[0].strip()) - region_start
            end = None
    else:
        raise ValueError('Invalid mutation description')
    if end is not None and end < start:
        raise ValueError('End < start')

    if mutation_class == 'sub':
        return substitution(dna, start, end, new_seq)
    elif mutation_class == 'ins':
        return insertion(dna, start, new_seq)
    elif mutation_class == 'del':
        return deletion(dna, start, end, return_removed=return_removed)
    elif mutation_class == 'inv':
        return inversion(dna, start, end)
    elif mutation_class == 'perm':
        return permutation(dna, n)
    else:
        raise ValueError("Mutation class should be 'trans', 'inv', 'ins', 'del'\
 , 'sub' or 'perm'")

def mutate(data, region, mutations, seqs=None, edge_policy='error'):
    _, region_start, _ = data._parse_region(region)
    dna = data.get_dna(region, seq=True, edge_policy=edge_policy)
    if isinstance(mutations, str):
        mutations = [mutations]
    if isinstance(seqs, str):
        seqs = [seqs]
    elif seqs is None:
        seqs = []
    require_new_dna = []
    for mutation in mutations:
        if mutation.startswith('ins') or mutation.startswith('sub'):
            require_new_dna.append(mutation)
    if len(require_new_dna) != len(seqs):
        raise ValueError(f'{len(seqs)} new sequences provided for \
{len(require_new_dna)} mutations requiring new sequences (insertions and \
substitutions)')
    seqs = iter(seqs)
    for mutation in mutations:
        if mutation in require_new_dna:
            new_seq = next(seqs)
        else:
            new_seq = None
        dna = _apply_mutation(mutation, dna, region_start, new_seq)
    dna = {'mutant_fragment':dna}
    return dna