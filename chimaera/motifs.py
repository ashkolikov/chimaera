import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

import torch.nn.functional as F

from .ism import SeqMatrix, MutGenerator
from . import plot_utils
from . import data_utils

class Motif(SeqMatrix):
    '''Class for motifs storing. Can sample sequences from pfm'''
    def __init__(
            self,
            pfm=None,
            file=None,
            seq=None,
            alignment=None,
            name=''
        ):
        self.name = name
        self.onehot = OneHotEncoder(sparse_output=False)
        self.onehot.fit(np.arange(4).reshape(4,1))
        if file is not None:
            pfm = read_pfm_from_file(file)

        elif alignment is not None:
            if isinstance(alignment[0], str):
                pfm = align_to_pfm(alignment)
            else:
                pfm = one_hot_align_to_pfm(alignment)

        elif seq is not None:
            pfm = str_to_pfm(seq)

        if len(pfm.shape) != 2 or pfm.shape[1] != 4:
            raise ValueError(f'Pfm shape should be (n, 4) but is {pfm.shape}')
        if np.any(pfm.sum(axis=1) > 1):
            pfm = pfm / pfm.sum(axis=1)[..., None]
        self.pfm = pfm
        self.consensus = self._make_consensus()
        self.revcomp = Motif(pfm=revcomp(self.pfm))

    def __len__(self):
        return len(self.pfm)

    def rc(self):
        return self.revcomp

    def rc_logo(self, *args):
        self.revcomp.logo(*args)

    def spawn(self, rc=False, shuffle=False):
        pfm = self.pfm
        if rc:
            pfm = np.flip(pfm)
        if shuffle:
            pfm = np.random.permutation(pfm)
        c = pfm.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        sample = (u < c).argmax(axis=1)
        return self.onehot.transform(sample[:, None])

    def _make_consensus(self):
        seq = self.pfm.argmax(axis=1)
        return ''.join(['acgt'[i] for i in seq])


    def logo(self, rc=False, ax=None):
        pfm = self.pfm if not rc else revcomp(self.pfm)
        return pfm_to_logo(pfm, ax=ax)

    def find_in_dna(self, dna):
        scores = scan(dna, self.pfm)
        return scores

def align_to_pfm(seqs):
    seqs = [list(i.lower()) for i in seqs]
    mtx = np.array(seqs).T
    pfm = np.zeros((mtx.shape[0], 4))
    for i in range(len(mtx)):
        for j in range(4):
            letter = 'acgt'[j]
            pfm[i,j] = (mtx[i]==letter).mean()
    return pfm

def one_hot_align_to_pfm(seqs):
    return np.mean(seqs)

def pfm_to_pwm(pfm):
    pfm = pfm + 0.001
    return np.log2(pfm / 0.25)

def parse_jaspar(motif):
    motif=motif.split('\n')
    motif = [[int(j) for j in i.split('[')[1][:-1].split(' ') if j] for i in motif]
    motif = np.array(motif)
    motif = motif/motif[:,0].sum()
    return Motif(pfm=motif.T)

def to_jaspar(arr):
    result = []
    for nucl, line in zip ('ACGT', arr.T):
        line = [str(i) for i in line]
        result.append(nucl + ' [' + ' '.join(line) + ']')
    return '\n'.join(result)

def pfm_to_logo(pfm, ax=None):
    if np.any(pfm<0):
        raise ValueError('pfm should be positive')
    if np.any(pfm>1):
        raise ValueError('pfm values should be <1')
    pfm = pfm + 0.001
    pwm = np.log2(pfm / 0.25)
    ic = np.sum(pfm * pwm, axis=1)[:, None]
    ic = ic * pfm
    plot_utils.LogoPlotter().plot_logo(ic=ic, ax=ax)

def str_to_pfm(seq):
        seq = seq.lower()
        encoded_seq = []
        for i in seq:
            if i == 'a':
                encoded_seq.append([1,0,0,0])
            elif i == 'c':
                encoded_seq.append([0,1,0,0])
            elif i == 'g':
                encoded_seq.append([0,0,1,0])
            elif i == 't':
                encoded_seq.append([0,0,0,1])
            elif i == 'n':
                encoded_seq.append([0.25,0.25,0.25,0.25])
            elif i == 'r':
                encoded_seq.append([0.5,0,0.5,0])
            elif i == 'y':
                encoded_seq.append([0,0.5,0,0.5])
            elif i == 'k':
                encoded_seq.append([0,0,0.5,0.5])
            elif i == 'm':
                encoded_seq.append([0.5,0.5,0,0])
            elif i == 's':
                encoded_seq.append([0,0.5,0.5,0])
            elif i == 'w':
                encoded_seq.append([0.5,0,0,0.5])
            elif i == 'b':
                encoded_seq.append([0,1/3,1/3,1/3])
            elif i == 'd':
                encoded_seq.append([1/3,0,1/3,1/3])
            elif i == 'h':
                encoded_seq.append([1/3,1/3,0,1/3])
            elif i == 'v':
                encoded_seq.append([1/3,1/3,1/3,0])
            else:
                raise ValueError(f"Symbol '{i}' is not a nucleotide code")
        return np.array(encoded_seq)

def read_pfm_from_file(file):
    if file.endswith('.npy'):
        return np.load(file)
    else:
        return np.array(pd.read_csv(index_col=None, header=None, sep='\t'))

def shuffle(pfm):
    return np.random.permutation(pfm)

def revcomp(pfm):
    return np.flip(pfm)

def _revcomp(dna):
    revcomp_site = []
    pairs = {'a':'t','c':'g','g':'c','t':'a','n':'n'}
    return ''.join(map(lambda base: pairs[base], dna))[::-1]
    
def one_hot(seqs):
    if isinstance(seqs, str):
        seqs = [seqs]
    encoded_seqs = []
    for seq in seqs:
        mapping = dict(zip("acgtn", range(5)))
        encoded_seqs.append([mapping[i] for i in seq])
    return np.eye(5)[encoded_seqs][..., :4]

def scan(dna, pfm):
    '''Scan DNA with pfm'''
    pwm = pfm_to_pwm(pfm)
    if len(dna.shape) < 3:
        dna = dna[None, ...]
    motif_size = pwm.shape[1]
    pwm = torch.Tensor(pwm.transpose((0,2,1))).cuda()
    dna = torch.Tensor(dna.transpose((0,2,1))).cuda()
    scores = F.conv1d(dna, pwm, padding=0)
    scores = scores.cpu().detach().numpy()[:,0,:]
    return scores

def mean_motif_effect(
        Model,
        motif,
        composition='>',
        long_spacer_length='auto',
        number=100,
        strand='one',
        between_insertions=20,
        experiment_index=0,
        fixed_scale=True,
        sample='val',
        plot=True,
        normalize=True,
        get_maxima=False
    ):
    '''Mean of maps predicted from sequences with motif insertion'''
    random_shifts = []
    if sample == 'val':
        regions_pool = Model.data.x_val.regions
    elif sample == 'test':
        regions_pool = Model.data.x_test.regions
    elif sample == 'both':
        regions_pool = Model.data.x_test.regions + Model.data.x_val.regions
    else:
        raise ValueError('Sample should be train, val or both')
    while len(random_shifts) < number:
        chrom, start, end = regions_pool[np.random.randint(len(regions_pool))]
        random_shift = np.random.randint(-Model.data.dna_len//2, Model.data.dna_len//2)
        if (start + random_shift) > 0 and (end + random_shift) < len(Model.data.DNA[chrom]):
            random_shifts.append((chrom, start+random_shift, end+random_shift))

    loader = data_utils.DNALoader(Model.data, random_shifts)
    if normalize:
        y_wt = Model.predict(
            data_utils.DNAPredictGenerator(
                loader,
                batch_size=Model.batch_size
                ),
            strand=strand,
            verbose=0
        )
    gen = MutGenerator(
        loader,
        motif,
        batch_size=Model.batch_size,
        between_insertions=between_insertions,
        long_spacer_length=long_spacer_length,
        composition=composition,
        offset=Model.data.offset)
    y_mut = Model.predict(gen, strand=strand, verbose=0)

    if normalize:
        if get_maxima:
            result = np.abs(y_mut-y_wt).max(axis=(1,2,3)).mean()
        else:
            result = (y_mut-y_wt).mean(axis=0)
    else:
        if get_maxima:
            raise ValueError('Set normalize=True if get_maxima is True')
        result = y_mut.mean(axis=0)
    if plot:
        _, ax = plt.subplots(1,1, figsize=(7,2))
        if fixed_scale:
            vmax = Model.sd
            vmin = -vmax
        else:
            vmin = vmax = None
        plot_utils.plot_map(
            result,
            experiment_index=experiment_index,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        plot_utils.annotate_boxes(ax, Model.data, gen.boxes[0])
    else:
        return result

def meme(seqs, size=10, random_seqs=100, runs=25):
    total_scores = []
    pfms = []
    inits = [''.join(['atgc'[i] for i in np.random.choice(4,size)]) for i in range(random_seqs)]
    for i in inits:
        #print(i)
        pfm = one_hot([i])
        for j in range(runs):
            dna = one_hot(seqs)
            scores = scan(dna, pfm)
            scores_rc = scan(dna, np.flip(pfm))
            best_scores_forw = scores.max(axis=1)
            best_scores_rc = scores_rc.max(axis=1)
            best_scores = np.maximum(best_scores_forw, best_scores_rc)
            chains = best_scores_forw > best_scores_rc
            positions_forw = scores.argmax(axis=1)
            positions_rc = scores_rc.argmax(axis=1)
            choose_chain = lambda idx, chain: positions_forw[idx] if chain else positions_rc[idx]
            positions = [choose_chain(k, chains[k]) for k in range(len(chains))]
            get_seq = lambda seq, chain: seq if chain else _revcomp(seq)
            found_motifs = [get_seq(seq[pos:pos+size], chain) for seq, pos, chain in zip(seqs, positions, chains)]
            pfm = align_to_pfm(found_motifs)[None,:]
        total_scores.append(np.mean(best_scores))
        pfms.append(pfm[0])
    return pfms[np.argmax(total_scores)]

def check_motif(Model, motif, vector, name=None, experiment_index=0):
    '''Shows motif insertion effect on a projection of a predicted map on some \
specified vector'''
    vector /= np.linalg.norm(vector)
    projections = []
    ns = [1,2,4,8,16]
    generator = data_utils.DNAPredictGenerator(Model.data.x_test,
                            batch_size=Model.batch_size)
    y_wt = Model.dna_encoder.predict(generator, verbose=0)[:, experiment_index]
    for modification in ['>', '~']:
        proj = []
        for i in ns:
            composition = modification * i
            generator = MutGenerator(Model.data.x_test,
                                    motif,
                                    batch_size=Model.batch_size,
                                    between_insertions=20,
                                    composition=composition,
                                    offset=Model.data.offset,
                                    )
            y_mut = Model.dna_encoder.predict(generator, verbose=0)[:, experiment_index]
            proj.append(np.dot(y_mut-y_wt, vector.T)[:,0])
        projections.append(proj)
    if name:
        motif.name = name
    if motif.name != '':
        site_name = motif.name[0].upper() + motif.name[1:]
        control_name = 'Shuffled ' + motif.name
    else:
        site_name = 'Site of interest'
        control_name = 'Shuffled site'
    plot_utils.plot_motiff_effect(ns, np.array(projections), [site_name, control_name])
    plot_utils.plot_significance_between(projections)
