import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from gimmemotifs.comparison import MotifComparer
from gimmemotifs.motif import motif_from_align
from gimmemotifs.motif import read_motifs
 


class MutSeqGen(Sequence):
    '''Making new sequences with mutations just before loading into model - saves memory'''
    def __init__(self, x, mutations, positions, anchor, batch_size=8):
        assert len(mutations) == len(positions)
        self.x = x
        self.mutations = mutations
        self.positions = positions
        self.batch_size = batch_size
        self.anchor = anchor
        self.indexes = np.arange(len(mutations))

    def __len__(self):
        return int(np.ceil(len(self.mutations) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x = np.repeat(self.x, len(batch_indexes), axis=0)
        for i, ind in enumerate(batch_indexes):
            mut = self.mutations[ind]
            if mut is None: # some methods may proceed no mutation
                continue
            if isinstance(mut, list) or isinstance(mut, np.ndarray):
                mutations = [mt.spawn() for mt in mut]
            else:
                mutations = mut.spawn()
            if not isinstance(mutations, list):
                mutations = [mutations]
            positions = self.positions[ind]
            if not (isinstance(positions, list) or isinstance(positions, np.ndarray)):
                positions = [positions]
            for pos, m in zip(positions, mutations):
                if m is None: # Mutation() class can also proceed no mutation
                    continue
                l = len(m)
                if self.anchor == 'center':
                    pos = max(pos - l // 2, 0)
                elif self.anchor == 'right':
                    pos = max(pos - l, 0)
                elif self.anchor == 'left':
                    pass
                else:
                    raise ValueError('Anchor should be left, center or right')
                x[i, pos : pos + l] = m

        return x, None

class Mutation():
    def __init__(self,
                 mutation=None,
                 p=None,
                 length=None,
                 permute_once=False,
                 permute_each_time=False,
                 revcomp_each_time=False,
                 revcomp=False,
                 rev=False,
                 comp=False,
                 stochastic=True):
        self.permute_once = permute_once
        self.permute_each_time = permute_each_time
        self.revcomp_each_time = revcomp_each_time
        self.stochastic = stochastic
        self.length = length
        self.p = p
        self.onehot = OneHotEncoder(sparse=False)
        self.onehot.fit(np.arange(4).reshape(4,1))

        mutation = self.preprocess(mutation)
        self.mutation = self.modify(mutation, permute_once, rev, comp, revcomp)

    def __len__(self):
        if self.mutation is None:
            return 0
        if isinstance(self.mutation, int):
            return self.mutation
        return len(self.mutation)

    def permute(self, seq):
        return np.random.permutation(seq)

    def revcomp(self, seq):
        return np.flip(seq)

    def comp(self, seq):
        return np.flip(seq, axis=1)
    
    def rev(self, seq):
        return np.flip(seq, axis=0)

    def preprocess(self, mutation):
        if mutation is None:
            if self.length is not None: # otherwise mutation remains None - no mutation
                if isinstance(self.length, list):
                    return self.length
                else:
                    return int(self.length)
        if isinstance(mutation, str):
            return self.str_to_arr(mutation)
        return mutation
    
    def modify(self, seq, permute_once, rev, comp, revcomp):
        if permute_once:
            seq = self.permute(seq)
        if revcomp:
            seq = self.revcomp(seq)
        if rev:
            seq = self.rev(seq)
        if comp:
            seq = self.comp(seq)
        return seq


    def str_to_arr(self, seq):
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
                raise ValueError(f'Input string ({seq}) is considered as nucleotide sequence but it contains non-nucleotide symbol {i}')
        return np.array(encoded_seq)
      
    def bernulli_seq(self, length):
        random_seq = np.random.choice(4, (length, 1), p=self.p)
        return self.onehot.transform(random_seq)

    def spawn(self):
        if isinstance(self.mutation, int):
            return self.bernulli_seq(self.mutation)
        if isinstance(self.mutation, list):
            return [self.bernulli_seq(i) for i in self.mutation]
        if self.permute_each_time:
            self.mutation = self.permute(self.mutation)
        if self.revcomp_each_time:
            self.mutation = self.revcomp(self.mutation)
        if self.stochastic:
            if self.mutation.max(axis=1).mean() == 1: # not to make random sample in trivial case
                return self.mutation
            try:
                profile = self.mutation / self.mutation.sum(axis=1)[:,None]
                c = profile.cumsum(axis=1)
                u = np.random.rand(len(c), 1)
                sample = (u < c).argmax(axis=1)
                return self.onehot.transform(sample[:, None])
            except:
                raise ValueError("Array can't be a frequency matrix")
            
        return self.mutation

class Combination():
    def __init__(self, mutations):
        self.mutations = mutations

    def spawn(self):
        return np.concatenate([i.spawn() for i in self.mutations])

    def __len__(self):
        return sum([len(i) for i in self.mutations])

           

class Mutagenesis():
    def __init__(self, Model, seq_model='bernulli', best_preds_quatile=0.2, sample_size=32, select_samples=False):
        self.Model = Model
        #self.att_model = Model.att_model
        self.data = Model.data

        g = c = self.data.gc_content / 2 # imperfect but close to real
        a = t = (1 - self.data.gc_content) / 2
        self.background_p = [a, c, g, t]
        
        if select_samples:
            data_good = self.select_good_predictions(best_preds_quatile, sample_size)
            self.x_val_good, self.y_val_good, self.idx = data_good

    def select_good_predictions(self, quantile, sample_size):
        val_scores,_ = self.Model.score(metric='pearson', plot=False, strand='one')
        val_scores = np.mean(val_scores, axis=-1)
        n = int(quantile * len(val_scores))
        idx = np.argsort(val_scores)[-n:]
        idx = np.random.choice(idx, sample_size)
        x_val_good = DNALoader(self.Model.data, self.Model.data._x_val[idx])
        y_val_good = self.Model.y_val[idx]
        return x_val_good, y_val_good, idx

    def mut_analyse_attention(self,
                              model,
                              n,
                              k=10, 
                              eps_power=10,
                              data='val',
                              **kwargs):
        if data == 'val':
            x = self.Model.data.x_val[n]
        elif data == 'test':
            x = self.Model.test_data.x_val[n]
            
        mha_mtx, q_sum, k_sum, y_pred, coords = self.analyse_attention(model,
                                                                       n,
                                                                       x,
                                                                       k,
                                                                       eps_power)


        x_mut = self.mutate(x, **kwargs)

        mha_mtx_mut, q_sum_mut, k_sum_mut, y_pred_mut, _ = self.analyse_attention(model,
                                                                                  n,
                                                                                  x_mut,
                                                                                  k,
                                                                                  eps_power)

        epsilon = 10**(-eps_power)
        diff_mtx = np.log10(mha_mtx_mut+epsilon) - np.log10(mha_mtx+epsilon)
        plot_attention_analysis(diff_mtx,
                                q_sum_mut - q_sum,
                                k_sum_mut - k_sum,
                                coords,
                                y_pred,
                                y_pred_mut,
                                log = False)
        return q_sum, q_sum_mut, coords


    def analyse_attention(self, model, n, x=None, k=10, eps_power=10):
        att = tf.keras.Sequential(lambda: [model.trunk._layers[0],
                    model.trunk._layers[1],
                    model.trunk._layers[2]._layers[0]._layers[0],
                    model.trunk._layers[2]._layers[k]._layers[0]._module._layers[0],
                    model.trunk._layers[2]._layers[k]._layers[0]._module._layers[1]],
                    name='att')
        if x is None:
            x = self.Model.data.x_val[n]
        mha_mtx = att(x, False)[1].numpy()[0]
        mha_mtx = np.sum(mha_mtx, axis=0)

        clip = mha_mtx.shape[0] // 4 if self.Model.data.expand_dna else 0
        bin_size = len(x[0]) // mha_mtx.shape[0]
        coords = np.arange(mha_mtx.shape[0]) * bin_size
        
        if self.Model.data.expand_dna:
            mha_mtx = mha_mtx[clip : -clip, clip : -clip]
            coords = coords[clip : -clip]

        y_true = get_2d(self.Model.y_val[n])
        y_pred = self.Model.dec.predict(model(x, False))[0,:,:,0]
        q_sum = mha_mtx.sum(axis=0)
        k_sum = mha_mtx.sum(axis=1)
        plot_attention_analysis(mha_mtx, q_sum, k_sum, coords, y_true, y_pred,
                                eps_power=eps_power)
        return mha_mtx, q_sum, k_sum, y_pred, coords

    def mutation_effect(self,
                        n,
                        positions,
                        mutations,
                        name,
                        lengths=None,
                        repeats=1,
                        spacer_length=0,
                        p=None,
                        anchor='center'):
        x = self.Model.x_val[n]
        x_mut, boxes = self.mutate(x,
                                   positions=positions,
                                   mutations=mutations,
                                   lengths=lengths,
                                   repeats=repeats,
                                   spacer_length=spacer_length,
                                   p=p,
                                   anchor=anchor)
        y1 = self.Model.predict(x)
        y2 = self.Model.predict(x_mut)
        _, axs = plt.subplots(6,1, figsize=(6,10), 
                            gridspec_kw={'height_ratios': [2, 2, 2, 2, 0.6, 2]})
        axs[4].axis('off')
        self.Model.data.plot_annotated_map(ax=axs[0],
                                y=self.Model.data.y_val[n],
                                sample='val',
                                number=n,
                                axis='both',
                                colorbar=True,
                                x_position='top',
                                name='Raw map',
                                show=False)
        self.Model.data.plot_annotated_map(ax=axs[1],
                                y=self.Model.y_val[n],
                                sample='val',
                                number=n,
                                axis='y',
                                show_position=False,
                                colorbar=True,
                                x_position='top',
                                name='AE output',
                                show=False)
        self.Model.data.plot_annotated_map(ax=axs[2],
                                y=y1,
                                sample='val',
                                number=n,
                                axis='y',
                                show_position=False,
                                colorbar=True,
                                x_position='top',
                                name='WT prediction',
                                show=False)
        self.Model.data.plot_annotated_map(y=y2,
                                ax=axs[3],
                                sample='val',
                                number=n,
                                axis='y',
                                colorbar=True,
                                show_position=False,
                                mutations=boxes,
                                mutation_names=[name]*len(boxes),
                                name='Mut prediction',
                                show=False)
        self.Model.data.plot_annotated_map(ax=axs[5],
                                y=y2-y1,
                                sample='val',
                                number=n,
                                axis='y',
                                show_position=False,
                                colorbar=True,
                                name='Difference',
                                show=True,
                                vmin=-0.3,
                                vmax=0.3)
        
    def mean_effect(self,
                    mutations,
                    name=None,
                    lengths=None,
                    repeats=1,
                    spacer_length=0,
                    p=None,
                    anchor='center'):
        ys_wt = self.Model.predict(self.x_val_good[:])
        sample_mut = [self.mutate(x,
                                  positions=self.Model.data.dna_len//2,
                                  mutations=mutations,
                                  lengths=lengths,
                                  spacer_length=spacer_length,
                                  repeats=repeats,
                                  p=p,
                                  anchor='center')[0][0][0] for x in self.x_val_good]
        sample_mut = np.concatenate(sample_mut)
        ys_mut = self.Model.predict(sample_mut)
        d = ys_mut - ys_wt
        if name:
            plot_map(np.mean(d, axis=0), name=f'Mean {name} insertion effect', colorbar=True)
        else:
            plot_map(np.mean(d, axis=0), colorbar=True)

    def bernulli_seq(self, length, p):
        mutation = np.random.choice(4, length, p=p)
        l = []
        for pos in mutation:
            vec = np.zeros(4)
            vec[pos] = 1
            l.append(vec)
        return np.array(l)


    def calc_position(self, pos, length, anchor):
        if anchor == 'left':
            start = pos
            end = pos + length
        elif anchor == 'center':
            start = pos - length // 2
            end = pos + length - length // 2
        elif anchor == 'right':
            start = pos - length
            end = pos
        return start, end
    

    def mutate(self, x, positions, mutations=None, lengths=None, spacer_length=0, p=None, anchor='center', repeats=1):
        # Unify inputs to make a list of mutations (even if it contains only one 
        # object) and corresponding lengths and positions
        if p is None:
            p = self.background_p
        if isinstance(positions, int):
            positions = [positions]
        if isinstance(lengths, int) or lengths is None:
            lengths = [lengths]*len(positions)
        if mutations is None:
            mutations = []
            boxes = []
            for pos, l in zip(positions, lengths):
                if anchor == 'center':
                    pos = max(pos - l // 2, 0)
                elif anchor == 'right':
                    pos = max(pos - l, 0)
                boxes.append([pos, pos + l])
                mutations.append(Mutation(p=p, length=l))
            x = MutSeqGen(x, [mutations], [positions], anchor=anchor)
            return x, np.array(boxes)

        if not isinstance(mutations, list):
            if isinstance(mutations, np.ndarray):
                if mutations.ndim == 3:
                    mutations = list(mutations)
                else:
                    mutations = [mutations]*len(positions)
            else:
                mutations = [mutations]*len(positions)
        # Make inputs Mutation() objects
        mutations_processed = []
        lengths_processed = []
        if spacer_length > 0:
            spacer = Mutation(p=p, length=spacer_length)
        for m, l in zip(mutations, lengths):
            mutation = Mutation(mutation=m)
            if l is None:
                if repeats < 2:
                    mutations_processed.append(mutation)
                    lengths_processed.append(len(m))
                else:
                    cassette = []
                    for k in range(repeats - 1):
                        cassette.append(mutation)
                        if spacer_length > 0:
                            cassette.append(spacer)
                    cassette.append(mutation)
                    mutations_processed.append(Combination(cassette))
                    lengths_processed.append(len(m) * repeats + spacer_length * (repeats - 1))
                    

            elif len(m) >= l:
                mutations_processed.append(mutation)
                lengths_processed.append(len(m))
            elif len(m) < l:
                n = l // (len(m) + spacer_length)
                if spacer_length > 0:
                    lst = [mutation, spacer] * n
                    lst.pop(-1)
                    mutations_processed.append(Combination(lst))
                    lengths_processed.append(len(m) * n + spacer_length * (n - 1))
                else:
                    mutations_processed.append(Combination([mutation] * n))
                    lengths_processed.append(len(m) * n)
        x = MutSeqGen(x, [mutations_processed], [positions], anchor=anchor)
        boxes = []
        for pos, l in zip(positions, lengths_processed):
            if anchor == 'center':
                pos = max(pos - l // 2, 0)
            elif anchor == 'right':
                pos = max(pos - l, 0)
            boxes.append([pos, pos + l])
        return x, np.array(boxes)


    def analyse_single_fragment(self,
                                number,
                                positions,
                                sample='val',
                                mutations=None,
                                lengths=None,
                                repeats=1,
                                p=None,
                                spacer_length=10,
                                control=None,
                                site_replics=16,
                                control_replics=16,
                                anchor='center'):
        if sample == 'train':
            x = self.Model.data.x_train[number]
            y_raw = self.Model.data.y_train[number]
            y_true = self.Model.y_train[number]
        elif sample == 'val':
            x = self.Model.data.x_val[number]
            y_raw = self.Model.data.y_val[number]
            y_true = self.Model.y_val[number]
        elif sample == 'test':
            x = self.Model.test_data.x_val[number]
            y_raw = self.Model.test_data.y_val[number]
            y_true = self.Model.y_test[number]

        x, boxes = self.mutate(x,
                               positions=positions,
                               mutations=mutations,
                               lengths=lengths,
                               repeats=repeats,
                               p=p,
                               spacer_length=spacer_length,
                               anchor=anchor)
        if control:
            pass
        return self.Model.predict(x, verbose=0), boxes

    def point_on_map(self, position, length, anchor='left'):
        offset = self.Model.data.offset
        resolution = self.Model.data.resolution
        if anchor=='left':
            shift = 0
        elif anchor == 'center':
            shift = length // 2
        elif anchor == 'right':
            shift = length
        left = int((position - offset - shift) / resolution)
        center = int((position - offset + length // 2 - shift) / resolution)
        right = int((position - offset + length - shift) / resolution)
        return left, center, right

    def plot_boxes(self, positions, lens, anchor, h):
        for i,j in zip(positions, lens):
            a,_,b = self.point_on_map(position=i, length=j, anchor=anchor)
            plt.plot([a, a, b, b, a],[1, h-1, h-1, 1, 1], c='r', linewidth=1.5)
        plt.axis('off')

    def analize_insulator(self,
                        motif,
                        p=None,
                        spacer_length=10,
                        fixed_length=False,
                        site_replics=4,
                        control_replics=6,
                        ns=2**np.arange(6),
                        names=[],
                        anchor='center',
                        control='permute',
                        n_loci_per_seq=2,
                        random_state=None,
                        plot=False):
        if not hasattr(self, 'x_val_good'):
            raise AttributeError('Subsample for mutagenesis is not defined. \
Create new Mutagenesis object with select_samples=True')
        if p is None:
            p = self.background_p
        if random_state is not None:
            np.random.seed(random_state)

        if isinstance(control, list) or isinstance(control, tuple):
            controls = control
        else:
            conrtols = [control]

        if isinstance(names, list) or isinstance(names, tuple):
            names = names
        else:
            names = [names]

        site_sample = []
        control_samples = [[] for i in controls]
        l_site = []
        l_control = [[] for i in controls]

        if fixed_length:
            target_length = ns[-1] * (spacer_length + len(motif)) - spacer_length

        site = Mutation(mutation=motif)
        spacer = Mutation(length=spacer_length, p=self.background_p)
        
        for n in ns:
            if n == 0:
                site_sample += [None] * site_replics
                for i in range(len(control_samples)):
                    control_samples[i] += [None] * control_replics
                continue
            locus_with_correct_site = [site, spacer] * n
            locus_with_correct_site.pop(-1)
            if fixed_length:
                left_margin = (target_length - len(locus_with_correct_site)) // 2
                right_margin = target_length - len(locus_with_correct_site) - left_margin
                left_spacer = Mutation(length=left_margin, p=p)
                right_spacer = Mutation(length=right_margin, p=p)
                locus_with_correct_site = [left_spacer] + locus_with_correct_site + [right_spacer]
            locus_with_correct_site = Combination(locus_with_correct_site)
            l_site += [[(len(site)+spacer_length)*i for i in range(n)]] * site_replics ###
            #locus_with_correct_site = [site] * n ###
            
            site_sample += [locus_with_correct_site] * site_replics
            if names:
                name = names[0]
            else:
                if isinstance(motif, str):
                    name = motif
                else:
                    motif = 'motif'
            control_names = []
            for k,control in enumerate(controls):
                for replic in range(control_replics):
                    # same but we permute site between replics but not in one replic
                    if control == 'permute' or control == 'shuffle' or control == 'permuted':
                        control_name = f'Randomly shuffled {name}'
                        control_site = Mutation(mutation=motif, permute_once=True)
                    elif control == 'revcomp':
                        control_name = f'Reverse-complement {name}'
                        control_site = Mutation(mutation=motif, revcomp=True)
                    elif control == 'rev':
                        control_name = f'Reverse {name}'
                        control_site = Mutation(mutation=motif, rev=True)
                    elif control == 'comp':
                        control_name = f'Complement {name}'
                        control_site = Mutation(mutation=motif, comp=True)
                    elif control == 'randomly_revcomp':
                        control_name = f'Mixed forward and reverse-complement {name}'
                        control_site = Mutation(mutation=motif, revcomp_each_time=True)
                    elif control == 'random':
                        control_name = f'Random {len(site)}-nucleotides'
                        control_site = Mutation(length=len(motif), p=p)
                    else:
                        if len(names) > k+1:
                            control_name = names[k+1]
                        else:
                            if isinstance(control, str):
                                control_name = control
                            else:
                                control_name = 'Control'
                        control_site = Mutation(mutation=control)

                    locus_with_control_site = [control_site, spacer] * n
                    locus_with_control_site.pop(-1)
                    if fixed_length:
                        locus_with_control_site = [left_spacer] + locus_with_control_site + [right_spacer]
                    locus_with_control_site = Combination(locus_with_control_site)
                    #locus_with_control_site = [control_site] * n ###
                    l_control[k].append([(len(control)+spacer_length)*i for i in range(n)]) ###
                    control_samples[k].append(locus_with_control_site)
                control_names.append(control_name)
        
        scores_site = []
        scores_control = [[] for i in controls]
        for x in tqdm(self.x_val_good):
            l = x.shape[1]
            loci = np.random.randint(l//2 - l//8, l//2 + l//8, size=n_loci_per_seq)
            y_wt = self.Model.predict(x)
            for position in loci:
                _, center, _ = self.point_on_map(position, 1, anchor=anchor)
                score_wt = y_wt[0,:-4, center-1:center+1,0].mean()
                scores = []
                for le, sample in zip([l_site, *l_control], [site_sample, *control_samples]):
                    positions = [position] * len(sample)
                    #positions = [[i+position for i in j] for j in le] ###
                    x_gen = MutSeqGen(x, sample, positions, anchor, self.Model.batch_size)
                    y = self.Model.predict(x_gen)
                    y = y.reshape(len(ns), -1, self.Model.data.h, self.Model.data.map_size, 1)

                    score = np.zeros(y.shape[:2])
                    for i in range(y.shape[0]):
                        for j in  range(y.shape[1]):
                            #y[i,j] = gaussian_filter(y[i,j], 0.3)

                            '''if fixed_length:
                                length = ns[-1] * (spacer + len(site)) - spacer
                            else:
                                length = max(ns[j] * (spacer + len(site)) - spacer, 0)'''

                            
                            if center - 10 < 0 or center + 10 > self.Model.data.map_size:
                                raise ValueError('Position should not be at the margin of the map. Select something located close to its center')

                            a = y[i,j,:-4, center-1:center+1,0].mean()

                            '''b = y[i,j,-6:-3, center-9:center-6,0].mean()
                            c = y[i,j,-6:-3, center+6:center+9,0].mean()'''

                            score[i,j] = a - score_wt#* 2 / (b + c)
                    scores.append(score.T)
                scores_site.append(scores[0])
                for i in range(len(scores_control)):
                    scores_control[i].append(scores[i+1])
        scores_site = np.concatenate(scores_site)
        for i in range(len(scores_control)):
            scores_control[i] = np.concatenate(scores_control[i])
        plot_motiff_effect(ns, [scores_site] + scores_control,
                           [name] + control_names)
        return scores_site, scores_control
