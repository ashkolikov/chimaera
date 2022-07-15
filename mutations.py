import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from gimmemotifs.comparison import MotifComparer
from gimmemotifs.motif import motif_from_align
from gimmemotifs.motif import read_motifs


def plot_motiff_effect(x, correct, permuted, name='Motif', control='Control'):
    sample = [name] * np.prod(correct.shape) + [control] * np.prod(permuted.shape)
    x = np.concatenate([np.tile(x, correct.shape[0]),  np.tile(x, permuted.shape[0])])
    y = np.concatenate([correct.flatten(), permuted.flatten()])
    df = pd.DataFrame({'Number of sites': x,
                       'Mut contacts - wt contacts': y,
                       'Sample': sample})
    
    plt.figure(figsize=(15,8))
    plt.rcParams.update({'font.size': 16})
    sns.violinplot(x='Number of sites', y="Mut contacts - wt contacts", data=df,
                   hue='Sample', palette={name:'#55a0f0', control:'#8b288d'})
    plt.legend(loc='lower left')
    #plt.ylim(0,2)
    plt.show()
    plt.rcParams.update({'font.size': 9})
    


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
        alphabet = {'a' : 0, 'c' : 1, 'g' : 2, 't' : 3}
        encoded_seq = np.array([[alphabet[i]] for i in list(seq)])
        return self.onehot.transform(encoded_seq)
      
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
        val_scores,_ = self.Model.score(metric='pearson', plot=False)
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

    def seq_to_mut(self, mutation):
        mutation = mutation.lower()
        l = []
        for nucleotide in mutation:
            vec = np.zeros(4)
            if nucleotide != 'n':
                pos = ['a','c','g','t'].index(nucleotide)
                vec[pos] = 1
            l.append(vec)
        return np.array(l)

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
    

    def mutate(self, x, positions, mutations=None, lengths=None, spacer_length=0, p=None, anchor='center'):
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
                mutations_processed.append(mutation)
                lengths_processed.append(len(m))
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

        site_sample = []
        control_sample = []
        l_site = []
        l_control = []

        if fixed_length:
            target_length = ns[-1] * (spacer_length + len(motif)) - spacer_length

        site = Mutation(mutation=motif)
        spacer = Mutation(length=spacer_length, p=self.background_p)
        
        for n in ns:
            if n == 0:
                site_sample += [None] * site_replics
                control_sample += [None] * control_replics
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
            if hasattr(control, 'name'):
                name = motif.name
            else:
                if isinstance(motif, str):
                    name = motif
                else:
                    motif = 'motif'
            for replic in range(control_replics):
                # same but we permute site between replics but not in one replic
                if control == 'permute' or control == 'shuffle':
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
                    control_name = f'Random sequence of the same length'
                    control_site = Mutation(length=len(motif), p=p)
                else:
                    if hasattr(control, 'name'):
                        control_name = control.name
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
                l_control.append([(len(control)+spacer_length)*i for i in range(n)]) ###
                control_sample.append(locus_with_control_site)
        
        scores_site = []
        scores_control = []
        for x in tqdm(self.x_val_good):
            l = x.shape[1]
            loci = np.random.randint(l//2 - l//8, l//2 + l//8, size=n_loci_per_seq)
            y_wt = self.Model.predict(x)
            for position in loci:
                _, center, _ = self.point_on_map(position, 1, anchor=anchor)
                score_wt = y_wt[0,-6:-3, center-1:center+1,0].mean()
                scores = []
                for le, sample in zip([l_site, l_control], [site_sample, control_sample]):
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

                            a = y[i,j,-6:-3, center-1:center+1,0].mean()

                            '''b = y[i,j,-6:-3, center-9:center-6,0].mean()
                            c = y[i,j,-6:-3, center+6:center+9,0].mean()'''

                            score[i,j] = a - score_wt#* 2 / (b + c)
                    scores.append(score.T)
                scores_site.append(scores[0])
                scores_control.append(scores[1])
        scores_site = np.concatenate(scores_site)
        scores_control = np.concatenate(scores_control)
        plot_motiff_effect(ns, scores_site, scores_control,
                           name=name.capitalize(),
                           control=control_name.capitalize())
        return scores_site, scores_control

class GeneticSearch():
    def __init__(self,
                 Model,
                 number,
                 sample='val',
                 critical_corr=0.7,
                 population_size=500,
                 initial_coverage=0.15,
                 n_mutations=1,
                 n_crossovers=10, 
                 factor=1.02):
        self.population_size = population_size
        self.factor = factor
        self.critical_corr = critical_corr
        self.initial_coverage = initial_coverage
        self.Model = Model
        g = c = self.Model.data.gc_content / 2 # imperfect but close to real
        a = t = (1 - self.Model.data.gc_content) / 2
        self.background_p = [a, c, g, t]
        self.data = Model.data
        self.sample = sample
        self.number = number
        if sample == 'val':
            self.x = data.x_val[number]
        elif sample == 'train':
            self.x = data.x_train[number]
        self.seq_len = self.x.shape[1]
        self.n_mutations = n_mutations
        self.n_crossovers = n_crossovers
        self.population = self.initialize()
        self.y_wt = self.Model.predict(self.x).flat
        self.corr_score = []
        self.len_score = []
        self.history = [self.population]

    def initialize(self):
        l = int(self.initial_coverage * self.seq_len)
        starts = np.random.randint(0, self.seq_len - l, self.population_size)
        return list(np.stack([starts, starts + l]).T.reshape(-1,1,2))
    
    def calculate_lens(self, population):
        return [(i[:, 1] - i[:, 0]) for i in population]

    def interbreed(self, population):
        np.random.shuffle(population)
        recombinant_population = []
        max_mut_len = np.concatenate(self.calculate_lens(population)).mean()
        for i in range(0, len(population), 2):
            couple = population[i], population[i+1]
            chiasmata = np.random.randint(0, self.seq_len, self.n_crossovers)
            recombinant_couple = self.crossover_and_mutate(*couple, chiasmata, max_mut_len)
            recombinant_population.extend(recombinant_couple)
            if not i % 50:
                pass#gc.collect()
        return np.array(recombinant_population)

    def crossover_and_mutate(self, a, b, chiasmata, max_mut_len):
        if len(chiasmata) % 2:
            chiasmata = np.append(chiasmata, self.seq_len)
        # unzip arrays of starts and ends:
        q = np.full(self.seq_len, False, dtype=bool)
        w = np.full(self.seq_len, False, dtype=bool)

        for index, arr in [(a, q), (b, w)]:
            for i,j in index:
                arr[i:j] = True
        # crossover:
        for i in range(0, len(chiasmata), 2):
            c1, c2 = chiasmata[i], chiasmata[i+1]
            q[c1:c2], w[c1:c2] = w[c1:c2].copy(), q[c1:c2].copy()
        
        # add mutations to random loci:
        if max_mut_len > 0 and self.n_mutations:
            positions = np.random.randint(0, self.seq_len - max_mut_len, self.n_mutations)
            lens = np.random.randint(0, max_mut_len, self.n_mutations)
            for pos, l in zip(positions, lens):
                q[pos : pos + l] = np.logical_not(q[pos : pos + l])
            positions = np.random.randint(0, self.seq_len - max_mut_len, self.n_mutations)
            lens = np.random.randint(0, max_mut_len, self.n_mutations)
            for pos, l in zip(positions, lens):
                w[pos : pos + l] = np.logical_not(w[pos : pos + l])

        # zip back to arrays of starts and ends:
        a = np.where(q[1:]!=q[:-1])[0]+1
        b = np.where(w[1:]!=w[:-1])[0]+1

        
        if q[0]:
            a = np.insert(a, 0, 0)
        if q[-1]:
            a = np.append(a, self.seq_len)
        if w[0]:
            b = np.insert(b, 0, 0)
        if w[-1]:
            b = np.append(b, self.seq_len)
        del q
        del w
        a = a.reshape(-1, 2)
        b = b.reshape(-1, 2)

        # add deletions
        if len(a) > 0 and len(b) > 0:
            a = np.delete(a, np.random.randint(len(a)), axis=0)
            b = np.delete(b, np.random.randint(len(b)), axis=0)
        
        return a, b

    def predict(self, population):
        all_positions = []
        all_mutations = []
        for i in population:
            lengths = list(i[:, 1] - i[:, 0])
            mutations = Mutation(length=lengths, p=self.background_p)
            all_positions.append(i[:, 0])
            all_mutations.append(mutations)
        gen = MutSeqGen(self.x, mutations=all_mutations, positions=all_positions, anchor='left', batch_size=self.Model.batch_size)
        return self.Model.predict(gen)

    def fitness(self, r):
        return 1 - 1 / (1 + np.exp((-r + self.critical_corr) * 100)) # 1 - sigmoid

    def len_fitness(self, lens):
        scores = np.array([(i ** 2).sum() for i in lens])
        return (scores.mean() / scores) ** 3

    def r(self, predictions):
        r = [np.corrcoef(y.flat, self.y_wt)[0,1] for y in predictions]
        r = np.array(r)
        return r

    def survive(self, population, fitness):
        survived = np.random.random(len(fitness)) < fitness
        return population[survived]

    def plot_progress(self, population, r, generations):
        clear_output(wait=True)
        self.corr_score.append(r.mean())
        self.len_score.append(np.mean([i.sum() for i in self.calculate_lens(population)]))

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        x = np.arange(len(self.corr_score), dtype=int) + 1
        ax[0].plot(x, self.corr_score, color='red', marker='o')
        ax[1].plot(x, self.len_score, color='blue', marker='o')
        ax[0].legend(['Mean Pearson correlation with wt'], loc='lower left')
        ax[1].legend(['Mean total length of knockouts'], loc='lower left')
        ax[0].set_xlim(0, generations + 1)
        ax[0].set_xticks(np.arange(1, generations + 1))
        ax[1].set_xlim(0, generations + 1)
        ax[1].set_xticks(np.arange(1, generations + 1))
        ax[0].set_ylim(-.1, 1.1)
        ax[1].set_yscale('log')
        ax[1].set_ylim(10, self.x.shape[1])
        plt.show()

    def replicate(self, population):
        lens = self.calculate_lens(population)
        scores = self.len_fitness(lens)
        nan_mask = np.isnan(scores)
        scores[nan_mask] = 0 # cases with no mutation are useless
        p = scores / scores.sum() 
        offspring = np.random.choice(population, size=self.population_size, p=p)
        return offspring

    def evolution(self, generations=10):
        for generation in range(generations):
            recombinant_population = self.interbreed(self.population)
            predictions = self.predict(recombinant_population)
            r = self.r(predictions)
            self.plot_progress(recombinant_population, r, generations)
            self.history.append(recombinant_population)

            fitness = self.fitness(r)
            survived = self.survive(recombinant_population, fitness)
            try:
                self.population = self.replicate(survived)
            except:
                print('Population became degenerated, evolution stopped')
                return self.population
            self.critical_corr *= self.factor
        return self.population
    
    def _get_seqs(self, threshold=0.5, generation=-1, selected=None):
        x = np.zeros(self.seq_len)
        for i in self.history[generation]:
            for j in i:
                x[j[0]:j[1]] +=1

        xmask = x > x.max() * threshold
        a = np.where(xmask[1:]!=xmask[:-1])[0]
        a = a.reshape(-1,2)
        if selected:
            a = a[selected]
        lens = a[:,1] - a[:,0]
        positions = a[:,0]
        return x, lens, positions, a
    
    def plot_results(self, generation=-1, threshold=0.5, selected=None):
        Mut = Mutagenesis(self.Model)
        plt.rcParams.update({'font.size': 9})
        f, axs = plt.subplots(4,1, figsize=(6,7.5),
                              gridspec_kw={'height_ratios': [2, 1, 2, 2.5]})
        x, lens, positions, boxes = self._get_seqs(threshold, generation, selected)
        y1 = self.Model.predict(self.x, verbose=0)
        x2, _ = Mut.mutate(x=self.x, positions=positions, lengths=lens, anchor='left')
        y2 = self.Model.predict(x2, verbose=0)
        r = np.corrcoef(y1.flat, y2.flat)[0,1]
        offset = self.data.offset
        axs[0].plot(x[offset:-offset])
        axs[0].set_xlim(0, len(x)//2)
        axs[1].axis('off')
        axs[0].set_xticks([])
        axs[0].axis('off')
        self.data.plot_annotated_map(ax=axs[2],
                                    y=y1,
                                    sample=self.sample,
                                    number=self.number,
                                    axis='both',
                                     colorbar=False,
                                    x_position='top', 
                                    full_name=True,
                                    show=False)
        txt = plt.text(5,7,f'r = {r:.3}', fontsize=20, color='white')
        txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='black')])
        self.data.plot_annotated_map(ax=axs[3],
                                    y=y2,
                                    sample=self.sample,
                                    number=self.number,
                                    axis='y',
                                    colorbar=False,
                                    show_position=False,
                                    mutations=boxes,
                                    mutation_names=['Δ']*len(boxes),
                                    vmin=y1.min(),
                                    vmax=y1.max(),
                                    show=True)
    
    def get_seqs(self, generation=-1, threshold=0.5, selected=None, string=True):
        _, _, _, boxes = self._get_seqs(threshold, generation, selected)
        seqs = []
        for i,j in boxes:
            seqs.append(self.x[0, i : j])
        if string:
            return [''.join(['acgt'[i] for i in np.argmax(s, axis=1)]) for s in seqs]
        else:
            return seqs

