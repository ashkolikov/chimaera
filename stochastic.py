import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from gimmemotifs.comparison import MotifComparer
from gimmemotifs.motif import motif_from_align
from gimmemotifs.motif import read_motifs

from .mutations import *


class GeneticSearch():
    def __init__(self,
                 Model,
                 x_index,
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
        self.number = x_index
        if sample == 'val':
            self.x = self.data.x_val[x_index]
        elif sample == 'train':
            self.x = self.data.x_train[x_index]
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


class Evolution():
    def __init__(self,
                 Model,
                 alpha=6,
                 repeats=10,
                 spacer_length=3,
                 population_size=500,
                 length=10,
                 mutation_rate=0.05,
                 n_crossovers=2,
                 use_n=True,
                 func='correlation'):
        self.Model = Model
        self.fitness_function(func)
        self.alpha = alpha
        self.repeats = repeats
        if population_size % 2:
            population_size += 1
        self.spacer_length = spacer_length
        self.population_size = population_size
        self.length = length
        self.use_n = use_n
        self.mutation_rate = 0.05
        self.n_crossovers = 3
        self.population = [''.join(['actg'[j] for j in np.random.choice(4, self.length)]) for i in range(self.population_size)]
        #self.scores = []
    
    def recombination(self, population):
        recombinant_population = []
        for i, j in population.reshape(-1,2):
            i, j = list(i), list(j)
            chiasmata = np.random.choice(self.length, size=self.n_crossovers)
            if len(chiasmata) % 2:
                chiasmata = np.append(chiasmata, self.length)
            for n in range(0, len(chiasmata), 2):
                c1, c2 = chiasmata[n], chiasmata[n+1]
                i[c1:c2], j[c1:c2] = j[c1:c2].copy(), i[c1:c2].copy()
            i, j = ''.join(i), ''.join(j)
            recombinant_population += [i,j]
        return np.array(recombinant_population)

    def add_mutations(self, population):
        recombinant_population = []
        for i in range(self.population_size):
            string = population[i]
            for m in range(self.length):
                if np.random.random() < self.mutation_rate:
                    if self.use_n:
                        mut = 'actgn'[np.random.choice(5)]
                    else:
                        mut = 'actg'[np.random.choice(4)]
                    string = "".join((string[:m], mut, string[m+1:]))
            recombinant_population.append(string)
        return np.array(recombinant_population)
            

    def fitness_function(self, func):
        if isinstance(func, str):
            if func == 'correlation':
                self.fitness = lambda x,y: 1 - np.corrcoef(x.flat, y.flat)[0,1]
            elif func == 'insulation':
                self.fitness = lambda x,y: x[:, x.shape[1]//2].mean() - y[:, x.shape[1]//2].mean()
            elif func == 'compactization':
                self.fitness = lambda x,y: y[:, x.shape[1]//2].mean() - x[:, x.shape[1]//2].mean()
        elif callable(func):
            self.fitness = func

    def reproduce(self, ys, y_wt):
        scores = np.array([self.fitness(y, y_wt) for y in ys])
        #self.scores.append(np.mean(scores))
        print(f'Mean fitness function = {scores.mean():.4f}')
        scores = scores ** self.alpha
        p = scores / scores.sum() 
        return np.random.choice(self.population, size=self.population_size, p=p)
    
    def evolution(self, x_index, generations=20):
        x = self.Model.data.x_val[x_index]
        if not self.Model.predict_as_training:
            y_wt = self.Model.predict(x)[0]
        else:
            y_wt = self.Model.predict(self.Model.data.x_val[x_index : x_index + self.Model.batch_size])[0]
        for i in range(generations):
            substitutions = []
            for i in self.population:
                cassette = []
                for k in range(self.repeats - 1):
                    cassette.append(Mutation(i))
                    if self.spacer_length > 0:
                        cassette.append(Mutation(length=self.spacer_length))
                cassette.append(Mutation(i))
                substitutions.append(Combination(cassette))
            ys = self.Model.predict(MutSeqGen(x, substitutions,
                                              [self.Model.data.dna_len//2]*len(substitutions),
                                              anchor='center',
                                              batch_size=Model.batch_size))
            new_population = self.reproduce(ys, y_wt)
            np.random.shuffle(new_population)
            new_population = self.recombination(new_population)
            self.population = self.add_mutations(new_population)

        return self.population
    
    def get_motif(self, plot=True):
        if self.use_n:
            population = []
            for i in self.population:
                population.append(i.replace('n', np.random.choice(list('acgt'))))
        else:
            population = self.population
        motif = motif_from_align([i.upper() for i in population])
        if plot:
            motif.plot_logo()
        return motif

    def show_effect(self, x_index):
        motif = self.get_motif(plot=False).consensus
        x = self.Model.x_val[x_index]
        y_wt = self.Model.predict(x)
        cassette = []
        for k in range(self.repeats - 1):
            cassette.append(Mutation(motif))
            if self.spacer_length > 0:
                cassette.append(Mutation(length=self.spacer_length))
        cassette.append(Mutation(motif))
        substitution = Combination(cassette)
        y = self.Model.predict(MutSeqGen(x,
                                         [substitution],
                                         [self.Model.data.dna_len//2],
                                         anchor='center',
                                         batch_size=Model.batch_size))
        plot_map(y_wt, name='WT')
        plot_map(y, name='With found motif')
        

