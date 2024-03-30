import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from IPython.display import Image
import imageio

from .ism import SeqMatrix, MutGenerator
from .motifs import Motif, align_to_pfm, pfm_to_logo
from . import plot_utils



class Evolution():
    def __init__(self,
                 Model,
                 population_size=100,
                 length=12,
                 plot_realtime=True,
                 progress_dir='Evo search'):
        self.Model = Model
        if population_size % 2:
            population_size += 1
        self.population_size = population_size
        self.length = length
        self.population = self.initialize_population()
        self.history = []
        self.ps = []
        self.progress_dir = progress_dir
        self.plot_realtime = plot_realtime
        if plot_realtime:
            if not os.path.exists(progress_dir):
                os.mkdir(progress_dir)

    def find_good_loci(self, n, target):
        lat = self.Model.hic_to_latent(self.Model.data.y_test[:], verbose=0)
        proj = lat.dot(target)[:,0]
        return np.argsort(proj)[:n] # top n least close to target

    def initialize_population(self):
        population = []
        for i in range(self.population_size):
            seq = ['actg'[j] for j in np.random.choice(4, self.length)]
            population.append(''.join(seq))
        return population

    def recombination(self, population, n_crossovers):
        recombinant_population = []
        for i, j in population.reshape(-1,2):
            i, j = list(i), list(j)
            chiasmata = np.random.choice(self.length, size=n_crossovers)
            if len(chiasmata) % 2:
                chiasmata = np.append(chiasmata, self.length)
            for n in range(0, len(chiasmata), 2):
                c1, c2 = chiasmata[n], chiasmata[n+1]
                i[c1:c2], j[c1:c2] = j[c1:c2].copy(), i[c1:c2].copy()
            i, j = ''.join(i), ''.join(j)
            recombinant_population += [i,j]
        return np.array(recombinant_population)

    def add_mutations(self, population, mutation_rate):
        recombinant_population = []
        for string in population:
            for m in range(self.length):
                if np.random.random() < mutation_rate:
                    mut = 'actg'[np.random.choice(4)]
                    string = "".join((string[:m], mut, string[m+1:]))
            recombinant_population.append(string)
        return np.array(recombinant_population)

    def reproduce(self, ys, target, population, alpha, elite_amount):
        fitness = lambda x, y: np.dot(x, y)#**2 / np.linalg.norm(x)
        scores = np.array([fitness(y, target[:,0]) for y in ys])
        argsort = np.argsort(-scores)
        elite = np.array(population)[argsort[:elite_amount]]
        mean_score = scores.mean()
        '''ranks = argsort.argsort()
        n = len(ranks)
        p = (alpha - 2*ranks*(alpha-1)/(n-1)) / n'''
        scores -= scores.min()
        scores /= scores.max()
        scores = scores ** alpha
        p = scores / scores.sum()
        self.ps.append(scores)
        new_population = np.random.choice(
            population,
            size=self.population_size - elite_amount,
            p=p
            )
        return mean_score, new_population, elite

    def add_shift(self, population, rate):
        recombinant_population = []
        for string in population:
            if np.random.random() < rate:
                if np.random.random() < 0.5:
                    string = string[1:] + 'acgt'[np.random.randint(4)]
                else:
                    string = 'acgt'[np.random.randint(4)] + string[:-1]
            recombinant_population.append(string)
        return np.array(recombinant_population)

    def evolution(
            self,
            target,
            alpha=5,
            composition='>>>>>>>',
            between_insertions=10,
            mutation_rate=0.1,
            n_crossovers=2,
            generations=20,
            experiment_index=0,
            experiment_name=None,
            elite=0.1,
            shift_rate=0.1,
            population_size=200,
            length = 20,
            n_loci = 2,
            plot_realtime=True,
            progress_dir='Evo search'
            ):

        target = target.T / np.linalg.norm(target)
        elite_amount = int(elite*population_size)

        if population_size % 2:
            population_size += 1
        self.population_size = population_size
        self.length = length
        self.population = self.initialize_population()
        self.history = []
        self.progress_dir = progress_dir
        self.plot_realtime = plot_realtime
        sample = self.find_good_loci(n_loci, target)
        if plot_realtime:
            if not os.path.exists(progress_dir):
                os.mkdir(progress_dir)
            else:
                files_in_saving_dir = os.listdir(progress_dir)
                if files_in_saving_dir:
                    for i in files_in_saving_dir:
                        if i.startswith('generation') and i.endswith('logo.png'):
                            os.remove(os.path.join(progress_dir, i))

        x = self.Model.data.x_test[sample]
        if experiment_name:
            experiment_index = self.Model.data.experiment_names.index(experiment_name)
        y_wt = self.Model.dna_to_latent(x, verbose=0)[:, experiment_index]
        for generation in range(generations):
            population = self.population
            pfm = self.get_motif()
            self.history.append(pfm)
            substitutions = [Motif(seq=i) for i in population]
            generator = MutGenerator(
                x,
                substitutions,
                batch_size=self.Model.batch_size,
                between_insertions=between_insertions,
                composition=composition,
                strategy='all_to_all',
                offset=self.Model.data.offset
                )
            ys = self.Model.dna_to_latent(generator, verbose=0)[:, experiment_index]
            ys = ys.reshape((len(sample), len(population), -1))
            ys = (ys - y_wt[:, None, :]).mean(axis=0)
            mean_score, new_population, elite = self.reproduce(ys, target, population, alpha, elite_amount)
            np.random.shuffle(new_population)
            new_population = self.recombination(new_population, n_crossovers)
            new_population = self.add_mutations(new_population, mutation_rate)
            new_population = self.add_shift(new_population, shift_rate)
            self.population = np.concatenate([elite, new_population])
            print(f'Generation {generation+1}: mean fitness = {mean_score:.4f}')

            if self.plot_realtime:
                self.plot_logo()
                filename = 'generation_'+str(generation+1)+'_logo.png'
                filename = os.path.join(self.progress_dir, filename)
                plt.savefig(filename)
                plt.close()

    def plot_evolution_gif(self):
        images = []
        files = os.listdir(self.progress_dir)
        files = [i for i in files if i.startswith('generation_')]
        files.sort(key=lambda x: int(x.split('_')[1]))
        for filename in files:
            filename = os.path.join(self.progress_dir, filename)
            image = imageio.v2.imread(filename)
            images.append(image)
        images += [image] * 15 # for pause in the gif
        gif_name = os.path.join(self.progress_dir, 'motif_evolution.gif')
        imageio.mimsave(gif_name, images, format='GIF', duration=5)
        return Image(gif_name)

    def plot_evolution(self, step=1):
        n_iters = len(self.history)
        figsize = (7, n_iters * 4)
        fig, axs = plt.subplots(n_iters, 1, figsize=figsize)
        for i in range(0, n_iters, step):
            pfm_to_logo(self.history[i].pfm, axs[i])
            axs[i].set_ylabel(f'Generation {i+1}')
        plot_utils.savefig('evolution')
        plt.show()

    def plot_logo(self):
        figsize = (7, 3)
        fig, axs = plt.subplots(1, 1, figsize=figsize)
        pfm_to_logo(self.history[-1].pfm, axs)

    def get_motif(self):
        pfm = align_to_pfm([i.upper() for i in self.population])
        motif = Motif(pfm=pfm, name=f'found motif')
        return motif