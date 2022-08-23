from scipy.ndimage import gaussian_filter1d
from skimage.measure import block_reduce
import os
import gc

import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence
from scipy.ndimage import rotate, zoom, gaussian_filter
from scipy import interpolate
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

from .plot import *
from .dataset import *
from .train import *


class IntegratedGradients():
    def __init__(self, model, steps=50):
        self.steps = steps
        self.Model = model

    def interpolate(self, x):
        alphas = tf.linspace(start=0.0, stop=1.0, num=self.steps+1)[:, tf.newaxis, tf.newaxis, tf.newaxis]
        x = tf.expand_dims(x, axis=0)
        return alphas * x

    def compute_gradients(self, inputs):
        grads = []
        for input in inputs:
            with tf.GradientTape() as tape:
                tape.watch(input)
                output = self.Model.model(input)            
                grads.append(tape.gradient(output, input))
        return grads

    def integral_approximation(self, gradients):
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    def integrated_gradients(self, n):
        x = self.Model.data.x_val[n]
        y = self.Model.y_val[n]
        y_pred = self.Model.predict(x)

        interpolated_inputs = self.interpolate(x)
        path_gradients = self.compute_gradients(interpolated_inputs)
        ig = self.integral_approximation(path_gradients).numpy()
        ig = np.sum(ig[0], axis=1)
        self.plot_ig(ig, y, y_pred)
        return ig

    def plot_ig(self,
                ig,
                y_true,
                y_pred,
                point=None,
                start=None,
                stop=None):

        fig = plt.figure(figsize=(16,8))

        gs1 = fig.add_gridspec(nrows=1000, ncols=1, left=0.01, right=0.99, top=1, bottom=0)
        ax1 = fig.add_subplot(gs1[:500])
        ax3 = fig.add_subplot(gs1[500:])
        ax2 = fig.add_subplot(gs1[300:700])
        
        if self.Model.data.expand_dna:
            ig = ig[len(ig)//4:-len(ig)//4]


        ax1.imshow(get_2d(y_true), aspect='auto', cmap="hic_cmap")
        ax1.axis('off')

        ax3.imshow(np.flip(get_2d(y_pred), axis=0), aspect='auto', cmap="hic_cmap")
        ax3.axis('off')

        ax2.plot(ig, c='white', alpha=0.6)
        ax2.plot(gaussian_filter1d(ig, 2), c='white', alpha=0.8)
        ax2.plot(gaussian_filter1d(ig, 10), c='white', alpha=1)
        ax2.set_xlim(0,len(ig))
        marginal = max(np.abs(ig))
        ax2.set_ylim(-marginal, marginal)
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

    def clusterize_peaks(self, ig):
        pass

    def get_seqs(self, ig):
        pass

#IG = IntegratedGradients(Model, steps=50)
