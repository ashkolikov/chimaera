import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class IntegratedGradients():
  
  def __inint__(self, model, steps=15):
    self.steps = steps
    self.model = model

  def interpolate(self, x):
      alphas = tf.linspace(start=0.0, stop=1.0, num=self.steps+1)[:, tf.newaxis, tf.newaxis, tf.newaxis]
      x = tf.expand_dims(x, axis=0)
      return alphas * x

  def compute_gradients(self, inputs):
      with tf.GradientTape() as tape:
          tape.watch(inputs)
          outputs = []
          for input in inputs:
              outputs.append(self.model(input))
          outputs = tf.stack(outputs)  
      return tape.gradient(outputs, inputs)

def integral_approximation(self, gradients):
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

def integrated_gradients(self, x):
    interpolated_inputs = self.interpolate(x)
    path_gradients = self.compute_gradients(interpolated_inputs)
    ig = self.integral_approximation(path_gradients).numpy()
    self.ig = np.sum(ig[0], axis=1)
    return self.ig

def plot_ig(self,
            point=None,
            start=None,
            stop=None):
    plt.figure(figsize=(20,10))

def clusterize_peaks(self, ig):
    pass

def get_seqs(self, ig):
    pass
