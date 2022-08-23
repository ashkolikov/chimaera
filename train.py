import os
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from skimage.measure import block_reduce
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d
import json
import gc
from .model import CompositeModel, VAE, MainModel
from .dataset import *
from .plot import *



class DataGenerator(Sequence):
    '''For loading into model while training - saves memory, shuffles batches, 
allows making random sequences reverse complement.'''
    def __init__(self, x, y, batch_size=32, shuffle=True, rev_comp=False):
        data = Model.data
        self.revcomp = rev_comp
        self.X = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X = self.X[indexes]
        y = self.y[indexes]
        X, y = self.conditional_rev_comp(X, y)
        return X, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def conditional_rev_comp(self, X, y):
        if self.revcomp is True:
            X = np.flip(X, axis=(1, 2))
            y = np.flip(y, axis=2)
        elif isinstance(self.revcomp, float):
            ind = np.random.random(len(X)) < self.revcomp
            X[ind] = np.flip(X[ind], axis=(1, 2))
            y[ind] = np.flip(y[ind], axis=2)
        return X, y


class HiCDataGenerator(Sequence):
    '''For loading into autoencoder while training - saves memory, shuffles batches, 
allows flipping random maps.'''
    def __init__(self, hic, rotate, shuffle = True):
        self.X = hic
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
        a[ind] = np.flip(a[ind], axis=1)
        return a

class AETrainer():
    def __init__(self, model, data=None):

        self.ae = model.ae
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.data = model.data if data is None else data
    
    def train(self, epochs, batch_size=64, use_test=True, rotate=False, shuffle=True):
        if self.ae is None:
            raise ValueError("Autoencoder loaded from a file can't be trained using this method")
        if use_test:
            y = np.concatenate([self.data.y_train[:], self.data.y_val[:]])
            datagen = HiCDataGenerator(y, rotate, shuffle)
        else:
            datagen = HiCDataGenerator(self.data.y_train[:], rotate, shuffle)
        self.ae.fit(datagen, epochs=epochs, batch_size=batch_size)
    
    def smooth_interpolation(self):
        point = self.encoder.predict(self.data.y_val[0])
        cloud = self.encoder.predict(self.data.y_val[0:64])        
        self._spheares(point, cloud)
    
    def _spheares(self, point, cloud):
        rs = [0.5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 30]
        _, axs = plt.subplots(2,len(rs),figsize=(25,10))
        for i, r in enumerate(rs):
            np.random.seed(1)
            random_vecs = cloud - point
            random_vecs /= np.linalg.norm(random_vecs, axis=1)[:,None]
            random_vecs *= r
            random_vecs[0] *= 0
            random_vecs += point
            pred_cloud = self.decoder.predict(random_vecs)
            pred_point = pred_cloud[0]
            c = []
            for prediction in pred_cloud[1:]:
                c.append(spearmanr(prediction.flat, pred_point.flat)[0])
            sns.boxplot(y=c, ax=axs[0,i])
            axs[0,i].set_ylim(-1,1)
            plt.setp(axs[0,i].get_xticklabels(), visible=False)
            axs[0,i].tick_params(axis='x', which='both', length=0)
            if i>0: 
                plt.setp(axs[0,i].get_yticklabels(), visible=False)
            axs[0,i].set_title(str(r))
            ind = np.random.randint(64)
            axs[1,i].imshow(pred_cloud[ind][:,:,0].T, cmap="hic_cmap")
            axs[1,i].axis('off')
        plt.show()

    def predict(self, y):
        return self.decoder.predict(self.encoder.predict(y))
    
    def plot_results(self, y = None, equal_scale=False):
        if y is None:
            y = self.data.y_val[:9]
        if isinstance(y, str):
            if y == 'train':
                y = self.data.y_train[:9]
        y_pred = self.predict(y)
        plot_results(y_pred, y, sample='val',
                     numbers=np.arange(0, 9),
                     data=self.data, equal_scale=equal_scale)
    
    def score(self):
        y = self.data.y_val[:]
        datagen = HiCDataGenerator(y, rotate=False)
        y_pred = self.ae.predict(datagen)
        return np.corrcoeff(y.flat, y_pred.flat)[0,1]


class Chimaera():
    """Contains model and functions to work with it"""
    def __init__(self, 
                 data = None,
                 genome_file_or_dir = None,
                 hic_file = None,
                 saving_dir = None, 
                 model_dir = None,
                 rewrite = False,
                 pretrain = False,
                 predict_as_training = False,
                 neural_net = None):

        super(Chimaera, self).__init__()

        self.data = data
        if data is None or isinstance(data, str):
            if data is None:
                data_params_path = os.path.join(model_dir, "data_params.json")
            else:
                data_params_path = data
            with open(data_params_path, "r") as read_file:
                data_params = json.load(read_file)
            if genome_file_or_dir is None or hic_file is None:
                raise ValueError('Indicate pathes to genome and Hi-C maps files')
            data_params['hic_file'] = hic_file
            data_params['genome_file_or_dir'] = genome_file_or_dir
            self.data = Dataset(**data_params)
        self.x_train = self.data.x_train
        self.y_train = self.data.y_train
        self.x_val = self.data.x_val
        self.y_val = self.data.y_val
        self.input_len = self.data.dna_len
        self.saving_dir = saving_dir
        self.predict_as_training = predict_as_training
        self.test_data = None
        self.dec = None
        self.enc = None
        self.ae = None
        self.model = None
        self.batch_size = None
        self.encoded = False
        self.pretrain = pretrain
        
        if saving_dir is None:
            pass
        elif os.path.exists(saving_dir) and not rewrite:
            raise ValueError(f"Type a new directory name for saving model or delete the existing one named {saving_dir}")
        elif os.path.exists(saving_dir) and rewrite:
            pass
        else:
            os.mkdir(saving_dir)

        if model_dir is None:
            if neural_net is None:
                print('No model files or object provided, build model using "build" method')
            else:
                self.build(neural_net)
        else:
            neural_net = CompositeModel(data=data, model_dir=model_dir)
            self.build(neural_net)
            
    def build(self, neural_net):
        self.enc = neural_net.encoder
        self.dec = neural_net.decoder
        self.model = neural_net.model
        self.model_1d = neural_net.model_1d
        for k,v in self.dec._get_trainable_state().items():
            k.trainable = False
        self.ae = neural_net.ae
        self.batch_size = max(1, 2**29//np.prod(self.model.layers[1].output_shape[1:]))
        self.encode_y()
        self.encoded = True
    
    def rebuild(self, model=None):
        if model is None:
            model = MainModel(self.data)
        inp = model.input
        latent_output = main_model.output
        self.attention_outputs = main_model.attention_outputs
        self.latent_model = tf.keras.Model(inp, latent_output)

        final_output = self.dec(latent_output)
        self.model = tf.keras.Model(inp, final_output)
        self.model.compile(optimizer='adam', loss='mse')


    def encode_y(self):       
        y_train = self.y_train[:]
        y_val = self.data.y_val[:]
        self.data.y_latent_train = self.enc.predict(y_train, verbose=0)
        self.data.y_latent_val = self.enc.predict(y_val, verbose=0)
        self.y_train = self.dec.predict(self.data.y_latent_train, verbose=0)
        self.y_val = self.dec.predict(self.data.y_latent_val, verbose=0)
        self.encoded = True

        if self.pretrain:
            y_train_1d = np.array([gaussian_filter1d(a[-3], sigma=1, axis=0) for a in self.y_train])
            y_val_1d = np.array([gaussian_filter1d(a[-3], sigma=1, axis=0) for a in self.y_val])
            min_ = y_train_1d.min()
            max_ = y_train_1d.max() - min_
            self.y_train_1d = (y_train_1d - min_) / max_
            self.y_val_1d = (y_val_1d - min_) / max_


    def save(self, save_main_model = True):
        if self.saving_dir is None:
            print("Model won't be saved while training")
            return
        with open(os.path.join(self.saving_dir, 'data_params.json'), 'w') as file:
            params = self.data.params.copy()
            del params['hic_file']
            del params['genome_file_or_dir']
            file.write(json.dumps(params))
        self.enc.save(os.path.join(self.saving_dir, 'enc.h5'))
        self.dec.save(os.path.join(self.saving_dir, 'dec.h5'))
        if self.model_1d is not None:
            self.model_1d.save(os.path.join(self.saving_dir, 'model_1d.h5'))
        if save_main_model:
            self.model.save(os.path.join(self.saving_dir, 'model.h5'))
            
    def describe(self, text):
        if not self.saving_dir:
            print('''Saving path is not provided. Set 'saving_dir' attribute''')
        else:
            with open(os.path.join(self.saving_dir, 'description.txt'), 'w') as file:
                file.write(text)
    
    def pretrain(self, epochs=50, batch_size=None, rev_comp=0.5):
        if batch_size is None:
            batch_size = self.batch_size
        else:
            self.batch_size = batch_size
        
        train_generator = DataGenerator(self.x_train, self.y_train_1d, 
                                        batch_size=batch_size, rev_comp=rev_comp)
        val_generator = DataGenerator(self.x_val, self.y_val_1d, 
                                      batch_size=batch_size, rev_comp=rev_comp)
        
        return self.model_1d.fit(train_generator, 
                                   validation_data = val_generator,
                                   epochs = epochs)
    def combine(self, n_layers):
        x = tf.keras.layers.Input(shape=(self.data.dna_len, 4))
        for layer in self.pretrained.layers[:n_layers]:
            x = layer(x)
        



    def train(self, epochs=1000, batch_size=None, callbacks='full', rev_comp=0.5):
        self.save(save_main_model = False)            
        if batch_size is None:
            batch_size = self.batch_size
        else:
            self.batch_size = batch_size 

        callbs = []
        if self.saving_dir:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(self.saving_dir, 'model.h5'),
                                                        save_weights_only = False)
            callbs.append(checkpoint)

        if callbacks == 'full':
            img_saving_dir = self.saving_dir if self.saving_dir else '.'
            map_callback = MapDrawingCallback(self, epochs = epochs)
            callbs.append(map_callback)

        train_generator = DataGenerator(self.x_train, self.y_train, 
                                        batch_size=batch_size, rev_comp=rev_comp)
        val_generator = DataGenerator(self.x_val, self.y_val, 
                                      batch_size=batch_size, rev_comp=rev_comp)
        return self.model.fit(train_generator, 
                              validation_data = val_generator,
                              epochs = epochs,
                              callbacks = callbs)

    def predict_in_training_mode(self, batch):
        res = []
        for batch_ in range(0, len(batch), self.batch_size):
            pred = self.model(batch[batch_ : batch_ + self.batch_size], training=True)
            pred = pred.numpy()
            if len(pred) == 1:
                print('Prediction is not correct because model predicted batch with one object - in training mode it works incorrect.')
                print(f'Change sample size ({len(batch)}) or set class attribute predict_as_training=False')
            res.append(pred)
        return np.concatenate(res)

    def _pearson(self, x1, x2):
        return [[np.corrcoef(i[-k-1].flat, j[-k-1].flat)[0,1] for k in range (x1.shape[1])] for i,j in zip(x1,x2)]
    
    def _spearman(self, x1, x2):
        return [[spearmanr(i[-k-1].flat, j[-k-1].flat)[0] for k in range (x1.shape[1])] for i,j in zip(x1,x2)]

    def predict(self, x, verbose=0, strand='one'):
        if isinstance(x, DataGenerator) or isinstance(x, MutSeqGen):
            if self.predict_as_training:
                y = []
                l = len(x)
                for i, batch in enumerate(x):
                    print((i+1)*'.', end='\r')
                    if isinstance(batch, tuple):
                        batch = batch[0]
                    y_ = self.predict_in_training_mode(batch)
                    if strand == 'both':
                        y_rc = np.flip(self.predict_in_training_mode(np.flip(batch, axis=(1,2))), axis=2)
                        y_ = (y_ + y_rc) / 2
                    y.append(y_)
                y = np.concatenate(y)
            else:
                y = self.model.predict(x, verbose=verbose)
                if strand == 'both':
                    x_rc = x.copy()
                    x_rc.revcomp = 1
                    y_rc = np.flip(self.model.predict(x_rc, verbose=verbose), axis=2)
                    y = (y + y_rc) / 2
        else:
            if self.predict_as_training:
                y = self.predict_in_training_mode(x)
                if strand == 'both':
                    x_rc = np.flip(x, axis=(1,2))
                    y_rc = np.flip(self.predict_in_training_mode(x_rc), axis=2)
                    y = (y + y_rc) / 2
            else:
                y = self.model.predict(x, verbose=verbose, batch_size=self.batch_size)
                if strand == 'both':
                    x_rc = np.flip(x, axis=(1,2))
                    y_rc = np.flip(self.model.predict(x_rc, verbose=verbose, batch_size=self.batch_size), axis=2)
                    y = (y + y_rc) / 2
        return y
            
    def predict_and_plot(self, x, strand='one', **kwargs):
        y = self.predict(x, verbose=0, strand=strand)
        plot_map(y, **kwargs)
    
    def get_filters(self, conv_layer):
        conv = [i for i in self.model.layers if 'conv' in i.name]
        filters = conv[conv_layer - 1].get_weights()[0].T
        return filters

    def score(self, metric='pearson', sigma=0, plot=True, strand='both', best_only=True):
        metric_name = metric
        metric = self._pearson if metric_name == 'pearson' else self._spearman
        generator = DataGenerator(self.x_val, self.y_val, 
                                  batch_size=self.batch_size, rev_comp=False, shuffle=False)
        revcomp_generator = DataGenerator(self.x_val, self.y_val, 
                                          batch_size=self.batch_size, rev_comp=True, shuffle=False)    
        y_pred = self.predict(generator, verbose=1)
        groundtruth = self.y_val
        if sigma > 0:
            y_pred = np.array([gaussian_filter(i, sigma) for i in y_pred])
            groundtruth = np.array([gaussian_filter(i, sigma) for i in groundtruth])
        if strand=='both':
            y_pred_revcomp = np.flip(self.predict(revcomp_generator, verbose=1),axis=2)
            if sigma > 0:
                y_pred_revcomp = np.array([gaussian_filter(i, sigma) for i in y_pred_revcomp])
            y_pred = (y_pred + y_pred_revcomp) / 2
            
        r = metric(y_pred, groundtruth)
        r_control = metric(y_pred, np.random.permutation(groundtruth))

        x = np.linspace(0, self.data.max_dist, self.data.h+1)
        if plot:
            plot_score(metric_name, r, r_control, x, best_only)
        else:
            return r, r_control

    def plot_results(self, sample='val', number=0, equal_scale=False, save=False, strand='both'):
        if sample == 'train':
            x_val = self.x_train[number:number+9]
            y_val = self.y_train[number:number+9]
        elif sample == 'val':
            x_val = self.x_val[number:number+9]
            y_val = self.y_val[number:number+9]
        elif sample == 'test':
            x_val = self.x_test[number:number+9]
            y_val = self.y_test[number:number+9]
        y_pred = self.predict(x_val, strand=strand)
        plot_results(y_pred, y_val, sample=sample,
                     numbers=np.arange(number, number+9),
                     data=self.data, equal_scale=equal_scale, save=save)


    def plot_filters(self, figsize = (16, 10), cmap = 'coolwarm', normalize=False):
        filters = self.model.layers[1].get_weights()[0].T
        plot_filters(filters, figsize = figsize, cmap = cmap, normalize=normalize)
    
    def summary(self):
        self.model.summary()

    
    def load_test_chromosome(self, chrom, **kwargs):
        params = self.data.params
        if isinstance(chrom, str):
            chrom = [chrom]
        for i in chrom:
            if i in self.data.names:
                raise ValueError(f"Chromosome {i} was in training sample! It can't be used for testing.")
        params['chroms_to_include'] = chrom
        params['val_split'] = 'test'
        params['min_max'] = self.data.min_max
        for key,value in kwargs.items():
            params[key] = value
        self.test_data = Dataset(**params)
        
        self.x_test = self.test_data.x_val
        self.y_test = self.ae.predict(self.test_data.y_val)

    def filter_analisis(self, num, n_layers=3, color_shifts={'heatmap': 50, 'filters': 200},  aggregation='max', return_filters=False):
        first_layers = tf.keras.models.Sequential(self.model.layers[:n_layers])
        pred = first_layers.predict(self.data.x_val[num])[0].T
        ind, start, _ = self.data._x_val[num]
        start, end = start, start + self.data.dna_len
        if self.data.expand_dna:
            pred = pred[:, pred.shape[1] // 4 : -pred.shape[1] // 4]
            start, end = start + self.data.offset, end - self.data.offset

        aggregation_rate =  pred.shape[1] // 512
        fun = np.max if aggregation == 'max' else np.mean
        pred = block_reduce(pred, (1, aggregation_rate), fun)

        y_true = self.y_val[num]
        if self.predict_as_training:
            y_pred = self.predict(self.data.x_val[num:num+self.batch_size])[0]
        else:
            y_pred = self.predict(self.data.x_val[num:num+1])

        filters=self.model.layers[1].get_weights()[0].T

        best_filters = plot_filter_analisis(pred, 
                                    y_pred, 
                                    y_true, 
                                    filters,
                                    color_shifts)
        if return_filters:
            return best_filters


class MapDrawingCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 Model,
                 epochs):
        super(MapDrawingCallback, self).__init__()
        self.epochs = epochs
        self.Model = Model
        self.x_sample = Model.x_val[:9]
        self.y_sample = Model.y_val[:9]
        self.saving_dir = self.Model.saving_dir
        self.saving_dir = os.path.join(self.saving_dir, 'progress')
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)
        if os.listdir(self.saving_dir):
            self.intit_epoch = int(os.listdir(self.saving_dir)[-1].split('.')[0]) + 1
        else:
            self.intit_epoch = 1

    def on_epoch_end(self, epoch, logs={}):
        file_name = f'{(self.intit_epoch+epoch):03d}.png' if self.epochs < 1000 else  f'{(self.intit_epoch+epoch):04d}.png'
        self.Model.plot_results(save = os.path.join(self.saving_dir, file_name), equal_scale=False)

