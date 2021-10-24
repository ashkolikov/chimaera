import os
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from skimage.measure import block_reduce
from scipy.stats import spearmanr
import json
import gc
from .dataset import *
from .plot import *

class ModelMaster():
    """Contains model and functions to work with it"""
    def __init__(self, 
                 data = None,
                 genome_file_or_dir = None,
                 hic_file = None,
                 saving_dir = None, 
                 model_dir = None, 
                 rewrite = False,
                 predict_as_training = False):

        super(ModelMaster, self).__init__()

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
            self.data = DataMaster(**data_params)
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
        self.ae_epochs = 200
        self.batch_size = 8
        self.encoded = False
        
        if saving_dir is None:
            pass
        elif os.path.exists(saving_dir) and not rewrite:
            raise ValueError(f"Type a new directory name for saving model or delete the existing one named {saving_dir}")
        elif os.path.exists(saving_dir) and rewrite:
            pass
        else:
            os.mkdir(saving_dir)

        if model_dir is None:
            print('No model files provided, build model parts manually using corresponding build_ methods')
        else:
            availible_files = os.listdir(model_dir)
            self.model = tf.keras.models.load_model(os.path.join(model_dir, 'model.h5'))
            self.enc = tf.keras.models.load_model(os.path.join(model_dir, 'enc.h5'))
            self.dec = tf.keras.models.load_model(os.path.join(model_dir, 'dec.h5'))
            self.batch_size = max(1, 2**27//np.prod(self.model.layers[1].output_shape[1:]))
            self.encode_y()
        if self.encoded:
            self.train_generator = DataGenerator(data=self.data, batch_size=self.batch_size, train=True, encoder=self.enc) # encoder is used only if dataset uses stochastic sampling
            self.val_generator = DataGenerator(data=self.data, batch_size=self.batch_size, train=False, encoder=self.enc)
            
    def build(self, neural_net):
        neural_net.update(input_len = self.data.dna_len,
                          map_size = self.data.map_size,
                          output_scale = self.data.scale)
        self.build_enc(neural_net.enc)
        self.build_dec(neural_net.dec)
        self.build_model(neural_net.model)
    
    def build_enc(self, fun, weights=None):
        if fun is not None:
            self.enc = fun()
        if weights is not None and self.enc is not None:
            self.enc.load_weights(weights)
            self.encode_y()

    def build_dec(self, fun, weights=None):
        if fun is not None:
            self.dec = fun()
            self.dec.build(input_shape = self.enc.output_shape)
        if weights is not None and self.dec is not None:
            self.dec.load_weights(weights)
        self.build_ae(train = (weights is None))

    def build_model(self, fun, weights=None):
        if fun is not None:
            self.model = fun()
            assert self.model.input_shape[1] == self.input_len
        if weights is not None:
            self.model.load_weights(weights)
        self.batch_size = max(1, 2**27//np.prod(self.model.layers[1].output_shape[1:]))
            

    def build_ae(self, train):
        if self.enc is not None and self.dec is not None:
            self.ae = tf.keras.models.Sequential([self.enc, self.dec])
            self.ae.build(input_shape = self.enc.input_shape)
            self.ae.compile(loss = 'mse', optimizer = 'adam')
        if train:
            self.train_ae()

    def train_ae(self):
        print('Training autoencoder for Hi-C maps ...')
        self.ae.fit(HiCDataGenerator(self.data), epochs = self.ae_epochs)
        self.encode_y()
        print('Autoencoder is ready')

    def encode_y(self):
        self.data.y_latent_train = self.hic_to_latent(self.data.y_train)
        self.data.y_latent_val = self.hic_to_latent(self.data.y_val)
        self.data.y_latent_train_sample = self.hic_to_latent(self.data.y_train_sample)
        self.data.y_latent_val_sample = self.hic_to_latent(self.data.y_val_sample)
        self.fit_pca()
        self.encoded=True
        print('Hi-C maps successfully encoded')

    def fit_pca(self):
        latent = np.concatenate((self.data.y_latent_train[:500], self.data.y_latent_val))
        pca = PCA(n_components=2)
        pca.fit(latent)
        self._pca = pca
        self.transformed_background = pca.transform(latent).T
        self.pca = lambda x: self._pca.transform(x).T

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
        if save_main_model:
            self.model.save(os.path.join(self.saving_dir, 'model.h5'))
            
    def describe(self, text):
        if not self.saving_dir:
            print('''Saving path is not provided. Set 'saving_dir' attribute''')
        else:
            with open(os.path.join(self.saving_dir, 'description.txt'), 'w') as file:
                file.write(text)

    def train(self, epochs, batch_size = None, callbacks = 'full', show_summary = True):
        self.save(save_main_model = False)            
        if batch_size is None:
            batch_size = self.batch_size
        else:
            self.batch_size = batch_size 

        if show_summary:
            self.model.summary()
        callbs = []
        if self.saving_dir:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(self.saving_dir, 'model.h5'),
                                                        save_weights_only = False)
            callbs.append(checkpoint)

        if callbacks == 'full':
            img_saving_dir = self.saving_dir if self.saving_dir else '.'
            pca_callback = LatentSpaceDrawingCallback(self,
                                                     x_train_sample = self.data.x_train_sample, 
                                                     x_val_sample = self.data.x_val_sample, 
                                                     y_train_sample = self.data.y_latent_train_sample, 
                                                     y_val_sample = self.data.y_latent_val_sample,
                                                     pca = self.pca,
                                                     transformed_background = self.transformed_background, 
                                                     saving_dir = img_saving_dir,
                                                     epochs = epochs)

            map_callback = MapDrawingCallback(self,
                                             x_sample = self.data.x_val_sample, 
                                             y_sample = self.data.y_val_sample,  
                                             saving_dir = img_saving_dir, 
                                             decoder = self.dec,
                                             epochs = epochs)

            callbs += [pca_callback, map_callback]

        self.train_generator = DataGenerator(data=self.data, batch_size=batch_size, train=True, encoder=self.enc) # encoder is used only if dataset uses stochastic sampling
        self.val_generator = DataGenerator(data=self.data, batch_size=batch_size, train=False, encoder=self.enc)
        return self.model.fit(self.train_generator, 
                              validation_data = self.val_generator,
                              epochs = epochs,
                              callbacks = callbs)

    def predict_large_batch(self, model, large_batch, batch_size = 4):
        res = []
        for batch in range(0, len(large_batch), batch_size):
            pred = model(large_batch[batch : batch + batch_size], training=self.predict_as_training)
            gc.collect()
            pred = pred.numpy()
            if len(pred) == 1 and self.predict_as_training:
                print('Prediction is not correct because model predicted batch with one object - in training mode it works incorrect.')
                print(f'Change batch_size ({batch_size}), sample size ({len(large_batch)}) or set class attribute predict_as_training=False')
            res.append(pred)
        return np.concatenate(res)

    def hic_to_latent(self, hic):
        return self.predict_large_batch(self.enc, hic, 128)

    def latent_to_hic(self, latent):
        return self.predict_large_batch(self.dec, latent, 128)

    def dna_to_latent(self, dna, batch_size=4):
        return self.predict_large_batch(self.model, dna, batch_size)

    def dna_to_hic(self, dna, batch_size=4):
        latent = self.dna_to_latent(dna, batch_size)
        return self.latent_to_hic(latent)


    def get_first_layer_weights(self):
        pass

    def get_n_layer_output(self, n):
        pass

    def _pearson(self, x1, x2):
        return [np.corrcoef(i.flat, j.flat)[0,1] for i,j in zip(x1,x2)]
    
    def _spearman(self, x1, x2):
        return [spearmanr(i.flat, j.flat)[0] for i,j in zip(x1,x2)]

    def predict(self, x, prediction = 'classical', verbose=1):
        if isinstance(x, DataGenerator):
            if prediction == 'classical':
                return self.dec.predict(self.model.predict(x, verbose=verbose))
            else:
                y = []
                l = len(x)
                for i, batch in enumerate(x):
                    print((i+1)*'.', end='\r')
                    y.append(self.predict_large_batch(self.dec, self.predict_large_batch(self.model, batch[0], self.batch_size), 64))
                    del batch
                    gc.collect()
                return np.concatenate(y)
        else:
            if prediction == 'classical':
                return self.dec.predict(self.model.predict(x, verbose=verbose))
            else:
                return self.predict_large_batch(self.dec, self.predict_large_batch(self.model, x, self.batch_size), 64)
            
    def predict_and_plot(self, x, hic_cmap='Reds', prediction = 'classical'):
        y = self.predict(x, prediction = prediction, verbose=0)
        plot_map(y, hic_cmap = hic_cmap)
    
    def get_filters(self, conv_layer):
        conv = [i for i in self.model.layers if 'conv' in i.name]
        filters = conv[conv_layer - 1].get_weights()[0].T
        return filters

    def score(self, metric='spearman', prediction='classical', include_train=True, **kwargs):
        metric_name = metric
        metric = self._pearson if metric_name == 'pearson' else self._spearman
        if include_train:
            train_generator = DataGenerator(data=self.data, batch_size=self.batch_size, train=True, shuffle=False)
            y_pred_train = self.predict(train_generator, prediction=prediction)
            r_train = metric(y_pred_train, self.data._y_train)
            r_train_negative_control = metric(y_pred_train, np.random.permutation(self.data._y_train))
        else:
            r_train = r_train_negative_control = None
        val_generator = DataGenerator(data=self.data, batch_size=self.batch_size, train=False, shuffle=False)    
        y_pred_val = self.predict(val_generator, prediction=prediction)
        r_val = metric(y_pred_val, self.data._y_val)
        gc.collect()

        r_val_negative_control = metric(y_pred_val, np.random.permutation(self.data._y_val))
        gc.collect()
        r_test = r_test_negative_control = test_chroms = None
        if self.test_data is not None:
            test_generator = DataGenerator(data=self.test_data, batch_size=self.batch_size, train=False, shuffle=False)       
            y_pred_test = self.predict(test_generator, prediction=prediction) 
            r_test = metric(y_pred_test, self.test_data._y_val)
            r_test_negative_control = metric(y_pred_test, np.random.permutation(self.test_data._y_val))
            test_chroms = self.test_data.names
            gc.collect()
        plot_score(metric_name,
                   r_train,
                   r_train_negative_control,
                   r_val,
                   r_val_negative_control,
                   r_test,
                   r_test_negative_control,
                   self.data.names,
                   test_chroms,
                   **kwargs)

    def plot_results(self, x_val = None, y_val = None):
        if x_val is None:
            x_val = self.data.x_val_sample
            y_val = self.data.y_val_sample
        if isinstance(x_val, str):
            if x_val == 'train':
                x_val = self.data.x_train_sample
                y_val = self.data.y_train_sample
            if x_val == 'test':
                x_val = self.test_data.x_train_sample
                y_val = self.test_data.y_train_sample
        if isinstance(x_val, tuple) or isinstance(x_val, list):
            sample, number = x_val
            if sample == 'train':
                x_val = self.data.x_train[number:number+9]
                y_val = self.data.y_train[number:number+9]
            elif sample == 'val':
                x_val = self.data.x_val[number:number+9]
                y_val = self.data.y_val[number:number+9]
            elif sample == 'test':
                x_val = self.test_data.x_val[number:number+9]
                y_val = self.test_data.y_val[number:number+9]
        plot_results(self.dna_to_hic(x_val, batch_size=5), y_val)

    def plot_pca(self, 
                 x_train = None,
                 x_val = None,
                 y_train = None, 
                 y_val = None):
        
        if x_val is None:
            x_train = self.data.x_train_sample
            x_val = self.data.x_val_sample
            y_train = self.data.y_latent_train_sample 
            y_val = self.data.y_latent_val_sample
            
        y_pred_train = self.pca(self.dna_to_latent(x_train, batch_size=5))
        y_pred_val = self.pca(self.dna_to_latent(x_val, batch_size=5))
        y_true_train = self.pca(y_train)
        y_true_val = self.pca(y_val)
        plot_pca(y_pred_train,
                             y_pred_val,
                             y_true_train,
                             y_true_val,
                             self.transformed_background)

    def plot_latent_mapping(self, y_val = None):
        if y_val is None:
            y_val = self.data.y_val_sample[:8]
        elif len(y_val) > 8:
            y_val = y_val [:8]
        elif len(y_val) < 8:
            raise ValueError('sample should contain 8 maps')
        latent = self.hic_to_latent(y_val)
        plot_latent_mapping(y_val, self.pca(latent), self.transformed_background)
    
    def mistake_map(self, sample='val', prediction='classical', cmap='RdYlGn_r', adjust=False):
        if sample == 'val' or sample == 'valid':
            gen = DataGenerator(data=self.data, batch_size=self.batch_size, train=False, shuffle=False)
            y_true = self.y_val[:]
        elif sample == 'train':
            gen = DataGenerator(data=self.data, batch_size=self.batch_size, train=True, shuffle=False)
            y_true = self.y_train[:]
        elif sample == 'test':
            gen = DataGenerator(data=self.test_data, batch_size=self.batch_size, train=False, shuffle=False)
            y_true = self.y_test[:]
        y_pred = self.predict(gen, prediction=prediction)
        mae = np.abs(y_pred - y_true)
        map = mae.mean(axis=(0,3))
        del y_pred, y_true, mae
        gc.collect()
        plt.figure(figsize=(7,7))
        if adjust:
            heatmap = plt.imshow(map, cmap=cmap, vmin=0, vmax=1)
        else:
            heatmap = plt.imshow(map, cmap=cmap)
        plt.colorbar(heatmap)
        plt.show()

    def plot_filters(self, figsize = (16, 10), cmap = 'coolwarm', normalize=False):
        filters = self.model.layers[0].get_weights()[0].T
        plot_filters(filters, figsize = figsize, cmap = cmap, normalize=normalize)
    
    def load_test_chromosome(self, chrom, cut_chromosome_ends=None):
        params = self.data.params
        if isinstance(chrom, str):
            chrom = [chrom]
        exclude = list(set(self.data.names + self.data.chroms_to_exclude))
        for i in chrom:
            exclude.remove(i)
        params['chroms_to_exclude'] = exclude
        params['val_split'] = 'test'
        params['min_max'] = self.data.min_max
        if cut_chromosome_ends is not None:
            params['cut_chromosome_ends'] = cut_chromosome_ends
        self.test_data = DataMaster(**params)
        self.train_generator = DataGenerator(data=self.test_data, batch_size=self.batch_size, train=False, encoder=self.enc)
        self.x_test = self.test_data.x_val
        self.y_test = self.test_data.y_val

    def graphic_analisis(self, num, n_layers=3, theme='dark', color_shifts={'heatmap': 50, 'filters': 200},  aggregation='max', return_filters=False):
        first_layers = tf.keras.models.Sequential(self.model.layers[:n_layers])
        pred = first_layers.predict(self.data.x_val[num])[0].T
        ind, start, _ = self.data._x_val[num]
        start, end = start, start + self.data.dna_len
        if self.data.expand_dna:
            pred = pred[:, pred.shape[1] // 4 : -pred.shape[1] // 4]
            brim = self.data.dna_len // 4
            start, end = start + brim, end - brim

        aggregation_rate =  pred.shape[1] // 512
        fun = np.max if aggregation == 'max' else np.mean
        pred = block_reduce(pred, (1, aggregation_rate), fun)

        #hic = self.data.get_region(self.data.names[ind], start, end)[0]
        hic = get_2d(self.data.y_val[num])
        y_lat = self.model.predict(self.data.x_val[num:num+1])
        y_pred = get_2d(self.dec.predict(y_lat))

        filters=self.model.layers[0].get_weights()[0].T

        best_filters = plot_filter_analisis(pred, 
                                    y_pred, 
                                    hic, 
                                    filters,
                                    theme, 
                                    color_shifts)
        if return_filters:
            return best_filters


class MapDrawingCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 model_master,
                 x_sample, 
                 y_sample, 
                 saving_dir, 
                 decoder,
                 epochs):
        super(MapDrawingCallback, self).__init__()
        self.x_sample = x_sample
        self.y_sample = y_sample
        self.decoder = decoder
        self.epochs = epochs
        self.model_master = model_master
        self.saving_dir = os.path.join(saving_dir, 'hic_progress')
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)
        if os.listdir(self.saving_dir):
            self.intit_epoch = int(os.listdir(self.saving_dir)[-1].split('.')[0])
        else:
            self.intit_epoch = 1

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model_master.predict_large_batch(self.model, self.x_sample, batch_size=5)
        y_pred = self.decoder(y_pred)
        file_name = f'{(self.intit_epoch+epoch):03d}_hic.png' if self.epochs < 1000 else  f'{(self.intit_epoch+epoch):04d}_hic.png'
        plot_results(y_pred, self.y_sample, save = os.path.join(self.saving_dir, file_name))



class LatentSpaceDrawingCallback(tf.keras.callbacks.Callback):
    def __init__(self, 
                 model_master,
                 x_train_sample, 
                 x_val_sample, 
                 y_train_sample, 
                 y_val_sample,
                 pca,
                 transformed_background, 
                 saving_dir,
                 epochs):
        super(LatentSpaceDrawingCallback, self).__init__()
        self.pca = pca
        self.x_train_sample = x_train_sample
        self.x_val_sample = x_val_sample
        self.y_true_train = self.pca(y_train_sample)
        self.y_true_val = self.pca(y_val_sample)
        self.transformed_background = transformed_background
        self.model_master = model_master
        self.epochs = epochs
        self.saving_dir = os.path.join(saving_dir, 'pca_progress')
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)
        if os.listdir(self.saving_dir):
            self.intit_epoch = int(os.listdir(self.saving_dir)[-1].split('.')[0])
        else:
            self.intit_epoch = 1

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.pca(self.model_master.predict_large_batch(self.model, self.x_train_sample, batch_size=5))
        y_pred_val = self.pca(self.model_master.predict_large_batch(self.model, self.x_val_sample, batch_size=5))
        file_name = f'{(self.intit_epoch+epoch):03d}_pca.png' if self.epochs < 1000 else  f'{(self.intit_epoch+epoch):04d}_pca.png'
        plot_pca(y_pred_train,
                             y_pred_val,
                             self.y_true_train,
                             self.y_true_val,
                             self.transformed_background,
                             save = os.path.join(self.saving_dir, file_name))
