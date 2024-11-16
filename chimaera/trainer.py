# -*- coding: utf-8 -*-

import os
import gc

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from scipy import stats

import tensorflow as tf
import torch
import torch.nn as nn
import torchsummary
import json

from . import data_utils 
from . import plot_utils
from . import train_utils
from . import models
from .dataset import Dataset
from .model_plotting import ModelPlotter

class ChimaeraModule():
    def __init__(self, model):
        self.model = model
        self.batch_size = 1

    def train_one_epoch(
            self,
            train_dataset,
            val_dataset,
            optimizer,
            loss,
            metrics,
            eval_in_both_modes,
        ):
        train_dataset.shuffle()
        val_dataset.shuffle()

        epoch_train_metric_history = train_utils.epoch_loop(
            fn=train_utils.train_step,
            model=self.model,
            dataset=train_dataset,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            verbose=True
            )
        epoch_val_metric_history = train_utils.epoch_loop(
            fn=train_utils.val_step,
            model=self.model,
            dataset=val_dataset,
            loss=loss,
            eval_in_both_modes=eval_in_both_modes,
            metrics=metrics,
            verbose=False
            )
        train_metrics = {i:np.nanmean(j) for i,j in epoch_train_metric_history.items()}
        val_metrics = {i:np.nanmean(j) for i,j in epoch_val_metric_history.items()}
        return train_metrics, val_metrics


    def predict(self, dataset, batch_size=None, verbose=True, mode='eval'):
        outputs = []
        l = len(dataset)
        t0 = time()

        if isinstance(dataset, data_utils.DNALoader) or \
           isinstance(dataset, data_utils.MutantDNALoader) or \
           isinstance(dataset, data_utils.ChimericDNALoader):
            dataset = data_utils.DNAPredictGenerator(
                dataset,
                batch_size=self.batch_size
            )
        elif isinstance(dataset, data_utils.MapDataLoader):
            dataset = data_utils.HiCTrainValGenerator(dataset, 64)

        if isinstance(dataset, data_utils.DataGenerator):
            for index in range(l):
                x_batch = dataset[index]
                if isinstance(x_batch, tuple):
                    if x_batch[0].shape != x_batch[0].shape:
                        raise ValueError(f"DataGenerator returned two batches. \
In prediction mode it's considered as mut and wt DNA but they have different \
shapes ({x_batch[0].shape} and {x_batch[0].shape})")
                    output_mut, _ = train_utils.val_step(self.model, x_batch[0], mode=mode)
                    output_wt, _ = train_utils.val_step(self.model, x_batch[1], mode=mode)
                    output = output_mut - output_wt
                else:
                    output, _ = train_utils.val_step(self.model, x_batch, mode=mode)
                outputs.append(output)
                t = time() - t0
                if verbose:
                    train_utils.progress_bar(train=False, step=index, l=l, t=t)
        else:
            if batch_size is None:
                raise ValueError("x isn't DataGenerator, specify batch_size")
            l = int(np.ceil(l / batch_size))
            for index in range(l):
                x_batch = dataset[index * batch_size : (index+1) * batch_size]
                output, _ = train_utils.val_step(self.model, x_batch, mode=mode)
                outputs.append(output)
                t = time() - t0
                if verbose:
                    train_utils.progress_bar(train=False, step=index, l=l, t=t)
        if verbose:
            print()
        if len(outputs) > 0:
            return np.concatenate(outputs)
        else:
            return np.array([])

    def summary(self):
        # input_shape attributes were set manually for all models
        torchsummary.summary(self.model, self.model.input_shape)

    def save(self, path):
        model_scripted = torch.jit.script(self.model)
        model_scripted.save(path+'.pt')
        #torch.save(self.model.state_dict(), path+'.pt')

class ModelContainer():
    def __init__(
            self,
            data=None,
            hic_data_path=None,
            genome=None,
            show_data=True,
            test_chroms_only=True,
            batch_size=8,
            model_dir=None,
            ae_dir=None,
            dna_encoder=None,
            hic_encoder=None,
            hic_decoder=None,
            lr=0.001,
            vae_lr=0.001,
            saving_dir='./',
            prediction_mode='eval',
            model_variant='small',
            **kwargs
            ):
        self.saving_dir = saving_dir
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)
        self.batch_size = batch_size
        if data is None and model_dir:
            data = self._load_data(model_dir,
                                   hic_data_path,
                                   genome,
                                   test_chroms_only=test_chroms_only,
                                   show_data=show_data)
        elif data is None and model_dir is None:
            raise ValueError("Data can't be loaded without model dir")
        self.data = data
        if model_variant == 'small':
            model_params = {
                'dropout_rates': [0.0, 0.1, 0.1, 0.1, 0.2],
                'n_residual_blocks': 0,
                'latent_dim': 96,
            }
        elif model_variant == 'middle':
            model_params = {
                'dropout_rates': [0.0, 0.08, 0.05, 0.1, 0.15],
                'n_residual_blocks': 4,
                'latent_dim': 128,
                'residual_block_dropout_rates': [0.1, 0.0]
            }
        elif model_variant == 'big':
            model_params = {
                'dropout_rates': [0.0, 0.05, 0.0, 0.05, 0.15],
                'n_residual_blocks': 8,
                'latent_dim': 128,
                'residual_block_dropout_rates': [0.05, 0.0]
            }
        else:
            raise ValueError("model_variant should be 'small', 'middle' or 'big'")

        model_params.update(kwargs)
        self.model_params = model_params
        if data.square_maps:
            self.hic_shape = (self.data.square_w//2,
                            self.data.square_w//2,
                            self.data.out_channels)
        else:
            self.hic_shape = (self.data.h,
                            self.data.map_size,
                            self.data.out_channels)
        self.dna_shape = (self.data.dna_len, 4)

        self.prediction_mode = prediction_mode
        self.lr = lr
        self.denoised = False

        self.sd = None # will be updated in .denoise_samples()
        self.mean = None # will be updated in .denoise_samples()


        if model_dir:
            if not ae_dir: # load all models from one dir
                dna_encoder = self._load_dna_model(model_dir)
                hic_encoder, hic_decoder = self._load_hic_models(model_dir)
            else: # load hi-c autoencoder from another dir
                dna_encoder = self._load_dna_model(model_dir)
                hic_encoder, hic_decoder = self._load_hic_models(ae_dir)
        elif ae_dir: # load only hi-c autoencoder, dna model will be new
            hic_encoder, hic_decoder = self._load_hic_models(ae_dir)

        self.dna_encoder = dna_encoder
        self.hic_encoder = hic_encoder
        self.hic_decoder = hic_decoder
        # if some models are None, create new ones
        self._build_missing_models()
        # if some saved autoencoder used, it is thought to be already trained
        # so it can be used for denoising

        # latent_dim could be changed in _build_missing_models() so using actual
        self.latent_dim = self.hic_encoder.model.output_shape[-1]

        # stacking dna_encoder and hic_decoder
        self.combined_model = self._make_main_model()
        self.hic_encoder.batch_size = 64
        self.hic_decoder.batch_size = 64
        self.dna_encoder.batch_size = batch_size
        self.combined_model.batch_size = batch_size
        self.dna_optimizer = self._optimizer(lr, self.combined_model.model)
        if model_dir and 'optimizer' in os.listdir(model_dir):
            state_dict = torch.load(os.path.join(model_dir, 'optimizer'))
            try:
                self.dna_optimizer.load_state_dict(state_dict)
            except:
                print("WARNING: saved optimizer can't be loaded so you can't \
continue model training")

        # stacking hic_encoder and hic_decoder, adding sampling layer
        self.vae = self._make_vae()
        self.decoder_frozen = False # now it can be trained
        self.vae_optimizer = self._optimizer(vae_lr, self.vae.model)

        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)


    def _load_data(self, model_dir, hic, genome, test_chroms_only, show_data):
        '''For ready models data may be loaded using saved params'''
        data_params_file = os.path.join(model_dir, 'data_params.json')
        if os.path.exists(data_params_file):
            with open(data_params_file, 'r') as f:
                data_params = json.load(f)
        data_params.pop('batch_size')
        if test_chroms_only:
            chroms = [i.split(':')[0] for i in data_params['test_regions'].split(';')]
            data_params['chroms_to_include'] = chroms

        data = Dataset(hic_data_path=hic,
                       genome=genome,
                       show=show_data,
                       **data_params)
        return data


    def _load_dna_model(self, model_dir):
        # check if data params now are the same as ones the model trained with
        data_params_file = os.path.join(model_dir, 'data_params.json')
        if os.path.exists(data_params_file):
            with open(data_params_file, 'r') as f:
                data_params = json.load(f)
            self._check_data_params(data_params)
        # check model settings
        model_params_file = os.path.join(model_dir, 'model_params.json')
        if os.path.exists(model_params_file):
            with open(model_params_file, 'r') as f:
                model_params = json.load(f)
            if 'prediction_mode' in model_params:
                if model_params['prediction_mode'] != self.prediction_mode:
                    print(f"WARNING: Using prediction mode of saved model: {model_params['prediction_mode']}")
                self.prediction_mode = model_params['prediction_mode']
            if 'lr' in model_params:
                self.lr = model_params['lr']
            if 'sd' in model_params:
                self.sd = model_params['sd']
            self.model_params.update(model_params)
            mean_map_file = os.path.join(model_dir, 'mean_map.npy')
            if os.path.exists(mean_map_file):
                self.mean = np.load(mean_map_file)   
        dna_encoder = ChimaeraModule(torch.jit.load(os.path.join(model_dir, 'dna_encoder.pt')))
        return dna_encoder


    def _load_hic_models(self, ae_dir):
        # you may use hic encoder and decoder trained on other organism as transfer learning 
        hic_encoder = ChimaeraModule(torch.jit.load(os.path.join(ae_dir, 'hic_encoder.pt')))   
        hic_decoder = ChimaeraModule(torch.jit.load(os.path.join(ae_dir, 'hic_decoder.pt')))
        return hic_encoder, hic_decoder

    def _check_data_params(self, loaded_params):
        # check if data params now are the same as ones the model trained with
        current_params = self._make_data_params()
        for param in ['h', 'offset', 'fragment_length']:
            if current_params[param] != loaded_params[param]:
                raise ValueError(f"Current param '{param}' is \
{current_params[param]} but the model was trained on data with {param}=\
{loaded_params[param]}. Now the model won't work, change this param or train a \
new model")
        for param in ['sigma','nan_threshold','val_fraction','remove_first_diag']:
            if current_params[param] != loaded_params[param]:
                print(f"WARNING: Current param '{param}' is \
{current_params[param]} but the model was trained on data with {param}=\
{loaded_params[param]}. It's not crucial but take it into account")
        if self.batch_size != loaded_params['batch_size']:
            print(f"WARNING: current batch size ({current_params['batch_size']}\
) differs from the original one ({loaded_params['batch_size']}). You may \
change it by setting batch_size attribute")
        if current_params['test_regions'] != loaded_params['test_regions']:
            print(f"WARNING: the loaded model was trained with the folowing \
test regions: {loaded_params['test_regions']}, now they are \
{current_params['test_regions']}. Ensure you are not testing this model on \
regions from its train sample (runing in-silico mutagenesis on train regions \
is also not recommended)")

    def _build_missing_models(self):
        hic_shape = self.hic_shape[0], self.hic_shape[1], 1
        dna_shape = self.dna_shape
        latent_dim = self.model_params.pop('latent_dim')
        if self.hic_encoder is not None:
            encoder_latent_dim = self.hic_encoder.model.output_shape[-1]
            if latent_dim != encoder_latent_dim:
                print(f'WARNING: using latent dim of loaded autoencoder: \
{encoder_latent_dim}')
                latent_dim = encoder_latent_dim
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.hic_encoder is None:
            self.hic_encoder = ChimaeraModule(
                model = models.hic_encoder(
                    input_shape=hic_shape,
                    latent_dim=latent_dim
                ).to(device)
            )
        if self.hic_decoder is None:
            self.hic_decoder = ChimaeraModule(
                model = models.hic_decoder(
                    output_shape=hic_shape,
                    latent_dim=latent_dim
                ).to(device)
            )
        if self.dna_encoder is None:
            self.dna_encoder = ChimaeraModule(
                model = models.dna_encoder(
                    input_shape=dna_shape,
                    latent_dim=latent_dim,
                    n_outputs=self.data.out_channels,
                    **self.model_params
                ).to(device)
            )

    def summary(self):
        # print the main model architecture
        self.combined_model.summary()

    def _make_data_params(self):
        # for saving with the models
        data_params = dict()
        data_params['fragment_length'] = self.data.mapped_len
        data_params['offset'] = self.data.offset
        data_params['batch_size'] = self.batch_size
        data_params['h'] = self.data.h
        data_params['val_fraction'] = self.data.val_fraction
        data_params['sigma'] = self.data.sigma
        data_params['nan_threshold'] = self.data.nan_threshold
        data_params['remove_first_diag'] = self.data.remove_first_diag
        data_params['organism_name'] = self.data.organism
        data_params['psi'] = self.data.psi
        data_params['cross_threshold'] = self.data.cross_threshold
        data_params['scale_by'] = self.data.scale_by
        data_params['skip_quadrants'] = self.data.skip_quadrants
        data_params['enlarge_crosses'] = self.data.enlarge_crosses
        data_params['interpolation_order'] = self.data.interpolation_order
        data_params['test_regions'] = '; '.join([i[0]+':'+str(i[1])+'-'+str(i[2]) for i in self.data.test_regions])
        return data_params

    def _save_params(self):
        # save params with the models
        data_params = self._make_data_params()
        with open(os.path.join(self.saving_dir, 'data_params.json'), 'w') as f:
            json.dump(data_params, f)

        model_params = self.model_params.copy()
        model_params['latent_dim'] = self.latent_dim
        model_params['dna_shape'] = self.dna_shape
        model_params['hic_shape'] = self.hic_shape
        model_params['prediction_mode'] = self.prediction_mode
        model_params['lr'] = self.lr
        model_params['sd'] = eval(str(self.sd))
        with open(os.path.join(self.saving_dir, 'model_params.json'), 'w') as f:
            json.dump(model_params, f)
        if self.mean is not None:
            np.save(os.path.join(self.saving_dir, 'mean_map.npy'), self.mean)

    def save(self, update_dna_encoder=False):
        # save all models and params
        if update_dna_encoder:
            self.dna_encoder.save(os.path.join(self.saving_dir, 'dna_encoder'))
            torch.save(self.dna_optimizer.state_dict(),
                    os.path.join(self.saving_dir, 'optimizer'))
        self.hic_encoder.save(os.path.join(self.saving_dir, 'hic_encoder'))
        self.hic_decoder.save(os.path.join(self.saving_dir, 'hic_decoder'))
        self._save_params()

    def _optimizer(self, lr, model):
        # create an Adam optimizer
        params = [p for p in model.parameters()]
        return torch.optim.NAdam(params, lr=lr)#, weight_decay=0.00001

    def _make_main_model(self):
        # combine the DNA encoder and Hi-C decoder
        combined_model = models.EnsembleModel(
            self.dna_encoder.model,
            self.hic_decoder.model
        )
        return ChimaeraModule(combined_model)

    def _make_vae(self):
        # combine Hi-C decoder and encoder and add reparametrization
        vae = models.VAE(self.hic_encoder.model, self.hic_decoder.model)
        return ChimaeraModule(model=vae)


    def _freeze_decoder(self):
        self.decoder_frozen = True
        for param in self.hic_decoder.model.parameters():
            param.requires_grad = False
        self.hic_decoder.model.eval()

    def on_epoch_end(
            self,
            train_metric_history,
            val_metric_history,
            epoch,
            use_raw_maps,
            metric
            ):
        if not os.path.exists(os.path.join(self.saving_dir, 'progress')):
            os.mkdir(os.path.join(self.saving_dir, 'progress'))
        if val_metric_history[metric][-1] == max(val_metric_history[metric]):
            self.dna_encoder.save(os.path.join(self.saving_dir, 'dna_encoder'))
            torch.save(self.dna_optimizer.state_dict(),
                    os.path.join(self.saving_dir, 'optimizer'))
        fig_saving_path = os.path.join(self.saving_dir,
                                        f'progress/metrics.png')
        plot_utils.plot_metrics_history(train_metric_history, val_metric_history,
                                save=fig_saving_path)
        '''fig_saving_path = os.path.join(self.saving_dir,
                                        f'progress/epoch_{epoch+1}.png')
        self.plot_results(
            save=fig_saving_path,
            raw_maps=use_raw_maps,
            exclude_imputed=True,
            strand='one',
            sample='val'
            )'''

    def train_hic_autoencoder(
            self,
            epochs,
            metrics='pearson',
            stop_at=0.9, # if val metric becomes more than this value, training
            #              stops (if many metrics provided, using the first one)
            mask_interpolated=False,
            shifts=16,
            lr=None,
            eval_in_both_modes=False,
            random_flip=True
        ):
        random_flip = 0.5 if random_flip else False

        if stop_at is None:
            stop_at = np.inf

        if metrics:
            if isinstance(metrics, str):
                early_stopping = lambda history: history[metrics][-1] > stop_at
            else:
                early_stopping = lambda history: history[metrics[0]][-1] > stop_at
        else:
            early_stopping = None

        if lr is not None:
            self._change_lr(self.vae_optimizer, lr)

        train_metric_history = defaultdict(list)
        val_metric_history = defaultdict(list)
        for epoch in range(epochs):
            self.data.reset_train_sample()
            for shift in range(shifts):
                print(f'Epoch {epoch+1}/{epochs}, shift {shift+1}/{shifts}:')
                self.data.shift_train_sample(1/shifts)
                y_train = data_utils.HiCLoader(self.data, self.data.train_sample, check='train')
                y_val = data_utils.HiCLoader(self.data, self.data.val_sample, check='not-train')
                if mask_interpolated:
                    mask_train = data_utils.MaskLoader(self.data, self.data.train_sample, check='train')
                    mask_val = data_utils.MaskLoader(self.data, self.data.val_sample, check='not-train')
                else:
                    mask_train = mask_val = None
                train_dataset = data_utils.HiCTrainValGenerator(
                    x=y_train,
                    mask=mask_train,
                    batch_size=8,
                    random_flip=random_flip
                    )
                val_dataset = data_utils.HiCTrainValGenerator(
                    x=y_val,
                    mask=mask_val,
                    batch_size=8,
                    random_flip=random_flip
                    )

                self.hic_encoder.model.train()
                self.hic_decoder.model.train()
                if self.decoder_frozen:
                    raise ValueError('Decoder is frozen')

                train_metrics, val_metrics = self.vae.train_one_epoch(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    optimizer=self.vae_optimizer,
                    loss=train_utils.vae_loss,
                    eval_in_both_modes=eval_in_both_modes,
                    metrics=metrics
                    )

                for i,j in train_metrics.items():
                    train_metric_history[i].append(j)
                for k,l in val_metrics.items():
                    val_metric_history[k].append(l)
                    print(f'val_{k} = {l:.4f}', end=' ')
                print()

                if early_stopping is not None:
                    check = early_stopping(val_metric_history)
                    if check:
                        print(f'Metric reached required value ({stop_at:.2f})')
                        return

    def train_dna_encoder(
            self,
            epochs=100,
            random_flip=True,
            raw_maps=False,
            raw_maps_sigma=1,
            lr=0.001,
            metrics='pearson',
            metric_for_choosing_best_model='pearson',
            shifts=2,
            mask_interpolated=True,
            eval_in_both_modes=False,
            freeze_decoder=True,
        ):
        # probability of random rev comp:
        rc = 0.5 if random_flip else False
        # if the main model was frozen:
        for p in self.dna_encoder.model.parameters():
            p.requires_grad = True
        # freeze decoder weights:
        if freeze_decoder:
            self._freeze_decoder()
        self.save()
        # use small gaussian blur if training on raw maps
        sigma = raw_maps_sigma if raw_maps else 0

        if lr is not None:
            self._change_lr(self.dna_optimizer, lr)

        for epoch in range(epochs):
            self.data.reset_train_sample()
            for shift in range(shifts):
                print(f'Epoch {epoch+1}/{epochs}, shift {shift+1}/{shifts}:')
                self.data.shift_train_sample(1/shifts)
                x_train = data_utils.DNALoader(self.data, self.data.train_sample, check='train')
                x_val = data_utils.DNALoader(self.data, self.data.val_sample, check='not-train')
                y_train = data_utils.HiCLoader(self.data, self.data.train_sample, check='train')
                y_val = data_utils.HiCLoader(self.data, self.data.val_sample, check='not-train')
                if not raw_maps:
                    y_train = self.denoise(y_train[:]) #!!!
                    y_val = self.denoise(y_val[:])
                if mask_interpolated:
                    mask_train = data_utils.MaskLoader(self.data, self.data.train_sample, check='train')
                    mask_val = data_utils.MaskLoader(self.data, self.data.val_sample, check='not-train')
                else:
                    mask_train = mask_val = None
                train_dataset = data_utils.DNATrainValGenerator(
                    x=x_train,
                    y=y_train,
                    mask=mask_train,
                    batch_size=self.batch_size,
                    rc=rc,
                    sigma=sigma)
                val_dataset = data_utils.DNATrainValGenerator(
                    x=x_val,
                    y=y_val,
                    mask=mask_val,
                    batch_size=self.batch_size,
                    rc=0,
                    sigma=sigma)

                self.dna_encoder.model.train()
                if freeze_decoder:
                    self.hic_decoder.model.eval()
                else:
                    self.hic_decoder.model.train()

                train_metrics, val_metrics = self.combined_model.train_one_epoch(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    optimizer=self.dna_optimizer,
                    loss=train_utils.mse_loss,
                    eval_in_both_modes=eval_in_both_modes,
                    metrics=metrics
                    )

                for i,j in train_metrics.items():
                    self.train_history[i].append(j)
                for k,l in val_metrics.items():
                    self.val_history[k].append(l)
                    print(f'val_{k} = {l:.4f}', end=' ')
                print()

                self.on_epoch_end(
                    self.train_history,
                    self.val_history,
                    epoch=epoch,
                    use_raw_maps=raw_maps,
                    metric=metric_for_choosing_best_model
                    )


    def _change_lr(self, optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def denoise_samples(self):
        self.data.y_train_denoised = self.denoise(self.data.y_train[:])
        self.data.y_val_denoised = self.denoise(self.data.y_val[:])
        self.data.y_test_denoised = self.denoise(self.data.y_test[:])
        self.denoised = True
        if self.sd is None:
            denoised_copy = self.data.y_train_denoised.copy()
            denoised_copy[self.data.y_train.mask[:]>0.2] = np.nan
            self.sd = np.nanstd(denoised_copy) # for standartization in some 
                                               # interpretation methods
            self.mean = np.nanmean(denoised_copy, axis=0) # for autoencoder bias
                                                          # avoiding

    def hic_to_latent(self, x, verbose=0):
        return self.hic_encoder.predict(x, verbose=verbose, batch_size=64)

    def latent_to_hic(self, x, verbose=0):
        return self.hic_decoder.predict(x, verbose=verbose, batch_size=64)

    def denoise(self, x, verbose=0):
        outputs = []
        for i in range(self.data.out_channels):
            outputs.append(self.latent_to_hic(self.hic_to_latent(x[...,i:i+1]), verbose))
        if len(outputs) > 0:
            return np.concatenate(outputs, axis=-1)
        else:
            return np.array([])

    def dna_to_latent(self, x, verbose=0):
        return self.dna_encoder.predict(x, verbose=verbose, batch_size=self.batch_size, mode=self.prediction_mode)

    def dna_to_hic(self, x, verbose=0):
        return self.combined_model.predict(x, verbose=verbose, batch_size=self.batch_size, mode=self.prediction_mode)

    def predict(self, x, strand='both', verbose=0):
        y = self.dna_to_hic(x, verbose)
        # predict both forward and revcomp, return average result
        if strand == 'both':
            if isinstance(x, data_utils.DNALoader) or isinstance(x, data_utils.MutantDNALoader):
                x = data_utils.DNAPredictGenerator(x, batch_size=self.batch_size)
            if isinstance(x, data_utils.DataGenerator):
                x.rc = True
            else:
                x = np.flip(x, axis=(1,2))
            y_rc_flipped = self.dna_to_hic(x, verbose)
            y_rc = np.flip(y_rc_flipped, axis=2)
            y = (y + y_rc) / 2
        return y

    def _apply_slides(self, loader, n):
        regions = loader.regions
        plateau_len = self.data.mapped_len // n
        margin = (self.data.mapped_len - plateau_len) // 2
        shifts = np.arange(n)*plateau_len - margin
        loaders = []
        for shift in shifts:
            new_regions = [(ch, s + shift, e + shift) for ch, s, e in regions]
            new_loader = data_utils.DNALoader(self.data, new_regions)
            loaders.append(new_loader)
        return loaders

    def score(self,
            metric='pearson',
            sigma=0,
            strand='both',
            raw_maps=False,
            distance='best',
            per_fragment=True,
            exclude_imputed=True,
            shifts=1,
            normalize=True,
            folder='',
        ):
        if metric == 'pearson':
            metric_fun = stats.pearsonr
        elif metric == 'spearman':
            metric_fun = stats.spearmanr
        else:
            raise ValueError("Metric should be 'pearson' or 'spearman'")
        if raw_maps:
            # metric with raw maps with small gaussian blur
            y_test = gaussian_filter(self.data.y_test[:], (0, 1, 1, 0))
        else:
            # metric with autoencoder processed maps
            if not self.denoised:
                self.denoise_samples()
            y_test = self.data.y_test_denoised.copy()
            if normalize:
                y_test -= self.mean # to avoid autoencoder bias

        # using only test sample
        loader = self.data.x_test

        # make predictions with sliding window and calculate mean
        if shifts > 1:
            loaders = self._apply_slides(loader, shifts)
            slides = [self.predict(l, strand=strand) for l in loaders]
            slides = np.stack(slides, axis=1)
            slides = [train_utils.combine_shifts(i, shifts) for i in slides]
            y_pred = np.array(slides)
            del slides
        else: # make only one prediction for each fragment
            y_pred = self.predict(loader, strand=strand)
        if normalize:
            y_pred -= self.mean # to avoid autoencoder bias
        # also may add gaussian blur to predictions
        if sigma > 0:
            y_pred = gaussian_filter(y_pred, (0,sigma,sigma,0))
            y_test = gaussian_filter(y_test, (0,sigma,sigma,0))
        # do not calculate score for interpolated pixels
        if exclude_imputed:
            mask = self.data.y_test.mask[:] < 0.2
        else:
            mask = np.ones(y_test.shape, dtype=bool)
        mask = train_utils.reshape_for_metric(
                mask[...,0],
                per_fragment,
                distance
            )
        for e in range(self.data.out_channels):
            # calculate scole for each experiment separately
            y_pred_i = y_pred[...,e]
            y_test_i = y_test[...,e]

            y_pred_i = train_utils.reshape_for_metric(
                y_pred_i,
                per_fragment,
                distance
            )
            y_test_i = train_utils.reshape_for_metric(
                y_test_i,
                per_fragment,
                distance
            )

            scores, pvals = train_utils.calculate_metric_numpy(
                y_test_i,
                y_pred_i,
                mask,
                metric_fun
            )
            # calculate metric between randomly shuffled pairs of true and pred
            if per_fragment:
                index = np.random.permutation(np.arange(len(y_test_i)))
                control_scores, control_pvals = train_utils.calculate_metric_numpy(
                    y_test_i,
                    y_pred_i[index],
                    mask,
                    metric_fun,
                    mask[index]
                )
            else:
                control_scores, control_pvals = train_utils.calculate_metric_numpy(
                    y_test_i,
                    np.flip(y_pred_i, axis=-1),
                    mask,
                    metric_fun,
                    np.flip(mask, axis=-1),
                )

            x = np.linspace(self.data.min_dist, self.data.max_dist, self.data.h+1)
            title = self.data.organism + ' ' + self.data.experiment_names[e]

            if distance == 'all' and per_fragment:
                plot_utils.plot_score_full(
                    metric,
                    scores,
                    control_scores,
                    x,
                    title,
                    folder
                    )
            elif distance == 'all' and not per_fragment:
                plot_utils.plot_score_line(
                    metric,
                    scores,
                    control_scores,
                    x,
                    title
                    )
            elif distance == 'best' and per_fragment:
                plot_utils.plot_score_one_distance(
                    metric,
                    scores,
                    control_scores,
                    x,
                    title
                    )
            elif distance == 'best' and not per_fragment:
                i = np.argmax(scores)
                print(f'Best {metric} value is {scores[i]:.4f} at {int(x[i])} \
bp distance with p-value {pvals[i]:1e}')
                print(f'{metric.capitalize()} between the true map and a revesed \
predicted one at the same distance is {control_scores[i]:.4f} with p-value \
{control_pvals[i]:.1e}')
            elif distance is None and per_fragment:
                plot_utils.plot_score_basic(
                    metric,
                    scores,
                    control_scores,
                    x,
                    title
                )
            elif distance is None and not per_fragment:
                print(f'{metric.capitalize()} value is {scores[0]:.4f} \
with p-value {pvals[0]:1e}')
                print(f'{metric.capitalize()} between the true map and a revesed \
predicted one is {control_scores[0]:.4f} with p-value {control_pvals[0]:.1e}')
            else:
                raise ValueError("Incorrect 'distance' arg")
                    
    def plot_results(self,
                        sample='test',
                        index=0,
                        style='square',
                        equal_scale='pairs',
                        save=False,
                        strand='both',
                        exclude_imputed=True,
                        raw_maps=False,
                        experiment_name=None,
                        experiment_index=0,
                        zero_centred=False,
                        shifts=1):
        experiment_name, experiment_index = self.data._parse_experiment_name(
            experiment_name,
            experiment_index
        )
        n = 9 if style=='square' else 8
        postfix = '' if raw_maps else '_denoised'
        if not raw_maps and not self.denoised:
            self.denoise_samples()
        x_sample = eval('self.data.x_'+sample)
        y_sample_ = eval('self.data.y_'+sample)
        y_sample = eval('self.data.y_'+sample+postfix)
        y = y_sample[index : index + n]

        if shifts == 1:
            x = x_sample[index : index + n]
            y_pred = self.predict(x, strand=strand)
        else:
            regions = x_sample.regions[index : index + n].copy()
            loader = data_utils.DNALoader(self.data, regions)
            loaders = self._apply_slides(loader, shifts)
            slides = [self.predict(l, strand=strand) for l in loaders]
            slides = np.stack(slides, axis=1)
            slides = [train_utils.combine_shifts(i, shifts) for i in slides]
            y_pred = np.array(slides)

        if exclude_imputed:
            y = y.copy()
            y_mask = y_sample_.mask[index : index + n]
            y_pred[y_mask > 0.1] = np.nan
            y[y_mask > 0.1] = np.nan
        plot_utils.plot_results(
            y_pred,
            y,
            sample=sample,
            numbers=np.arange(index, index+n),
            experiment_name=experiment_name,
            experiment_index=experiment_index,
            data=self.data,
            equal_scale=equal_scale,
            save=save,
            zero_centred=zero_centred
        )


    def plot_vae_results(self,
                        sample='val',
                        index=0,
                        style='square',
                        exclude_imputed=True,
                        zero_centred=False,
                        experiment_name=None,
                        experiment_index=0,
                        equal_scale='pairs', save=False):
        experiment_name, experiment_index = self.data._parse_experiment_name(
            experiment_name,
            experiment_index
        )
        n = 9 if style=='square' else 8
        y_sample = eval('self.data.y_'+sample)
        y = y_sample[index : index + n]
        y_pred = self.denoise(y)
        if exclude_imputed:
            y = y.copy()
            y_mask = y_sample.mask[index : index + n]
            y_pred[y_mask > 0.1] = np.nan
            y[y_mask > 0.1] = np.nan
        plot_utils.plot_results(
            y_pred,
            y,
            sample=sample,
            numbers=np.arange(index, index+n),
            data=self.data,
            equal_scale=equal_scale,
            experiment_name=experiment_name,
            experiment_index=experiment_index,
            save=save, zero_centred=zero_centred
        )


    def plot_model(self):
        if 'n_residual_blocks' in self.model_params.keys():
            n_blocks = self.model_params['n_residual_blocks']
        else:
            n_blocks = 6
        plotter = ModelPlotter(
            n_blocks,
            [256,128],
            certain_len=True).plot(**self.model_params)
