# -*- coding: utf-8 -*-

import os
import gc

import numpy as np
from time import time
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

import torch

def reshape_for_metric(y, per_fragment, distance):
    n, h, w = y.shape
    if distance is None:
        y = y.reshape((n, h*w))
        if not per_fragment:
            y = y.reshape(n*h*w)
    else:
        if not per_fragment:
            y = y.transpose((1,0,2)).reshape((h, n*w))
    return y

def calculate_metric_numpy(y_true, y_pred, mask, metric, mask2=None):
    if mask2 is None:
        mask2 = mask
    if y_true.ndim == 1:
        y_true = [y_true]
        y_pred = [y_pred]
        mask = [mask]
        mask2 = [mask2]
    scores = []
    for t, p, m1, m2 in zip(y_true, y_pred, mask, mask2):
        m = np.logical_or(m1,m2)
        if t.ndim == 1:
            score = metric(t[m], p[m])
        elif t.ndim == 2:
            score = [metric(i[k], j[k]) for i,j,k in zip(t,p,m)]
        else:
            score = [[metric(i[k][l], j[k][l]) for i,j,k in zip(t,p,m)] for l in range(t.shape[1])]
        scores.append(score)
    scores = np.array(scores)
    return scores[..., 0], scores[..., 1]

def rsquared_numpy(x, y):
    y_bar = y.mean()
    ss_tot = ((y-y_bar)**2).sum()
    ss_res = ((y-x)**2).sum()
    return 1 - (ss_res/ss_tot)

def _pearson_r(x, y):
    eps = 0.0001
    x = x.flatten()
    y = y.flatten()
    xmean = x.mean()
    ymean = y.mean()
    x_centred = x-xmean
    y_centred = y-ymean
    ssxy = torch.sum(x_centred*y_centred)
    ssx = torch.sum(x_centred**2)
    ssy = torch.sum(y_centred**2)
    return ssxy / ((ssx*ssy) ** 0.5 + eps)

def spearman(y_pred, y_true, mask=None):
    if mask is not None:
        mask = mask==0
    else:
        mask = torch.isfinite(y_true)
    scores = []
    y_pred_ranks = y_pred.argsort().argsort()
    y_true_ranks = y_true.argsort().argsort()
    for i, j, m in zip(y_pred_ranks, y_true_ranks, mask):
        r = _pearson_r(i[m], j[m])
        scores.append(r.cpu().detach().numpy())
    return scores

def pearson(y_pred, y_true, mask=None):
    if mask is not None:
        mask = mask==0
    else:
        mask = torch.isfinite(y_true)
    scores = []
    for i, j, m in zip(y_pred, y_true, mask):
        r = _pearson_r(i[m], j[m])
        scores.append(r.cpu().detach().numpy())
    return scores

def rsquared(y_pred, y_true, mask):
    if mask is not None:
        mask = mask==0
    else:
        mask = torch.isfinite(y_true)
    scores = []
    for i, j, m in zip(y_pred, y_true, mask):
        y_bar = j.mean()
        ss_tot = ((j-y_bar)**2).sum()
        ss_res = ((j-i)**2).sum()
        r2 = 1 - (ss_res/ss_tot)
        scores.append(r2.cpu().detach().numpy())
    return scores

def epoch_loop(
    fn,
    model,
    dataset,
    loss=None,
    optimizer=None,
    eval_in_both_modes=False,
    metrics=None,
    verbose=True,
    ):
    metric_history = defaultdict(list)
    if metrics is not None:
        if not (isinstance(metrics, list) or isinstance(metrics, tuple)):
            metrics = [metrics]
    metric_funs = []
    if metrics:
        for metric in metrics:
            if metric == 'pearson':
                metric_funs.append(pearson)
            elif metric =='spearman':
                metric_funs.append(spearman)
            elif metric == 'rsquared':
                metric_funs.append(rsquared)
            else:
                raise ValueError(f'metric {metric} not recognized')
    t0 = time()
    l = len(dataset)
    for index in range(l):
        batch = dataset[index]
        x_batch, y_batch = batch[0], batch[1]
        if len(batch) == 3:
            y_mask = batch[2]
        else:
            y_mask = None
        if optimizer is None:
            _, scores = fn(
                model,
                x_batch,
                y_batch,
                mask_batch=y_mask,
                loss=loss,
                mode='eval',
                metrics=metric_funs
                )
            if eval_in_both_modes:
                _, scores_tr = fn(
                    model,
                    x_batch,
                    y_batch,
                    mask_batch=y_mask,
                    loss=loss,
                    mode='train',
                    metrics=metric_funs
                    )
                scores = [scores[0]] + [(i,j) for  i,j in zip(scores[1:], scores_tr[1:])]
        else:
            _, scores = fn(
                model,
                x_batch,
                y_batch,
                mask_batch=y_mask,
                optimizer=optimizer,
                loss=loss,
                metrics=metric_funs
                )
        metric_history['loss'] += scores[0]
        if metrics:
            for metric, value in zip(metrics, scores[1:]):
                if isinstance(value, tuple):
                    metric_history[metric] += value[0]
                    metric_history[metric+'_train_mode'] += value[1]
                else:
                    metric_history[metric] += value
        t = time() - t0
        if verbose:
            progress_bar(
                train=True,
                l=l,
                step=index,
                t=t,
                metric_history=metric_history)
    return metric_history



def progress_bar(train, l, step, t, metric_history=None):
    step += 1
    out_string = f'Batch {step}/{l}: '
    complete = int(step / l * 30)
    bar = '=' * complete + '>' + '.' * (30 - complete)
    bar = '[' + bar + '] '
    eta = int(t * (l - step) / step)
    eta_str = 'ETA: ' + _parse_time(eta) + ' '
    window_size = max(1, l // 10)
    if train:
        metric_str = ''
        for name, scores in metric_history.items():
            metric_val_str = f'{np.mean(scores[-window_size:]):.4f}'
            metric_str += name + ' = ' + metric_val_str + ' '
    if step < l:
        end = ''
    else:
        eta_str = 'time: ' + _parse_time(t) + ' '
        end = ' '
    if train:
        out_string = out_string + bar + eta_str + metric_str
    else:
        out_string = out_string + bar + eta_str
    print('\r'+out_string, end=end)


def _parse_time(t):
    if t > 3600:
        t = int(t)
        hours = t // 3600
        minutes = t % 3600
        t_str = f'{hours}:{minutes // 60}:{(minutes % 60):0>2}'
    elif t > 60:
        t = int(t)
        t_str = f'{t // 60}:{(t % 60):0>2}'
    else:
        t = np.round(t, 1)
        t_str = str(t) + 's'
    return t_str


def _calculate_metrics(outputs, y_batch, mask_batch, metrics):
    scores = []
    if metrics:
        for metric in metrics:
            score = metric(outputs, y_batch, mask_batch)
            scores.append(score)
    return scores

def train_step(
        model,
        x_batch,
        y_batch,
        mask_batch,
        optimizer,
        loss,
        metrics=None
        ):
    x_batch, y_batch, mask_batch = _to_pytorch_format(
        x_batch,
        y_batch,
        mask_batch
    )

    optimizer.zero_grad()
    outputs = model(x_batch)
    loss_value = loss(y_batch, outputs, mask_batch)
    loss_value.mean().backward()
    optimizer.step()

    scores = []
    scores.append(list(loss_value.cpu().detach().numpy()))
    if isinstance(outputs, list) or isinstance(outputs, tuple):
        outputs = outputs[0]
    scores += _calculate_metrics(outputs, y_batch, mask_batch, metrics)

    return None, scores

def val_step(
    model,
    x_batch,
    y_batch=None,
    mask_batch=None,
    loss=None,
    metrics=None,
    mode='eval',
    no_grad=True,
    ):
    if mode=='eval':
        model.eval()
    else:
        model.train()
    x_batch, y_batch, mask_batch = _to_pytorch_format(
        x_batch,
        y_batch,
        mask_batch
    )
    '''grad_policy = torch.no_grad if no_grad else Pass
    with grad_policy():'''
    outputs = model(x_batch)
    scores = []
    if loss:
        loss_value = loss(y_batch, outputs, mask_batch)
        loss_value = loss_value.cpu().detach().numpy()
        scores.append(list(loss_value))

    if isinstance(outputs, list) or isinstance(outputs, tuple):
        if len(outputs) == 4:
            if outputs[0].size != outputs[1].size: # in vae
                outputs = outputs[0]
            else: # in dna_encoder
                outputs = torch.stack(outputs, axis=1)
        else: # in dna_encoder
            outputs = torch.stack(outputs, axis=1)
    scores += _calculate_metrics(outputs, y_batch, mask_batch, metrics)

    outputs = outputs.cpu().detach().numpy()
    if len(outputs.shape) == 4:
        outputs = outputs.transpose((0,2,3,1))
    return outputs, scores

def _to_pytorch_format(x_batch, y_batch, mask_batch):
    if x_batch.ndim == 3:
        new_shape = (0,2,1)
    elif x_batch.ndim == 4:
        new_shape = (0,3,1,2)
    else:
        new_shape = (0,1)
    x_batch = torch.Tensor(x_batch.copy()).cuda().permute(new_shape)
    if y_batch is not None:
        y_batch = torch.Tensor(y_batch.copy()).cuda().permute((0,3,1,2))
    if mask_batch is not None:
        mask_batch = torch.Tensor(mask_batch.copy()).cuda().permute((0,3,1,2))
    return x_batch, y_batch, mask_batch

def mse_loss(y_true, y_pred, y_mask=None):
    if y_mask is None:
        loss = torch.nn.MSELoss(reduction='none')(y_true, y_pred).mean(axis=1).mean(axis=1).mean(axis=1)
    else:
        y_mask = 1 - y_mask
        mask_mean = y_mask.mean()
        mask_mean = 1 if mask_mean == 0 else mask_mean
        loss = (y_true-y_pred)**2.0 * y_mask
        loss = loss.mean(axis=1).mean(axis=1).mean(axis=1) / mask_mean
    return loss

def vae_loss(y_true, outputs, y_mask=None):
    output, z_mean, z_log_var, z = outputs
    if y_mask is not None:
        y_mask = 1 - y_mask
        mask_mean = y_mask.mean()
        mask_mean = 1 if mask_mean == 0 else mask_mean
        reconstruction_loss = (y_true-output)**2.0 * y_mask
        reconstruction_loss = reconstruction_loss.sum(axis=1).sum(axis=1).sum(axis=1) / mask_mean
    else:
        reconstruction_loss = (y_true-output)**2.0
        reconstruction_loss = reconstruction_loss.sum(axis=1).sum(axis=1).sum(axis=1)
    kl_loss = -0.5*(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var)).sum(axis=1)
    total_loss = reconstruction_loss + kl_loss
    return total_loss

def mask_margins(array, plateau_len):
    '''Gradually masks left and right margins of input maps
    |                  |         |                 |
    |▒▒▒▒▓▒▒▒▒▒▒▒▒▒▓▓▓▓|         |  ░▓▒▒▒▒▒▒▒▒▒▓▒░ |
    |▒▒▓▓▓▓▓▒▒▒▒▒▓▓▓▓▓▓|  ──>    | ░▒▓▓▓▒▒▒▒▒▓▓▓▒░ | 
    |▓▓▓▓▓▓▓▓▓▒▓▓▓▓▓▓▓▓|         | ░▒▓▓▓▓▓▒▓▓▓▓▓▒░ |
    '''
    n, h, w, c = array.shape
    x = np.arange(w)
    y = np.zeros(w)
    plateau_start = (w - plateau_len) // 2
    plateau_end = plateau_start + plateau_len
    y[plateau_start:plateau_end] = 1
    y[:plateau_start] = np.linspace(0, 1, plateau_start)
    y[plateau_end:] = np.linspace(1, 0, w - plateau_end)
    y = y[None, None, :, None]
    mask = y.repeat(n, axis=0).repeat(h, axis=1).repeat(c, axis=3)
    return array * mask

def combine_shifts(y, coverage):
    '''For combining overlapping predictions'''
    n, h, w, c = y.shape
    step = w // coverage
    remainder = w % coverage
    y_masked = mask_margins(y, step)
    left_margin = (w - step) // 2
    right_margin = w - step - left_margin
    w_tot = w * (n//coverage) + left_margin + right_margin
    result = np.zeros((h, w_tot, c))
    # if remainder != 0, some steps should be enlarged
    to_be_enlarged = np.zeros(n).astype(bool)
    to_be_enlarged[:remainder] = True
    # for uniform distibution just enlarge steps randomly
    np.random.shuffle(to_be_enlarged)
    ind = 0
    for i in range(n):
        if to_be_enlarged[i]:
            ind += 1
        result[:, ind : ind + w] += y_masked[i]
        ind += step
    return result[:, left_margin : w_tot - right_margin]