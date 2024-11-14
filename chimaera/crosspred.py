import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import gaussian_filter
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

def save_pred(root, y, org1, org2=None, rep=None, mask=False):
    if org2:
        if rep:
            saving_path = root+org1+'/'+'predicted_by_'+org2+'_model_control_'+str(rep)+'.npy'
        else:
            saving_path = root+org1+'/'+'predicted_by_'+org2+'_model.npy'
    else:
        if not mask:
            saving_path = root+org1+'/groundtruth.npy'
        else:
            saving_path = root+org1+'/mask.npy'
    np.save(saving_path, y)


def predict_by_all_models(datasets, data, region, data_org):
    ys = {}
    seq = data.get_dna(region, seq=True)
    param_names = ['dna_len', 'mapped_len', 'offset', 'binsize']
    # binsize here is not an actual binsize of Hi-C map but 1/128 of mapped_len
    data_params = {i:getattr(data, i) for i in param_names}
    for org, chi in datasets.items():
        actual_params = {i:getattr(chi.data, i) for i in param_names}
        actual_model_data = chi.data
        current_model_org = chi.data.organism
        chi.data = data # replace data for model (DNA, Hi-C)
        for param, value in actual_params.items(): # change its params for this model
            setattr(chi.data, param, value)
        y = chi.predict_region(
                region,
                plot=False,
                edge_policy='cyclic',
                )
        save_pred(root, y, data_org, org, i)
        ys[org] = y
        for param, value in data_params.items(): # set params back
            setattr(chi.data, param, value)
        chi.data = actual_model_data # set dataset back
    return ys


def cross_predictions(models):
    use_list = list(models.keys())
    all_rs = []
    significance = []
    all_between_replics = []
    for i, (org, model) in enumerate(models.items()):
        if not os.path.exists(root+org):
            os.mkdir(root+org)
        data = model.data
        print(org)
        chrom = data.chromnames[0]
        region = regions[org]
        hic = data.get_hic(region, edge_policy='cyclic')
        mask = data.get_mask(region, edge_policy='cyclic')
        if plot:
            y_plot = hic.copy()
            y_plot[mask>0.2] = np.nan
            _, ax = plt.subplots(figsize=(20,5))
            ax.set_title('True '+ data.organism)
            plot_map(y_plot, ax=ax)
            plt.show()
        save_pred(root, hic, org)
        save_pred(root, hic, org, mask=True)
        ys = predict_by_all_models(models, data, region, org)

def cross_correlate(y, y_correct, mask):
    rs_control = []
    target_len = y_correct.shape[1] # shape (n diagonals, taget map length)
    zoom_rate = target_len / y.shape[1]
    y = zoom(y, (zoom_rate, zoom_rate, 1), order=1)
    shifts = 20
    step = target_len // shifts
    for i in range(shifts):
        if i > 0:
            y = np.concatenate([y[:, step:], y[:, 0:step]], axis=1)
        n_diag = min(y.shape[0], y_correct.shape[0])
        y_correct_slice = y_correct[-n_diag:]
        y_slice = y[-n_diag:]
        mask_slice = mask[-n_diag:]
        if mask_slice.shape[1] < y_slice.shape[1]:
            y_slice = y_slice[:, :mask_slice.shape[1]]
            y_correct_slice = y_correct_slice[:, :mask_slice.shape[1]]
        elif mask_slice.shape[1] > y_slice.shape[1]:
            mask_slice = mask_slice[:, :y_slice.shape[1]]
        y_correct_slice = gaussian_filter(y_correct_slice, 0.5)
        y_slice = gaussian_filter(y_slice, 0.5)
        r = pearsonr(
            y_correct_slice[mask_slice].flat,
            y_slice[mask_slice].flat
            ).statistic
        # also applying controls by correelating with reversed map:
        flipped_mask = np.flip(mask_slice, axis=1)
        r_reverse_control = pearsonr(
            y_correct_slice[mask_slice].flat,
            np.flip(y_slice, axis=1)[flipped_mask].flat
            ).statistic
        if i == 0:
            r_true = r # first one is wt prediction
        else:
            rs_control.append(r) # the rest are controls
        rs_control.append(r_reverse_control)
    rs_control = np.array(rs_control)
    return r_true, rs_control

def make_prediction_quality_mask(groundtruth, true_pred):
    step = 32
    mask = np.zeros(groundtruth.shape, dtype=bool)
    n_blocks = int(np.ceil(groundtruth.shape[1]/step))
    for i in range(n_blocks):
        r = np.corrcoef(groundtruth[:, i*step:(i+1)*step].flat,
                        true_pred[:, i*step:(i+1)*step].flat)[0,1]
        if r > 0.1:
            mask[:, i*step:(i+1)*step] = True
    return mask


def get_matrix(root, organism_list)
    all_rs = []
    significance = []
    all_between_replics = []
    for org in organism_list:
        folder = root+org+'/'
        groundtruth = np.load(folder+'groundtruth.npy')
        data_quality_mask = np.load(folder+'mask.npy')
        true_pred = np.load(folder+f'predicted_by_{org}_model.npy')
        length1 = groundtruth.shape[1]
        length2 = true_pred.shape[1]
        if not (0.99 < length1 / length2 < 1.01):
            print('Warning: shapes are different')
        length = min(length1, length2)
        groundtruth = groundtruth[:, :length]
        true_pred = true_pred[:, :length]
        data_quality_mask = data_quality_mask[:, :length]
        mask = data_quality_mask
        for org2 in l:
            true = np.load(folder+f'predicted_by_{org2}_model.npy')

            ys = np.array(true)
            r, control_rs = cross_correlate(ys, true_pred, mask)
            higher_rs = np.abs(control_rs) > np.abs(r)
            if np.mean(higher_rs) > 0.05:
                significance.append(False)
            else:
                significance.append(True)
            all_rs.append(r)

    all_rs = np.array(all_rs)
    significance = np.array(significance)
    all_rs_correct = all_rs.copy()
    all_rs_correct[~significance] = 0

    mtx = all_rs_correct.reshape(16,16).copy()
    plt.figure(figsize=(8,8))
    plt.imshow(mtx, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.xticks(np.arange(len(organism_list)), organism_list, rotation=90)
    plt.yticks(np.arange(len(organism_list)), organism_list)
    for i in range(len(l)):
        for j in range(len(l)):
            plt.text(i,j,f'{mtx[j,i]:.2f}', ha='center', va='center')
    return mtx
  
def tree(mtx, organism_list, mtx2=None):
  if mtx2 is not None:
    mtx = mtx*0.5 + mtx2*0.5
  mtx[np.diag_indices(len(mtx)]=1
  dissimilarity = 1 - (mtx+mtx.T)
  for i in range(len(organism_list)):
      dissimilarity[i,i]=0
  Z = linkage(squareform(dissimilarity), 'average')
  dendrogram(Z, labels=organism_list, orientation='left')
  plt.show()
