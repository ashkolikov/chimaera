from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import gc
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from traitlets import default

from .gradients import IntegratedGradients, GradientDescent
from .evosearch import Evolution
from .motifs import Motif
from . import latent
from . import data_utils
from . import train_utils
from . import plot_utils
from . import motifs
from . import genes
from . import ism


class Chimaera():
    '''model for experiments with model'''
    def __init__(self, model):
        self.model = model
        self.data = model.data
        self.int_grad = IntegratedGradients(model)
        self.grad_desc = GradientDescent(model)
        self.evo = Evolution(model)
        self.vec_dict = self._make_basic_vecs()

    def get_dna(self, region, one_hot=False):
        return self.data.get_dna(region, seq=not one_hot)

    def get_hic(self, region, experiment_index=0):
        hic = self.data.get_hic(
            region, 
            plot=True,
            experiment_index=experiment_index
        )

    def mask_to_vec(self, mask):
        return latent.mask_to_vec(self.model, mask)

    def _make_basic_vecs(self):
        vec_dict = dict()
        vec_dict['insulation'] = self.mask_to_vec(latent.insulation_mask())
        vec_dict['fountain'] = self.mask_to_vec(latent.fountain_mask())
        vec_dict['loop'] = self.mask_to_vec(latent.loop_mask())
        return vec_dict

    def add_vec(self, mask, name):
        self.vec_dict[name] = self.mask_to_vec(mask)

    def predict_region(
            self,
            region,
            mutations=None,
            new_sequences=None,
            plot=True,
            exclude_imputed=True,
            equal_scale=True,
            shifts=2,
            edge_policy='empty',
            return_mask=False,
        ):
        '''Predicts fragment of any length specified by genomic coordinates'''
        if plot:
            hic = self.data.get_hic(region, edge_policy=edge_policy)
            mask = self.data.get_mask(region, edge_policy=edge_policy)
        if mutations is not None:
            parsed_region = self.data._parse_region(region)
            chrom, start, end = parsed_region
            size = end-start
            start -= self.data.offset
            end += self.data.dna_len # add some extra DNA because we can
                                     # predict only full-size fragments 
            region = f'{chrom}:{start}-{end}'
            mutated_dna = ism.mutate(
                self.data,
                region,
                mutations, 
                new_sequences,
                edge_policy=edge_policy
            )
            # data.get_dna_loader() needs specified region but we have already
            # sliced region of interest so we will specify the whole region
            # (except offsets)
            chrom = 'mutant_fragment'
            start_in_mutated_dna = self.data.offset
            end_in_mutated_dna = self.data.offset + size
            region_new = f'{chrom}:{start_in_mutated_dna}-{end_in_mutated_dna}'
            dna_loader, _, remainder_frac = self.data.get_dna_loader(
                region_new,
                mutated_dna,
                edge_policy=edge_policy
            )
        else:
            dna_loader, parsed_region, remainder_frac = self.data.get_dna_loader(
                region, 
                edge_policy=edge_policy
            )
        spare_frac = 1 - remainder_frac
        if shifts == 1:
            pred = self.model.predict(dna_loader, strand='both')
            pred = np.concatenate(pred, axis=1)
        else:
            loaders = self.model._apply_slides(dna_loader, shifts)
            regions = []
            for loader in loaders:
                regions += loader.regions
            regions = sorted(regions, key=lambda x:x[1])
            dna_loader.regions = regions
            pred = self.model.predict(dna_loader, strand='both')
            pred = train_utils.combine_shifts(pred, shifts)
        
        spare_hic_length = int(spare_frac * self.data.map_size)
        if spare_hic_length:
            pred = pred[:, :-spare_hic_length]
        if plot:
            if pred.shape[1] != hic.shape[1]:
                zoom_rate = hic.shape[1] / pred.shape[1]
                pred = zoom(pred, (1,zoom_rate,1))
            if exclude_imputed:
                if mutations is None:
                    hic = hic.copy()
                    hic[mask > 0] = np.nan
                    pred[mask > 0] = np.nan
            chrom, start, end = parsed_region
            for i in range(len(self.data.experiment_names)):
                _, ax = plt.subplots(ncols=1, nrows=2, figsize=((hic.shape[1]/25, 2)))
                if equal_scale:
                    vmin = min(np.nanmin(hic), np.nanmin(pred))
                    vmax = max(np.nanmax(hic), np.nanmax(pred))
                else:
                    vmin, vmax = None, None
                self.data.plot_annotated_map(hic_map=hic,
                                             chrom=chrom, start=start, end=end,
                                             experiment_index = i,
                                             axis=None,
                                             ax=ax[0],
                                             vmin=vmin,
                                             vmax=vmax,
                                             show=False,
                                             )

                self.data.plot_annotated_map(hic_map=np.flip(pred, axis=0),
                                             chrom=chrom, start=start, end=end,
                                             experiment_index = i,
                                             ax=ax[1],
                                             vmin=vmin,
                                             vmax=vmax,
                                             axis='x',
                                             show_position=False,
                                             )
        else:
            if return_mask:
                mask = self.data.get_mask(region, edge_policy=edge_policy)
                return pred, mask
            else:
                return pred

    def scan_projections(self,
                         vecs=None,
                         central_bins_only=True,
                         vec_names=None,
                         step=8,
                         nan_threshold=0.1,
                         skip_empty_center=True,
                         empty_bins_offset=1,
                         metric='projection',
                         chroms=None,
                         return_latent=False):
        if vecs is None:
            vecs = list(self.vec_dict.values())
            vec_names = list(self.vec_dict.keys())
        return latent.scan_projections(
            self.model,
            vecs=vecs,
            central_bins_only=central_bins_only,
            vec_names=vec_names,
            step=step,
            nan_threshold=nan_threshold,
            skip_empty_center=skip_empty_center,
            return_latent=return_latent,
            metric=metric,
            chroms=chroms,
            empty_bins_offset=empty_bins_offset,
        )

    def explore_latent_space(
            self,
            central_point='mean', # center is mean of many dots or a sigle one
            vecs='random', # random vecs in the latent space or real map representations
            n_vecs=10,
            max_r=6,
            channel=0, # latent space is the same for all channels so this arg
            #            is not imortant
            n_spheares=13
        ):

        # make vectors for insulation and loop
        ins_vec = self.mask_to_vec('insulation')
        ins_vec /= np.linalg.norm(ins_vec)
        loop_vec = self.mask_to_vec('loop')
        loop_vec /= np.linalg.norm(loop_vec)
        special_vecs = np.concatenate([ins_vec, loop_vec])


        rs = np.linspace(0, max_r, n_spheares+1)
        rs = rs[1:]
        if n_vecs > len(self.data.y_test) and vecs=='real':
            print(f"WARNING: 'real' vecs are maps from the test sample and \
their amount is {len(self.data.y_test)} so more vecs can't be shown")
            n_vecs = len(self.data.y_test)

        if isinstance(central_point, int):
            central_maps = self.data.y_test[central_point]
        elif central_point == 'random':
            i = np.random.randint(0, len(self.data.y_test))
            central_maps = self.data.y_test[i]
        else:
            central_maps = self.data.y_test[:n_vecs]
        maps_for_channel = central_maps[..., channel][...,None]
        central_points = self.model.hic_to_latent(maps_for_channel)
        central_maps_pred = self.model.latent_to_hic(central_points)

        if vecs == 'random':
            vecs_cloud = np.random.normal(0,1,(n_vecs, self.model.latent_dim))
        elif vecs == 'real':
            if central_point == 'mean':
                maps = self.data.y_test[n_vecs:n_vecs*2]
            else:
                maps = self.data.y_test[:n_vecs]
                maps = maps[np.all(maps!=central_maps, axis=(1,2,3))]
            maps_for_channel = maps[..., channel][...,None]
            vecs_cloud = self.model.hic_to_latent(maps_for_channel)

        corrs = []
        corrs_special = []
        maps_special = []
        for r in rs:
            c = []
            c_special = []
            m_special = []
            for central_map, central_point in zip(central_maps_pred, central_points):
                central_point = central_point[None, ...]
                directions = vecs_cloud
                directions /= np.linalg.norm(directions, axis=1)[:,None]
                vecs = directions * r
                vecs += central_point
                maps_from_vecs = self.model.latent_to_hic(vecs)
                for distant_map in maps_from_vecs:
                    c.append(np.corrcoef(distant_map.flat, central_map.flat)[0,1])
                if special_vecs is not None:
                    directions = special_vecs
                    directions /= np.linalg.norm(directions, axis=1)[:,None]
                    vecs = directions * r
                    vecs += central_point
                    maps_from_special_vecs = self.model.latent_to_hic(vecs)
                    cc=[]
                    for distant_map in maps_from_special_vecs:
                        cc.append(np.corrcoef(distant_map.flat, central_map.flat)[0,1])
                    c_special.append(cc)
                    m_special.append(maps_from_special_vecs)
            corrs.append(c)
            corrs_special.append(c_special)
            maps_special.append(m_special)
        plot_utils.plot_latent_space(
            rs, np.array(corrs), np.array(corrs_special),
            special_vecs, maps_special)

    def analyze_features(self,
                         vecs=None,
                         central_bins_only=True,
                         vec_names=None,
                         step=8,
                         nan_threshold=0.1,
                         skip_empty_center=True):
        dfs = self.scan_projections(vecs, central_bins_only,
                                     vec_names, step,
                                     nan_threshold, skip_empty_center)
        return latent.analyze_projections(self.data, dfs)

    def ig(self, region=None, fragment_index=None, experiment_index=0, experiment_name=None, steps=20,
           peak_width=20, n_peaks=10, between_peaks=10, annotate_peaks=True, annotation=None,
           saving_dir=None, return_gradients=False):
        if experiment_name:
            experiment_index = self.data.experiment_names.index(experiment_name)
        if fragment_index is not None or region is not None:
            _, profile = self.int_grad.integrated_gradients(
                fragment_index=fragment_index,
                region=region,
                experiment_index=experiment_index,
                steps=steps,
                peak_width=peak_width,
                n_peaks=n_peaks,
                between_peaks=between_peaks,
                plot=True,
                precise=True,
                annotate_peaks=annotate_peaks,
                annotation=annotation,
                )
            if return_gradients:
                return profile
        else:
            tables = []
            profiles = {}
            for fragment_index in tqdm(range(len(self.data.y_test))):
                table, profile = self.int_grad.integrated_gradients(
                    fragment_index=fragment_index,
                    experiment_index=experiment_index,
                    steps=steps,
                    peak_width=peak_width,
                    n_peaks=n_peaks,
                    between_peaks=between_peaks,
                    plot=False,
                    precise=False,
                    )
                tables.append(table)
                if return_gradients:
                    region = self.data.x_test.regions[fragment_index]
                    chrom, start, end = region
                    region = (chrom, start+self.data.offset, end-self.data.offset)
                    profiles[region] = profile
            df = pd.concat(tables, ignore_index=True)
            if saving_dir is not None:
                organism_name = self.data.organism.replace(' ', '')
                file_name = 'chimaera_ig_output_' + organism_name +'.csv'
                file_name = os.path.join(saving_dir, file_name)
                df.to_csv(file_name, index=False)
                if return_gradients:
                    return profiles
            else:
                if return_gradients:
                    return df, profiles
                else:
                    return df

    def mean_gradients(self, table, profiles, name='Target'):
        target_means = []
        control_means = []
        for region, profile in profiles.items():
            t = genes.make_subtable(table, [region])
            t.loc[:, 'start'] -= region[1]
            t.loc[:, 'end'] -= region[1]
            control = np.flip(profile)
            for start, end in zip(t.start, t.end):
                target_means.append(profile[start:end].mean())
                control_means.append(control[start:end].mean())
        target = np.array(target_means)
        target = target[~np.isnan(target)]
        control = np.array(control_means)
        control = control[~np.isnan(control)]
        plot_utils.compare_mean_gradients(target, control, name)

    def grad_search(
            self,
            vec,
            size=40,
            n_repeats=1,
            epochs=15,
            number=1,
            saving_dir=None,
            experiment_index=0,
            experiment_name=None,
            sd_threshold=0.33
        ):
        if experiment_name:
            experiment_index = self.data.experiment_names.index(experiment_name)
        if isinstance(vec, str):
            vec = self.vec_dict[vec]
        if number == 1:
            self.grad_desc.reset()
            return self.grad_desc.gradient_descent(
                                    vec,
                                    size=size,
                                    n_repeats=n_repeats,
                                    experiment_index=experiment_index,
                                    epochs=epochs)
        else:
            if saving_dir is None:
                raise ValueError('In case of multiple runs you should pass \
saving_dir arg')
            organism_name = self.data.organism.replace(' ', '')
            file_name = 'chimaera_gd_output_' + organism_name +'.txt'
            file_name = os.path.join(saving_dir, file_name)
            bad_counter = 0
            threshold = self.model.sd * sd_threshold
            for i in range(number):
                self.grad_desc.reset()
                if bad_counter==3 and i<=5:
                    print('Seems too little significant motifs can be found with such params')
                    return
                motif = self.grad_desc.gradient_descent(
                                    vec,
                                    size=size,
                                    n_repeats=n_repeats,
                                    experiment_index=experiment_index,
                                    epochs=epochs)
                motif = Motif(seq=motif.consensus)
                check = self.mean_motif_effect(
                    motif, 
                    '>>>>>',
                    number=10,
                    plot=False,
                    experiment_index=experiment_index,
                    sample='val',
                    normalize=True)
                if np.abs(check).max() < threshold:
                    bad_counter += 1
                    print("Motif doesn't cause required prediction change")
                else:
                    seq = motif.consensus
                    with open(file_name, 'a') as file:
                        file.write(seq+'\n')
                    
                

    def evo_search(
            self,
            vec,
            size=20,
            composition='>>>>>>>',
            alpha=5,
            generations=20,
            population_size=100,
            between_insertions=10,
            mutation_rate=0.1,
            elite=0.1,
            shift_rate=0.1,
            n_crossovers=2,
            experiment_index=0,
            experiment_name=None
        ):
        if experiment_name:
            experiment_index = self.data.experiment_names.index(experiment_name)
        if isinstance(vec, str):
            vec = self.vec_dict[vec]
        self.evo.evolution(
            target=vec,
            alpha=alpha,
            composition=composition,
            experiment_index=experiment_index,
            between_insertions=between_insertions,
            mutation_rate=mutation_rate,
            n_crossovers=n_crossovers,
            elite=elite,
            shift_rate=shift_rate,
            generations=generations,
            population_size=population_size,
            length=size)
        return self.evo.get_motif()

    def mean_motif_effect(
            self,
            motif,
            composition='>',
            long_spacer_length='auto',
            number=100,
            strand='one',
            between_insertions=20,
            experiment_index=0, 
            experiment_name=None,
            fixed_scale=True,
            sample='val',
            plot=True,
            normalize=True
        ):
        if experiment_name:
            experiment_index = self.data.experiment_names.index(experiment_name)
        return motifs.mean_motif_effect(
            self.model,
            motif,
            composition=composition,
            number=number,
            strand=strand,
            long_spacer_length=long_spacer_length,
            between_insertions=between_insertions,
            experiment_index=experiment_index,
            fixed_scale=fixed_scale,
            sample=sample,
            plot=plot,
            normalize=normalize
        )

    def check_motif(self, motif, vec, experiment_index=0, experiment_name=None, name=None):
        if experiment_name:
            experiment_index = self.data.experiment_names.index(experiment_name)
        if isinstance(vec, str):
            vec = self.vec_dict[vec]
        return motifs.check_motif(self.model, motif, vec, experiment_index=experiment_index, name=name)

    def gene_composition(
            self,
            gene_table,
            max_len='auto',
            min_len='auto',
            min_score=-np.inf,
            max_score=np.inf,
            upstream=0,
            downstream=0,
            long_spacer_length='auto',
            between_genes='auto',
            n_replicates=128,
            composition='>>><<<_>>><<<',
            experiment_index=0,
        ):
        return genes.gene_composition(
            self.model,
            gene_table,
            max_len=max_len,
            min_len=min_len,
            upstream=upstream,
            downstream=downstream,
            min_score=min_score,
            max_score=max_score,
            long_spacer_length=long_spacer_length,
            between_genes=between_genes,
            n_replicates=n_replicates,
            composition=composition,
            experiment_index=0
        )


    def scan_motif(self, motif, threshold=0):
        table = {'chrom':[], 'start':[], 'end':[], 'strand':[], 'score':[]}
        for chrom in self.data.chromnames:
            whole_chrom = self.data.DNA[chrom]
            for start in range(0, len(whole_chrom), 1000000-len(motif)):
                dna = whole_chrom[start:start+1000000]
                dna = data_utils.one_hot(dna)
                profile_forw = motifs.scan(dna, motif.pfm)
                profile_rc = motifs.scan(dna, motif.rc())
                strand = profile_forw > profile_rc
                profile = np.maximum(profile_forw, profile_rc)
                index = np.where(profile > threshold)
                table['score'] += list(profile[index])
                table['strand'] += [{True:'+', False:'-'}[i] for i in strand[index]]
                table['chrom'] += [chrom]*len(index[0])
                table['start'] += list(index[0]+start)
                table['end'] += list(index[0]+len(motif))

        table = pd.DataFrame(table)
        table['end'] = table.start + len(motif)
        table.data_name = motif.name
        return table

    def modify_all(self, table, modification, threshold=None, regions=None):
        new_genome = ism.modify_genome(self.data, table, modification,
                                   threshold=threshold)
        if regions is None:
            regions = self.data.x_test.regions + self.data.x_val.regions
        loader = data_utils.MutantDNALoader(self.data, new_genome, regions)
        gen = data_utils.DNAPredictGenerator(loader, batch_size=self.model.batch_size)
        y = self.model.predict(gen)
        del new_genome
        gc.collect()
        return y

    def change_all_seqs(self, motif=None, threshold=None, table=None, regions=None, modification='~'):
        if table is None:
            table = self.scan_motif(motif)
        if regions is None:
            regions = self.data.x_test.regions + self.data.x_val.regions
            table = genes.make_subtable(table, regions)
        else:
            table = genes.make_subtable(table, regions)
        y_wt_loader = data_utils.DNALoader(self.data, regions)
        ys_wt = self.model.predict(
            data_utils.DNAPredictGenerator(
                y_wt_loader,
                batch_size=self.model.batch_size
                )
            )
        ys_mut = self.modify_all(table, modification, threshold=threshold, regions=regions)
        # control is a table with positions inverted relative to the chromosome coordinates
        # this procedure saves distribution of length and distances but makes mapping almost random
        control_table = genes.flip_annotation(table)
        ys_control = self.modify_all(control_table, modification, threshold=threshold, regions=regions)
        return ys_wt, ys_mut, ys_control
