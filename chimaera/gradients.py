import torch
import numpy as np
from pandas import DataFrame
from .motifs import Motif
from .data_utils import DNAPredictGenerator
from .plot_utils import plot_ig


class GradientDescent():
    '''Searching for the best substitution in the input sequence for some \
feature appearance in prediction using gradient descent'''
    def __init__(self, container, size=30):
        self.container = container
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.site = self._init_site(size)

    def _init_site(self, size):
        site = np.random.random((1, 4, size))
        site = torch.Tensor(site).to(self.device)
        site.requires_grad = True
        return site

    def reset(self):
        self.site = self._init_site(self.size)

    def gradient_descent(
            self,
            vec,
            size=30,
            n_repeats=1,
            experiment_index=0,
            epochs=100
        ):
        if size != self.size:
            self.size = size
            self.site = self._init_site(size)
        n = n_repeats
        v = vec/np.linalg.norm(vec)
        target = torch.Tensor(v).to(self.device)
        for p in self.container.dna_encoder.model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam([self.site], 1)
        self.container.dna_encoder.model.eval()
        gen = DNAPredictGenerator(self.container.data.x_test,
                                  batch_size=self.container.batch_size)
        gen.shuffle()
        for epoch in range(epochs):
            inp = gen[0] # use always the first batch (after shuffling once)
            inp = inp.transpose((0,2,1))
            optimizer.zero_grad()
            part_5_prime = torch.Tensor(inp[:, :, :self.container.data.dna_len//2-(size*n)//2]).to(self.device)
            part_3_prime = torch.Tensor(inp[:, :, self.container.data.dna_len//2+(size*n)//2:]).to(self.device)
            site_ = torch.concatenate([self.site]*len(inp), axis=0) # copy for batch
            input_ = torch.concatenate([part_5_prime]+ [site_]*n + [part_3_prime], axis=2)
            input_ = (torch.nn.Softmax(1)(input_)-0.1749)/(0.4754-0.1749) # rough scaling to [0,1]
            prediction = self.container.dna_encoder.model(input_)[experiment_index]
            losses = [-torch.dot(prediction[i], target[0]) for i in range(len(inp))]
            loss_value = losses[0]
            for i in losses:
                loss_value += i
            loss_value = loss_value / len(losses)
            loss_value.backward()
            optimizer.step()
            print(f'Epoch {epoch+1} loss: {float(loss_value):.5}')
        return self.get_site()

    def get_site(self):
        matrix = torch.nn.Softmax(0)(self.site[0,:,:]).cpu().detach().numpy().T
        return Motif(pfm=matrix)


class IntegratedGradients():
    '''Prediction interpretation using integrated gradients method'''
    def __init__(self, model, steps=10):
        self.steps = steps
        self.Model = model
        self.offset = self.Model.data.offset

    def interpolate(self, x):
        alphas = torch.linspace(0.0, 1.0, self.steps+1).reshape((-1,1,1,1)).cuda()
        x = torch.Tensor(x.transpose((0,2,1))).cuda()
        return alphas * x

    def compute_gradients(self, inputs, experiment_index):
        grads = []
        model = self.Model.dna_encoder.model
        model.eval()
        for input in inputs:
            input.requires_grad = True
            output = model(input)[experiment_index]
            model.zero_grad()
            output.sum().backward()
            gradient = input.grad.detach().cpu().numpy()[0]
            grads.append(gradient.transpose((1,0)))
        return np.array(grads)

    def integral_approximation(self, gradients):
        grads = (gradients[:-1] + gradients[1:]) / 2
        integrated_gradients = np.mean(grads, axis=0)
        return integrated_gradients

    def integrated_gradients(self,
                             region=None,
                             fragment_index=None,
                             experiment_index=0,
                             annotate_peaks=True,
                             steps=10,
                             peak_width=20,
                             n_peaks=10,
                             between_peaks=10,
                             plot=True,
                             precise=True,
                             annotation=None):
        if steps is not None:
            self.steps = steps
        if fragment_index is not None:
            chrom, start, end = self.Model.data.x_test.regions[fragment_index]
        else:
            chrom, start, end = self.Model.data._parse_region(region)

        if start < self.Model.data.dna_len//2 or end > self.Model.data.chromsizes[chrom] - self.Model.data.dna_len//2:
            if plot:
                raise ValueError('Integrated gradients need more space around \
the fragment of analysis. Select regions more distant from chromoseme edges')
            else:
                print(f"WARNING: fragment {chrom}:{start}-{end} is too close to\
 chrom edge - it can't be analyzed")
                return [], DataFrame({'chrom':[],'start':[],'end':[],'score':[]})
        if end-start != self.Model.data.dna_len:
            end = start + self.Model.data.dna_len
            print(f'WARNING: fragment size should be {self.Model.data.dna_len}')

        if annotation is not None:
            if not (isinstance(annotation, list) or isinstance(annotation, tuple)):
                annotation = [annotation]
            for j in range(len(annotation)):
                name = annotation[j].data_name
                annotation[j] = annotation[j][annotation[j].chrom == chrom]
                annotation[j] = annotation[j][annotation[j].end >= start + self.offset]
                annotation[j] = annotation[j][annotation[j].start <= end - self.offset].copy()
                annotation[j].start -= start
                annotation[j].end -= start
                annotation[j].data_name = name

        step = self.Model.data.dna_len // 4
        ig_shifts = [np.zeros(self.Model.data.dna_len)]
        if precise:
            range_ = [-step*2, -step, 0, step, step*2]
        else:
            range_ = [0, step]
        for shift in range_:
            current_start , current_end = start + shift, end + shift
            dna_region = f'{chrom}:{current_start}-{current_end}'
            x = self.Model.data.get_dna(dna_region)
            interpolated_inputs = self.interpolate(x)
            path_gradients = self.compute_gradients(interpolated_inputs, experiment_index)
            ig = self.integral_approximation(path_gradients)
            ig = np.abs(np.sum(ig, axis=1))
            if shift != 0:
                empty = np.full(np.abs(shift), np.nan)
                if shift < 0:
                    ig = np.concatenate([ig[-shift:], empty])
                else:
                    ig = np.concatenate([empty, ig[:-shift]])
            ig_shifts.append(ig)
        ig = np.nanmax(ig_shifts, axis=0)
        ig = ig[self.offset:-self.offset]

        peak_table = self.get_seqs(ig, chrom, start, peak_width, n_peaks, between_peaks)

        if plot:
            hic_region = f'{chrom}:{start+self.offset}-{end-self.offset}'
            y = self.Model.data.get_hic(hic_region)
            if y.shape[1] == self.Model.data.map_size:
                y = self.Model.denoise(y[None,...])
            elif y.shape[1] == self.Model.data.map_size+1:
                y = self.Model.denoise(y[None,:-1,...])

            dna_region = f'{chrom}:{start}-{end}'
            x = self.Model.data.get_dna(dna_region)
            y_pred = self.Model.predict(x, strand='both')
            plot_ig(ig,
                    y,
                    y_pred,
                    peak_table,
                    data=self.Model.data,
                    annotate_peaks=annotate_peaks,
                    region=hic_region,
                    experiment_index=experiment_index,
                    annotation=annotation)
        return peak_table, ig

    def get_seqs(self, ig, chrom, start, peak_width, n_peaks, between_peaks):
        peak_half_width = peak_width // 2
        values, seqs, regions = [], [], []
        # make a copy for iteratively removing peaks with surrounding
        arr = ig.copy()
        for j in range(n_peaks):
            peak = arr.argmax()
            peak_value = arr[peak]
            # set peak and surrounding to 0
            arr[max(peak-peak_half_width*between_peaks, 0):
                peak+peak_half_width*between_peaks] = 0
            peak_start = peak - peak_half_width + start + self.offset
            peak_end = peak + peak_half_width + start + self.offset
            seq = self.Model.data.get_dna(f"{chrom}:{peak_start}-{peak_end}",
                                          seq=True)
            values.append(peak_value)
            seqs.append(seq)
            regions.append((chrom, peak_start, peak_end))
        peak_table = DataFrame({'chrom':[i[0] for i in regions],
                                   'start':[i[1] for i in regions],
                                   'end':[i[2] for i in regions],
                                   'score':values,
                                   'seq':seqs})
        return peak_table