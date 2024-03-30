import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
import matplotlib.patheffects as PathEffects
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib

from . import plot_utils


def make_cube(space, center_pos, size):
    w, d, h = size
    x, y, z = space
    x0, y0, z0 = center_pos
    x0 = x0 - w//2
    y0 = y0 - d//2
    z0 = z0
    return (x0 < x) & (x < x0 + w+1) & (y0 < y) & \
           (y < y0 + d+1) & (z0 < z) & (z < z0 + h+1)

class ModelPlotter():
    '''Plots model erchitecture in 3D'''
    def __init__(self, n_residual_blocks, residual_block_filters, certain_len=False):
        res_block_h = n_residual_blocks * (sum(residual_block_filters)//64 + 6)
        canvas_size=(100, 20, 65+res_block_h)
        self.xlim, self.ylim, self.zlim = canvas_size
        self.space = np.indices(canvas_size)
        self.voxels = np.full(canvas_size, False)
        self.colors = np.empty(canvas_size, dtype=(float,4))
        self.certain_len = certain_len

    def build_1d_layer(self, ax, layer_index, total_layers,
                       row_index, size, filters,
                       kernel_size, dilation_rate, dropout_rate,
                       sum_arrow=False, certain_len=False):
        h = filters//64
        size *= 2
        center_pos = (self.xlim//2, 1, self.zlim-row_index-h)
        layer_size = (size, 1, h)
        layer = make_cube(self.space, center_pos, layer_size)
        if certain_len:
            output_shape = f'Out shape:\n{2**(size//2)} x {filters}'
        else:
            output_shape = f'Out shape:\n2^(n-{18-size//2}) x {filters}'
        text = f'C(k={kernel_size},d={dilation_rate}), B, R'
        if dropout_rate > 0:
            text += f', D({dropout_rate})'
        brightness = 1 - layer_index / total_layers
        self.colors[layer] = (brightness, brightness, 1, 0.5)
        self.voxels |= layer
        txt = ax.text(center_pos[0]+0.5, center_pos[1], center_pos[2]+h/2,
                text, zdir='x', ha='center', va='center', c='k', fontsize=9,
                zorder=1000000, style='italic')
        ax.text(center_pos[0]-size//2, center_pos[1], center_pos[2]+h/2,
                output_shape, zdir='x', style='italic',
                zorder=1000000)
        ax.text(self.xlim//2, 1, center_pos[2]-1.5,
                '↓', ha='center', va='center')
        if sum_arrow:
            '''ax.annotate('+', xy=(center_pos[0]+size//2,  center_pos[2]+h/2),
                        xytext =  (center_pos[0]+size//2, center_pos[2]-3),
                        arrowprops=dict(arrowstyle="->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="bar,fraction=0.3",))'''
            x_pos = center_pos[0]+size//2
            z_pos = center_pos[2]+h/2
            ax.plot([x_pos+1, x_pos+3, x_pos+3, x_pos+1],
                    [center_pos[1]]*4,
                    [z_pos, z_pos, z_pos-5, z_pos-5],
                    color='k',
                    linewidth=0.7)
            ax.plot([x_pos+1.5, x_pos+1, x_pos+1.5],
                    [center_pos[1]]*3,
                    [z_pos-4.5, z_pos-5, z_pos-5.5],
                    color='k',
                    linewidth=0.7)
            ax.text(x_pos + 4.5, center_pos[1], z_pos-4, '+', zdir='x', style='italic', fontsize=13)

        return row_index + h + 3

    def plot_fc(self, ax, row_index, latent_dim, n_cell_types):
        for i in range(n_cell_types):
            x_center = self.xlim//2 + (5*((i+1)//2))*(-1)**i - 0.5
            center_pos = (x_center, 1, self.zlim-row_index-1)
            layer_size = (2, 1, 1)
            layer = make_cube(self.space, center_pos, layer_size)
            self.colors[layer] = (0.3, 0, 0.3, 0.7)
            self.voxels |= layer
            txt = ax.text(x_center+0.5, center_pos[1], center_pos[2]+0.5,
                'FC', zdir='x', ha='center', va='center', c='w',
                zorder=1000000, style='italic')
            if i:
                txt = ax.text(x_center+0.5, center_pos[1], center_pos[2]+2.5,
                '↓', ha='center', va='center')
        output_shape = f'Out shape: {latent_dim}'
        ax.text(self.xlim//2 - (5*((i+1)//2))-3 , center_pos[1], center_pos[2]+0.5,
                output_shape, zdir='x', style='italic',
                zorder=1000000)

    def plot_fc_2(self, ax, row_index, shape, latent_dim):
        w,d,h = shape
        center_pos = (self.xlim-row_index-h, self.ylim//2, 5)
        layer = make_cube(self.space, center_pos, shape)
        self.colors[layer] = (0.3, 0, 0.3, 0.7)
        self.voxels |= layer
        output_shape = f'Out\nshape:\n{latent_dim}'
        ax.text(center_pos[0]-0.5, 1, 3,
                'FC', zdir='x', ha='center', va='center', c='k', fontsize=10,
                zorder=1000000, style='italic')
        ax.text(center_pos[0]+0.5, 14, 1,
                output_shape, zdir='x', style='italic', ha='center', fontsize=10,
                zorder=1000000)
        return row_index + h + 4

    def plot_fc_3(self, ax, row_index, shape, latent_dim):
        w,d,h = shape
        center_pos = (self.xlim-row_index-h, self.ylim//2, 5)
        layer = make_cube(self.space, center_pos, shape)
        self.colors[layer] = (0.3, 0, 0.3, 0.7)
        self.voxels |= layer
        output_shape = f'Out\nshape:\n{latent_dim}'
        ax.text(center_pos[0], 0, 9,
                'FC,\nReshape\nB, R', zdir='x', ha='center', va='center', c='k', fontsize=10,
                zorder=1000000, style='italic')
        ax.text(center_pos[0]+0.5, 16, -1,
                output_shape, zdir='x', style='italic', ha='center', fontsize=10,
                zorder=1000000)
        return row_index + h + 4


    def build_2d_layer(self, ax, layer_index, total_layers,
                       row_index, size, filters,
                       kernel_size, bn_and_relu=True):
        h = max(filters//16, 1)
        size *= 2
        center_pos = (self.xlim-row_index-h, self.ylim//2, 5)
        layer_size = (h, size, size//4)
        layer = make_cube(self.space, center_pos, layer_size)
        output_shape = f'Out\nshape:\n{2**(size//2-1)//4}x{2**(size//2-1)}x\nx{filters}'
        text2 = ',\nB, R' if bn_and_relu else ''
        text = f'C(k=\n=({kernel_size},{kernel_size}))'+text2
        brightness = 1 - layer_index / total_layers
        self.colors[layer] = (1, brightness, brightness, 0.5)
        self.voxels |= layer
        txt = ax.text(center_pos[0]+0.5, 1, center_pos[2]+size//8+4.5,
                text, zdir='x', ha='center', va='center', c='k', fontsize=10,
                zorder=1000000, style='italic')
        ax.text(center_pos[0]+0.5, 19, -3,
                output_shape, zdir='x', style='italic', ha='center', fontsize=10,
                zorder=1000000)
        return row_index + h + 6

    def plot(
            self,
            filters = [64, 128, 256, 512, 64],
            kernel_sizes = [11, 9, 5, 3, 3],
            dilation_rates = [1, 2, 16, 1, 1],
            n_residual_blocks = 6,
            residual_block_filters = [128, 128],
            residual_block_kernel_sizes = [5, 5],
            residual_block_dilation_rate_factors = [4, 4],
            dropout_rates = [0, 0.1, 0.15, 0.15, 0.15],
            residual_block_dropout_rates = [0, 0],
            latent_dim = 128,
            dna_shape=None,
            hic_shape=None,
            **kwargs
        ):
        fig = plt.figure(figsize=(25, 25))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(30, 70)

        pow = int(np.log2(dna_shape[0])) if self.certain_len else 18
        total_layers = 6 + (pow-10) + 2*n_residual_blocks


        h = self.build_1d_layer(
            ax, 0, total_layers, 2, pow-1,
            filters[0],  kernel_sizes[0], dilation_rates[0], dropout_rates[0],
            certain_len=self.certain_len
        )
        for i in range(3):
            h = self.build_1d_layer(
                ax, i+1, total_layers, h, pow-i-2,
                filters[1],  kernel_sizes[1], dilation_rates[1], dropout_rates[1],
                certain_len=self.certain_len
            )
        ax.text(self.xlim//2, 1, self.zlim-h-1,
                '↓\n', ha='center', va='center')
        ax.text(self.xlim//2, 1, self.zlim-h-2,
                'Repeating downsampling to size of 512',
                zdir='x', ha='center', va='center', style='italic')
        ax.text(self.xlim//2, 1, self.zlim-h-3.5,
                '↓', ha='center', va='center')
        h += 5
        index = pow-10+1
        j=0
        for j in range(n_residual_blocks):
            h = self.build_1d_layer(
                    ax, index+j, total_layers, h, 9,
                    residual_block_filters[0], residual_block_kernel_sizes[0],
                    residual_block_dilation_rate_factors[0]*(j+1),
                    residual_block_dropout_rates[0], sum_arrow=True,
                    certain_len=True
                )

            h = self.build_1d_layer(
                    ax, index+j+1, total_layers, h, 9,
                    residual_block_filters[1], residual_block_kernel_sizes[1],
                    residual_block_dilation_rate_factors[1]*(j+1),
                    residual_block_dropout_rates[0],
                    certain_len=True
                )
        index = index + j + 1
        h = self.build_1d_layer(
                ax, index+1, total_layers, h, 9,
                filters[2],  kernel_sizes[2], dilation_rates[2], dropout_rates[2],
                    certain_len=True
            )
        h = self.build_1d_layer(
                ax, index+2, total_layers, h, 9,
                filters[3],  kernel_sizes[3], dilation_rates[3], dropout_rates[3],
                    certain_len=True
            )
        h = self.build_1d_layer(
                ax, index+3, total_layers, h, 8,
                filters[4],  kernel_sizes[4], dilation_rates[4], dropout_rates[4],
                    certain_len=True
            )
        n_cell_types = hic_shape[-1] if hic_shape else 3
        self.plot_fc(ax, h, latent_dim, n_cell_types)
        h = 2
        h = self.build_2d_layer(ax, 1, 5, h, 8, 16, 4)
        h = self.build_2d_layer(ax, 2, 5, h, 8, 32, 2)
        h = self.build_2d_layer(ax, 3, 5, h, 7, 32, 2)
        h = self.build_2d_layer(ax, 4, 5, h, 6, 64, 2)
        h = self.build_2d_layer(ax, 5, 5, h, 5, 64, 2)
        h = self.plot_fc_2(ax, h, (1,3,1), str(latent_dim))
        h += 9
        h = self.plot_fc_3(ax, h, (4,8,2), '4x16x\nx64')
        h = self.build_2d_layer(ax, 5, 5, h, 5, 64, 2)
        h = self.build_2d_layer(ax, 4, 5, h, 6, 32, 2)
        h = self.build_2d_layer(ax, 3, 5, h, 7, 32, 2)
        h = self.build_2d_layer(ax, 2, 5, h, 8, 16, 2)
        h = self.build_2d_layer(ax, 1, 5, h, 8, 1, 4, False)

        ls = LightSource(10,70)
        ax.voxels(self.voxels, facecolors=self.colors, edgecolor=(0.5,0.5,0.5,0.5), linewidth=0.5, lightsource=ls)
        ax.set_aspect('equal')
        ax.axis('off')