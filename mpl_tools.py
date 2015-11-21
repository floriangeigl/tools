from __future__ import division, print_function
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import os


def plot_set_limits(values=None, min_v=None, max_v=None, axis=None, ax=None, offset_percent=5):
    if values is not None:
        values = np.array(values)
        if min_v is None:
            min_v = values.min()
        if max_v is None:
            max_v = values.max()
    assert min_v is not None and max_v is not None
    range_v = max_v - min_v
    offset = range_v * (offset_percent / 100)
    min_v -= offset
    max_v += offset
    if axis is not None:
        axis = axis.lower()
    if ax is None:
        if axis is None:
            plt.xlim([min_v, max_v])
            plt.ylim([min_v, max_v])
        else:
            if 'x' in axis:
                plt.xlim([min_v, max_v])
            elif 'y' in axis:
                plt.ylim([min_v, max_v])
            else:
                print('axis:', ax, 'unknown. use "x" or "y"')
    else:
        if axis is None:
            ax.set_xlim([min_v, max_v])
            ax.set_ylim([min_v, max_v])
        else:
            if 'x' in axis:
                ax.xlim([min_v, max_v])
            elif 'y' in axis:
                ax.ylim([min_v, max_v])
            else:
                print('axis:', ax, 'unknown. use "x" or "y"')


def plot_legend(ax, filename, font_size=None, figsize=(16, 3), ncols=None, nrows=None, crop=True,
                legend_name_idx=None, legend_name_style='bf', labels_right_to_left=True):
    default_font_size = matplotlib.rcParams['font.size']
    if font_size is not None:
        matplotlib.rcParams.update({'font.size': font_size})
    f2 = plt.figure(figsize=figsize)
    handles, labels = ax.get_legend_handles_labels()
    if ncols is None:
        num_labels = len(labels)
        if nrows is not None:
            ncols = int(num_labels / nrows)
            if num_labels % nrows != 0:
                ncols += 1
        else:
            ncols = num_labels
    if legend_name_idx is not None:
        if legend_name_style == 'bf':
            labels[legend_name_idx] = r'\textbf{' + labels[legend_name_idx] + '}'
        elif legend_name_style == 'it':
            labels[legend_name_idx] = r'\textit{' + labels[legend_name_idx] + '}'
        elif legend_name_style == 'bfit' or legend_name_style == 'itbf':
            labels[legend_name_idx] = r'\textit{\textbf{' + labels[legend_name_idx] + '}}'
        else:
            labels[legend_name_idx] = labels[legend_name_idx]
        if legend_name_style is not None:
            use_text_default = matplotlib.rcParams['text.usetex']
            matplotlib.rcParams['text.usetex'] = True
    if ncols > 1 and labels_right_to_left:
        sorted_handle_labels = list()
        for i in range(ncols):
            for idx, h_l in enumerate(zip(handles, labels)):
                if idx % ncols == i:
                    sorted_handle_labels.append(h_l)
        handles, labels = zip(*sorted_handle_labels)

    f2.legend(handles, labels, loc='center', ncol=ncols)
    plt.savefig(filename, bbox_tight=True)
    plt.close('all')
    if filename.endswith('.pdf') and crop:
        crop_pdf(filename)
    if font_size is not None:
        matplotlib.rcParams.update({'font.size': default_font_size})
    if legend_name_idx is not None and legend_name_style is not None:
        matplotlib.rcParams['text.usetex'] = use_text_default


def save_n_crop(fn):
    plt.savefig(fn)
    if fn.endswith('.pdf'):
        crop_pdf(fn)


def crop_pdf(fn, out_filename=None, bgtask=True):
    out_filename = fn if out_filename is None else out_filename
    return os.system('pdfcrop ' + fn + ' ' + out_filename + ' > /dev/null 2>&1' + (' &' if bgtask else ''))


def plot_scatter_heatmap(x, y, logx=False, logy=False, logbins=False, bins=100, cmap='jet', interpolation='none',
                         aspect='auto', origin='lower', colorbar=True, replace_not_finite=True, axis_range=None,
                         cb_range=None, **kwargs):
    x = np.array(x)
    y = np.array(y)
    if logx:
        x = np.log10(x)
        if axis_range is not None:
            axis_range[0] = np.log10(np.array(axis_range[0]))
    if logy:
        y = np.log10(y)
        if axis_range is not None:
            axis_range[1] = np.log10(np.array(axis_range[1]))

    data_matrix, xedges, yedges = np.histogram2d(x, y, bins=bins, range=axis_range)
    if logbins:
        data_matrix = np.log10(data_matrix)
    if replace_not_finite:
        data_matrix_filter = np.invert(np.isfinite(data_matrix))
        data_matrix[data_matrix_filter] = 0.

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax = plt.imshow(data_matrix.T if origin is 'lower' else data_matrix, extent=extent, origin=origin, aspect=aspect,
                    interpolation=interpolation, cmap=cmap, **kwargs)

    tick_labels_log = lambda x: r'$\mathdefault{10^{' + str(x) + '}}$'
    if logx:
        ticks, _ = plt.xticks()
        ticks = range(int(ticks[0]), int(ticks[-1]) + 1)
        if len(ticks) > 6:
            step = int(len(ticks)/6)
            ticks = ticks[0::step]
        plt.xticks(ticks, map(tick_labels_log, map(int, ticks)))
    if logy:
        ticks, _ = plt.yticks()
        ticks = range(int(ticks[0]), int(ticks[-1]) + 1)
        if len(ticks) > 6:
            step = int(len(ticks)/6)
            ticks = ticks[0::step]
        plt.yticks(ticks, map(tick_labels_log, map(int, ticks)))

    plt.xlim([xedges[0], xedges[-1]])
    plt.ylim([yedges[0], yedges[-1]])
    if colorbar:
        cb = plt.colorbar(ax)
        ticks = []
        if cb_range is not None:
            if len(cb_range) == 2:
                ticks = range(cb_range[0], cb_range[1])
            else:
                ticks = cb_range
            cb.set_ticks(ticks)
        elif logbins:
            ticks = range(int(cb.vmin), int(cb.vmax) + 1)
            cb.set_ticks(ticks)
        if logbins:
            cb.ax.set_yticklabels(map(tick_labels_log, map(int, ticks)))
    return ax
