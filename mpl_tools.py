from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import os


def plot_legend(ax, filename, font_size=None, figsize=(16, 3), ncols=None):
    default_font_size = matplotlib.rcParams['font.size']
    if font_size is not None:
        matplotlib.rcParams.update({'font.size': font_size})
    f2 = plt.figure(figsize=figsize)
    handles, labels = ax.get_legend_handles_labels()
    num_labels = len(labels)
    f2.legend(handles, labels, loc='center', ncol=num_labels if ncols is None else ncols)
    plt.savefig(filename, bbox_tight=True)
    if filename.endswith('.pdf'):
        crop_pdf(filename)
    plt.show()
    plt.close('all')
    if font_size is not None:
        matplotlib.rcParams.update({'font.size': default_font_size})


def crop_pdf(fn, out_filename=None, bgtask=True):
    out_filename = fn if out_filename is None else out_filename
    return os.system('pdfcrop ' + fn + ' ' + out_filename + ' &> /dev/null' + (' &' if bgtask else ''))


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
