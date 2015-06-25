from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np


def plot_scatter_heatmap(x, y, logx=False, logy=False, logbins=False, bins=100, cmap='jet', interpolation='none',
                         aspect='auto', origin='lower', colorbar=True):
    x = np.array(x)
    y = np.array(y)
    if logx:
        x = np.log10(x)
    if logy:
        y = np.log10(y)

    data_matrix, xedges, yedges = np.histogram2d(x, y, bins=bins)
    if logbins:
        data_matrix = np.log10(data_matrix + 1)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax = plt.imshow(data_matrix.T if origin is 'lower' else data_matrix, extent=extent, origin=origin, aspect=aspect,
                    interpolation=interpolation, cmap=cmap)

    tick_labels_log = lambda x: r'$\mathdefault{10^{' + str(x) + '}}$'
    if logx:
        ticks, _ = plt.xticks()
        ticks = range(int(ticks[0]), int(ticks[-1]) + 1)
        plt.xticks(ticks, map(tick_labels_log, map(int, ticks)))
    if logy:
        ticks, _ = plt.yticks()
        ticks = range(int(ticks[0]), int(ticks[-1]) + 1)
        plt.yticks(ticks, map(tick_labels_log, map(int, ticks)))

    plt.xlim([xedges[0], xedges[-1]])
    plt.ylim([yedges[0], yedges[-1]])
    if colorbar:
        cb = plt.colorbar(ax)
        if logbins:
            ticks = range(int(cb.vmin), int(cb.vmax) + 1)
            cb.set_ticks(ticks)
            cb.ax.set_yticklabels(map(tick_labels_log, map(int, ticks)))
    return ax
