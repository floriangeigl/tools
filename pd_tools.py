from __future__ import division
from sys import platform as _platform
import matplotlib
if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import printing
import basics


def set_output_fmt(max_colwidth=100000, width=100000, max_rows=10000):
    pd.set_option('display.max_colwidth', max_colwidth)
    pd.set_option('display.width', width)
    pd.set_option('display.max_rows', max_rows)


def print_tex_table(df, cols=None, mark_min=True, mark_max=True, digits=6):
    result_str = ''
    if cols is not None:
        df = df[cols].copy()
    else:
        df = df.copy()
    width = len(df.columns)
    col_fmt = '{l|' + '|'.join('c' * width) + '}'
    result_str += '\\begin{tabular*}{\\linewidth}' + col_fmt + '\n\\toprule\n'
    header = ' & ' + ' & '.join(df.columns) + '\\\\\\midrule\n'
    for col in df.columns:
        col_min, col_max = df[col].min(), df[col].max()
        # print len(df[col])
        color_dict = {key: color + '!' + str(val)[:3] for key, val, color in
                      zip(sorted(np.array(df[col])), [40, 30, 20, 20, 30, 40],
                          ['blue', 'blue', 'blue', 'red', 'red', 'red'])}
        # print color_dict
        df[col] = df[col].apply(lambda x: '\\cellcolor{' + color_dict[x] + '} ' + (
            "$\\bm{" + str(x == col_min) + str(x)[:digits] + "}$" % x if (x == col_min or x == col_max) else '$' + str(
                x)[:digits] + '$'))
        if mark_min:
            df[col] = df[col].apply(lambda x: x.replace('False', '\\wedge'))
        if mark_max:
            df[col] = df[col].apply(lambda x: x.replace('True', '\\vee'))
    #result_str += str(df)
    result_str += header
    for idx, i in df.iterrows():
        row_name = idx
        if '_' in row_name:
            row_name = ' '.join([x[:3] + '.' for x in idx.split('_')])
        result_str += row_name + ' & ' + ' & '.join(i) + ' \\\\\n'
    result_str += '\\bottomrule\n\\end{tabular*}'
    return result_str


def plot_df(df, filename, max=True, min=True, median=True, mean=True, x_label="", y_label="", max_orig_lines=1000, alpha=0.1, verbose=True, file_ext='.png',lw=2):
    if verbose:
        printing.print_f('plot dataframe', class_name='pd_tools')
    basics.create_folder_structure(filename)
    fig, ax = plt.subplots()
    if (isinstance(df, pd.DataFrame) and len(df.columns) < max_orig_lines) or (
        isinstance(df, pd.Series) and (max_orig_lines > 0)):
        df.plot(alpha=alpha, ax=ax, lw=lw if isinstance(df, pd.Series) else 1)
    stat = pd.DataFrame()
    legend_labels = set()
    if max:
        stat['max'] = df.max(axis=1)
        legend_labels.add('max')
    else:
        stat['max'] = np.nan
    if mean:
        stat['mean'] = df.mean(axis=1)
        legend_labels.add('mean')
    else:
        stat['mean'] = np.nan
    if median:
        stat['median'] = df.median(axis=1)
        legend_labels.add('median')
    else:
        stat['median'] = np.nan
    if min:
        stat['min'] = df.min(axis=1)
        legend_labels.add('min')
    else:
        stat['min'] = np.nan
    if any([max, mean, median, min]):
        stat[['max', 'mean', 'median', 'min']].plot(lw=lw, ax=ax,
                                                    color=['darkgreen', 'blue', 'red', 'lightgreen'], alpha=0.8)
    try:
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = zip(*[(i, j) for i, j in zip(handles, labels) if j in legend_labels])
        ax.legend(handles, labels)
    except ValueError:
        pass
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename if filename.endswith(file_ext) else filename + file_ext, bbox_inches='tight')
    return df
