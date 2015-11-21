from __future__ import division, print_function
from sys import platform as _platform
import matplotlib
if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import printing
import basics
import operator
from collections import defaultdict
import copy
from pandas.core.base import FrozenList


def set_output_fmt(max_colwidth=100000, width=100000, max_rows=10000):
    pd.set_option('display.max_colwidth', max_colwidth)
    pd.set_option('display.width', width)
    pd.set_option('display.max_rows', max_rows)


def create_multi_index_df(data_frame_dict):
    # dataframe dict must be a dictionary where the key is the first level index and the value is a dataframe of which the columns will be the second level indices
    data_frame_dict = sorted(data_frame_dict.iteritems(), key=operator.itemgetter(0))
    columns = pd.MultiIndex.from_tuples([(i, n) for i, j in data_frame_dict for n in j])
    first_level_idx, data_frames = zip(*data_frame_dict)
    data = np.array([np.array(i[j]) for i in data_frames for j in i.columns]).T
    return pd.DataFrame(columns=columns, data=data)


def print_tex_table(df, cols=None, mark_min=r'\wedge\ ', mark_max=r'\vee\ ', mark_any_space=r'\ \ ', digits=6, trim_zero_digits=False, print_index=False,
                    thousands_mark=True, scientific_notation=False, colors=True, diverging=False, color1='blue',
                    color2='red', min_style=r'\bm', max_style=r'\bm'):
    if diverging:
        num_steps = int(len(df) / 2) + 1
        steps = np.linspace(40, 20, num_steps).astype('int')
        default_colors = zip(steps, [color1] * len(steps)) + zip(steps, [color2] * len(steps))
    else:
        steps = np.linspace(40, 20, len(df)).astype('int')
        default_colors = zip(steps, [color1] * len(steps))

    # default_colors = [(40, 'blue'), (30, 'blue'), (20, 'blue'), (20, 'red'), (30, 'red'), (40, 'red')]
    assert len(default_colors) >= len(df)
    if colors is True:
        colors = defaultdict(lambda: default_colors)
    elif isinstance(colors, list):
        default_colors = colors
        colors = defaultdict(lambda: default_colors)
    elif isinstance(colors, dict):
        colors = defaultdict(lambda: default_colors, colors)
    result_str = ''
    if cols is not None:
        df = df[cols].copy()
    else:
        df = df.copy()
    width = len(df.columns)
    if print_index:
        col_fmt = '{l|' + '|'.join('c' * width) + '}'
    else:
        col_fmt = '{l|' + '|'.join('c' * (width - 1)) + '}'
    result_str += r'\begin{tabular*}{\linewidth}' + col_fmt + '\n' + r'\toprule' + '\n'
    if print_index:
        header = ' & '
    else:
        header = ''
    header += ' & '.join(df.columns) + r'\\' + '\n' + r'\midrule' + '\n'

    def create_list(a, list_len):
        if not isinstance(a, list):
            return [a] * list_len
        else:
            return a
    num_cols = len(df.columns)
    thousands_mark = create_list(thousands_mark, num_cols)
    scientific_notation = create_list(scientific_notation, num_cols)
    min_style = create_list(min_style, num_cols)
    max_style = create_list(max_style, num_cols)
    mark_min = create_list(mark_min, num_cols)
    mark_max = create_list(mark_max, num_cols)
    digits = create_list(digits, num_cols)
    trim_zero_digits = create_list(trim_zero_digits, num_cols)

    for col_idx, col in enumerate(df.columns):
        col_min, col_max = df[col].min(), df[col].max()
        # print(len(df[col]))
        if colors:
            col_colors = colors[col]
        col_digits = digits[col_idx]
        if df[col].dtype == np.float:
            base_format = 'e' if scientific_notation[col_idx] else 'f'
            if thousands_mark[col_idx]:
                format_string = '{:,.' + str(col_digits) + base_format + '}'
            else:
                format_string = '{:.' + str(col_digits) + base_format + '}'
            if trim_zero_digits[col_idx]:
                if base_format == 'e':
                    num_format = lambda x: (format_string.format(x)).split('e')[0].rstrip('0').rstrip('.') + 'e' + (format_string.format(x)).split('e')[-1]
                else:
                    num_format = lambda x: (format_string.format(x)).rstrip('0').rstrip('.')
            else:
                num_format = lambda x: format_string.format(x)
        elif df[col].dtype == np.int:
            num_format = lambda x: str(x)
        else:
            num_format = None

        if num_format is not None:
            sorted_vals = map(num_format, sorted(df[col]))
            df[col] = df[col].map(num_format)
            col_min, col_max = num_format(col_min), num_format(col_max)
            if colors:
                color_dict = {key: color + '!' + str(val)[:3] for key, val, color in zip(sorted_vals, *zip(*col_colors))}

            df[col] = df[col].apply(lambda x: ((r'\cellcolor{' + color_dict[x] + '} ') if colors else '') + (
                "$" + (min_style[col_idx] if x == col_min else max_style[col_idx]) + "{" + str(x == col_min) + x + "}$" if (
                x == col_min or x == col_max) else '$' + x + '$'))
            mark_any = mark_min[col_idx] or mark_max[col_idx]
            if mark_any:
                df[col] = df[col].apply(lambda x: mark_any_space + x if 'True' not in x and 'False' not in x else x)

            if mark_min[col_idx]:
                df[col] = df[col].apply(lambda x: x.replace('False', mark_min[col_idx]))
            else:
                df[col] = df[col].apply(lambda x: x.replace('False', ''))
            if mark_max[col_idx]:
                df[col] = df[col].apply(lambda x: x.replace('True', mark_max[col_idx]))
            else:
                df[col] = df[col].apply(lambda x: x.replace('True', ''))
    #result_str += str(df)
    result_str += header
    for idx, i in df.iterrows():
        if print_index:
            idx = str(idx) + ' & '
        else:
            idx = ''
        result_str += idx + ' & '.join(i) + r' \\' + '\n'
    result_str += r'\bottomrule' + '\n' + r'\end{tabular*}'
    result_str = r'\usepackage{ctable}' + '\n' + r'\usepackage{tabularx}' + '\n' + r'\usepackage{colortbl}' + '\n\n' + result_str
    return result_str


def float_levels_to_str(df, precision=5):
    fmt_str = "%." + str(int(precision)) + "f"
    levels_0 = map(lambda x: (fmt_str % x).rstrip('0'), map(float, df.columns.levels[0]))
    levels_1 = df.columns.levels[1]
    df.columns.set_levels(FrozenList([levels_0, levels_1]))
    return df


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


def df_decay(_df, _function):
    def _element_helper_func(x):
        if np.isnan(x):
            _element_helper_func.last_val = _function(_element_helper_func.last_val)
        else:
            _element_helper_func.last_val = x
        return _element_helper_func.last_val

    def _series_helper(s):
        _element_helper_func.last_val = 0
        return s.apply(func=_element_helper_func)

    return _df.apply(func=_series_helper, axis=0)


def df_sample(_df, _frequency):
    assert _frequency > 1
    num_rows = len(_df.index)
    frequency_range = np.arange(0, num_rows, _frequency)
    frequency_range[0] = 1
    if num_rows not in frequency_range:
        frequency_range = np.append(frequency_range, [num_rows])
    return _df.loc[frequency_range]
