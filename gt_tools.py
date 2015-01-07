from __future__ import division
from sys import platform as _platform
import matplotlib
import matplotlib.cm as colormap

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
from graph_tool.all import *
import os
import matplotlib.cm as colormap
from matplotlib.colors import ColorConverter as color_converter
import pandas as pd
import Image
import ImageDraw
import ImageFont
import subprocess
import printing
import random
import datetime
import copy
import shutil
import numpy as np
import operator
import math
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
import sys
from scipy.stats import powerlaw, poisson
from collections import defaultdict
import traceback
from basics import create_folder_structure


def print_f(*args, **kwargs):
    if 'class_name' not in kwargs:
        kwargs.update({'class_name': 'gt_tools'})
    printing.print_f(*args, **kwargs)


class GraphAnimator():
    def __init__(self, dataframe, network, filename='output/network_evolution.png', verbose=1, df_iteration_key='iteration', df_vertex_key='vertex', df_cat_key=None,
                 df_size_key=None, plot_each=1, fps=10, output_size=1080, bg_color='white', fraction_groups=None, smoothing=1, rate=30, cmap=None,
                 inactive_fraction_f=lambda x: x == {-1}, inactive_value_f=lambda x: x <= 0, deactivated_color_nodes=None, mu=3):
        assert isinstance(dataframe, pd.DataFrame)
        self.df = dataframe
        self.df_iteration_key = df_iteration_key
        self.df_vertex_key = df_vertex_key
        self.df_cat_key = df_cat_key
        self.draw_fractions = self.df_cat_key is not None
        self.df_size_key = df_size_key
        if any(isinstance(i, int) for i in self.df_vertex_key):
            self.print_f('convert vertex column to vertex instances')
            self.df[self.df_vertex_key] = self.df[self.df_vertex_key].apply(func=lambda x: network.vertex(x))

        self.categories = set.union(*list(self.df[self.df_cat_key])) if self.df_cat_key is not None else None
        self.network = network
        assert isinstance(self.network, Graph)
        self.output_filenum = 0

        filename = filename if filename.endswith('.png') else filename + '.png'
        filename = filename.rsplit('/', 1)
        if len(filename) == 1:
            filename = ['.', filename[0]]
        filename[1] = str('_' + filename[1])
        filename = '/'.join(filename)
        self.filename = filename
        splited_filename = self.filename.rsplit('/', 1)
        self.filename_folder = splited_filename[0]
        self.filename_basename = splited_filename[-1]
        self.tmp_folder_name = 'graph_animator_tmp/'
        self.edges_filename = self.filename_folder + '/' + self.tmp_folder_name + 'edges_' + self.filename_basename
        self.mu = mu
        if not os.path.isdir(self.filename_folder + '/' + self.tmp_folder_name):
            try:
                create_folder_structure(self.filename_folder + '/' + self.tmp_folder_name)
            except:
                self.print_f('Could not create tmp-folder:', self.filename_folder + '/' + self.tmp_folder_name)
                raise Exception
        self.verbose = verbose
        self.plot_each = plot_each
        self.fps = fps
        self.output_size = output_size
        self.bg_color = bg_color
        self.fraction_groups = fraction_groups
        self.smoothing = smoothing
        self.rate = rate
        self.generate_files = None
        self.pos = None
        self.pos_abs = None
        self.cmap = cmap
        self.inactive_fraction_f = inactive_fraction_f
        self.inactive_value_f = inactive_value_f
        self.active_nodes = None
        if self.cmap is None:
            def_cmap = 'gist_rainbow'
            self.print_f('using default cmap:', def_cmap, verbose=2)
            self.cmap = colormap.get_cmap(def_cmap)
        elif isinstance(self.cmap, str):
            self.cmap = colormap.get_cmap(self.cmap)
        self.deactivated_color_nodes = [0.179, 0.179, 0.179, 0.05] if deactivated_color_nodes is None else deactivated_color_nodes
        efilt = self.network.new_edge_property('bool')
        for e in self.network.edges():
            efilt[e] = False
        self.network.ep['no_edges_filt'] = efilt

    def generate_filename(self, filenum):
        return self.filename_folder + '/' + self.tmp_folder_name + str(int(filenum)).rjust(6, '0') + self.filename_basename

    @property
    def network(self):
        return self.network

    @network.setter
    def network(self, network):
        self.pos = None
        self.network = network

    def get_color_mapping(self, categories=None, groups=None, cmap=None):
        self.print_f('get color mapping', verbose=2)
        if cmap is None:
            cmap = colormap.get_cmap('gist_rainbow')
        if groups and categories:
            try:
                g_cat = set.union(*[groups[i] for i in categories])
                g_cat_map = {i: idx for idx, i in enumerate(g_cat)}
                num_g_cat = len(g_cat)
                color_mapping = {i: g_cat_map[random.sample(groups[i], 1)[0]] / num_g_cat for i in sorted(categories)}
            except:
                self.print_f('Error in getting categories color mapping.', traceback.print_exc(), verbose=1)
                return self.get_color_mapping(categories, cmap=cmap)
        elif categories:
            num_categories = len(categories)
            color_mapping = {i: idx / num_categories for idx, i in enumerate(sorted(categories))}
        else:
            return cmap
        result = {key: (cmap(val), val) for key, val in color_mapping.iteritems()}
        result.update({-1: (self.deactivated_color_nodes, -1)})
        return result

    def print_f(self, *args, **kwargs):
        kwargs.update({'class_name': 'GraphAnimator'})
        if 'verbose' not in kwargs or kwargs['verbose'] <= self.verbose:
            print_f(*args, **kwargs)

    def calc_absolute_positions(self, init_pos=None, network=None, mu=None, **kwargs):
        if network is None:
            network = self.network
        if mu is None:
            mu = self.mu
        if init_pos is not None:
            pos = copy.copy(init_pos)
        else:
            pos = sfdp_layout(network, mu=mu, **kwargs)
        pos_ar = np.array([pos[v].a for v in network.vertices()])
        max_x, max_y = pos_ar.max(axis=0)
        min_x, min_y = pos_ar.min(axis=0)
        max_x -= min_x
        max_y -= min_y
        spacing = 0.15 if network.num_vertices() > 10 else 0.3
        shift_mult = self.output_size * (1 - spacing)
        shift_add = self.output_size * (spacing / 2)
        max_v = np.array([max_x, max_y])
        min_v = np.array([min_x, min_y])
        mult = 1 / max_v * shift_mult
        for v in network.vertices():
            if pos[v]:
                pos[v].a = (pos[v].a - min_v) * mult + shift_add
        return pos

    def calc_grouped_sfdp_layout(self, network=None, groups_vp='groups', pos=None, mu=None, **kwargs):
        if network is None:
            network = self.network
        if mu is None:
            mu = self.mu
        orig_groups_map = network.vp[groups_vp] if isinstance(groups_vp, str) else groups_vp
        e_weights = network.new_edge_property('float')
        for e in network.edges():
            src_g, dest_g = orig_groups_map[e.source()], orig_groups_map[e.target()]
            try:
                e_weights[e] = len(src_g & dest_g) / len(src_g | dest_g)
            except ZeroDivisionError:
                e_weights[e] = 0
        groups_map = network.new_vertex_property('int')
        for v in network.vertices():
            v_orig_groups = orig_groups_map[v]
            if len(v_orig_groups) > 0:
                groups_map[v] = random.sample(v_orig_groups, 1)[0]
            else:
                groups_map[v] = -1
        return sfdp_layout(network, pos=pos, groups=groups_map, eweight=e_weights, mu=mu, **kwargs)

    def plot_network_evolution(self, dynamic_pos=False, infer_size_from_fraction=True):
        self.output_filenum = 0
        tmp_smoothing = self.fps * self.smoothing
        smoothing = self.smoothing
        fps = self.fps
        while tmp_smoothing > self.rate:
            smoothing -= 1
            tmp_smoothing = fps * smoothing
        smoothing = max(1, smoothing)
        fps *= smoothing
        init_pause_time = 1.5 * fps / smoothing

        if init_pause_time == 0:
            init_pause_time = 2
        init_pause_time = int(math.ceil(init_pause_time))
        self.print_f('Framerate:', fps, verbose=1)
        self.print_f('Iterations per second:', fps / smoothing, verbose=1)
        self.print_f('Smoothing:', smoothing, verbose=1)
        self.print_f('Init pause:', init_pause_time, verbose=1)

        # get colors
        color_mapping = self.get_color_mapping(self.categories, self.fraction_groups, self.cmap)

        # get positions
        if not dynamic_pos:
            self.print_f('calc graph layout', verbose=1)
            try:
                self.pos = self.calc_grouped_sfdp_layout(network=self.network, groups_vp='groups')
            except KeyError:
                self.pos = sfdp_layout(self.network, mu=self.mu)
            # calc absolute positions
            self.pos_abs = self.calc_absolute_positions(self.pos, network=self.network)

        # PLOT
        total_iterations = self.df[self.df_iteration_key].max() - self.df[self.df_iteration_key].min()

        self.print_f('iterations:', total_iterations, verbose=1)
        if self.draw_fractions:
            self.print_f('draw fractions')
            self.network.vp[self.df_cat_key] = self.network.new_vertex_property('object')
            fractions_vp = self.network.vp[self.df_cat_key]
            for v in self.network.vertices():
                fractions_vp[v] = set()
        else:
            fractions_vp = None

        size_map = self.network.new_vertex_property('float')
        if self.df_size_key is not None:
            self.network.vp[self.df_size_key] = size_map

        try:
            _ = self.network.vp['NodeId']
        except KeyError:
            mapping = self.network.new_vertex_property('int')
            for v in self.network.vertices():
                mapping[v] = int(v)
            self.network.vp['NodeId'] = mapping
        self.df[self.df_iteration_key] = self.df[self.df_iteration_key].astype(int)
        grouped_by_iteration = self.df.groupby(self.df_iteration_key)
        self.print_f('Resulting video will be', int(total_iterations / self.plot_each * smoothing / fps) + (init_pause_time * 2 / fps * smoothing), 'seconds long')

        last_iteration = self.df[self.df_iteration_key].min() - 1
        draw_edges = True
        just_copy = True
        last_progress_perc = -1
        start = datetime.datetime.now()
        self.print_f('Iterations with changes:', ', '.join([str(i) for i, j in grouped_by_iteration]))
        self.generate_files = defaultdict(list)
        iteration_idx = -1
        for iteration, data in grouped_by_iteration:
            for one_iteration in range(last_iteration + 1, iteration + 1):
                iteration_idx += 1
                last_iteration = one_iteration
                self.print_f('iteration:', one_iteration, verbose=2)
                if one_iteration == iteration:
                    for idx, row in data.iterrows():
                        vertex = row[self.df_vertex_key]
                        if self.draw_fractions:
                            old_f_vp = fractions_vp[vertex]
                            new_f_vp = row[self.df_cat_key]
                            if not draw_edges:
                                len_old, len_new = len(old_f_vp), len(new_f_vp)
                                if len_old != len_new and (len_old == 0 or len_new == 0):
                                    draw_edges = True
                            if just_copy:
                                if old_f_vp != new_f_vp:
                                    just_copy = False
                            fractions_vp[vertex] = new_f_vp
                        else:
                            old_size = size_map[vertex]
                            new_size = row[self.df_size_key]
                            if not draw_edges and old_size != new_size and (old_size == 0 or new_size == 0):
                                draw_edges = True
                            if just_copy:
                                if old_size != new_size:
                                    just_copy = False
                            size_map[vertex] = new_size
                        self.print_f(one_iteration, vertex, 'has', fractions_vp[vertex] if self.draw_fractions else (size_map[vertex] if size_map is not None else ''), verbose=2)
                if iteration_idx % self.plot_each == 0 or iteration_idx == 0 or iteration_idx == total_iterations:
                    current_perc = int(iteration_idx / total_iterations * 100)
                    if iteration_idx > 0:
                        avg_time = (datetime.datetime.now() - start).total_seconds() / iteration_idx
                        est_time = datetime.timedelta(seconds=int(avg_time * (total_iterations - iteration_idx)))
                    else:
                        est_time = '-'
                    if self.verbose >= 2 or current_perc > last_progress_perc:
                        last_progress_perc = current_perc
                        ext = ' | draw edges' if draw_edges else ''
                        ext += ' | just copy' if just_copy else ''
                        self.print_f('plot network evolution iteration:', one_iteration, '(' + str(current_perc) + '%)', 'est remain:', est_time, ext, verbose=1)
                    if iteration_idx == 0 or iteration_idx == total_iterations:
                        for i in xrange(init_pause_time):
                            if i == 1:
                                self.print_f('init' if iteration_idx == 0 else 'exit', 'phase', verbose=1)
                            self.generate_files[one_iteration].extend(
                                self.__draw_graph_animation_pic(color_mapping, size_map=size_map, fraction_map=fractions_vp, draw_edges=draw_edges, just_copy_last=i != 0,
                                                                smoothing=smoothing, dynamic_pos=dynamic_pos, infer_size_from_fraction=infer_size_from_fraction))
                    else:
                        self.generate_files[one_iteration].extend(
                            self.__draw_graph_animation_pic(color_mapping, size_map=size_map, fraction_map=fractions_vp, draw_edges=draw_edges, smoothing=smoothing,
                                                            just_copy_last=just_copy, dynamic_pos=dynamic_pos, infer_size_from_fraction=infer_size_from_fraction))
                    draw_edges = False
                    just_copy = True

        text_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", int(round(self.output_size * 0.05)))
        self.print_f('label pictures')
        for label, files in self.generate_files.iteritems():
            label_img = Image.new("RGBA", (self.output_size, self.output_size), (255, 255, 255, 0))
            label_drawer = ImageDraw.Draw(label_img)
            label_drawer.text((self.output_size - (0.25 * self.output_size), self.output_size - (0.1 * self.output_size)), str(label), font=text_font, fill=(0, 0, 0, 255))
            for img_fname in files:
                img = Image.open(img_fname)
                img.paste(label_img, (0, 0), label_img)
                img.save(img_fname)

        if self.filename_basename.endswith('.png'):
            file_basename = self.filename_basename[:-4]
        else:
            file_basename = self.filename_basename
        if _platform == "linux" or _platform == "linux2":
            with open(os.devnull, "w") as devnull:
                self.print_f('create movie...', verbose=1)
                p_call = ['ffmpeg', '-framerate', str(fps), '-r', str(fps), '-i', self.filename_folder + '/' + self.tmp_folder_name + '%06d' + file_basename + '.png', '-framerate',
                          str(fps), '-r', str(self.rate), '-c:v', 'libx264', '-y', '-pix_fmt', 'yuv420p',
                          self.filename_folder + '/' + file_basename.strip('_') + '.avi']
                self.print_f('call:', p_call, verbose=2)
                exit_status = subprocess.check_call(p_call, stdout=devnull, stderr=devnull)
                if exit_status == 0:
                    self.print_f('delete pictures...', verbose=1)
                    _ = subprocess.check_call(['rm ' + str(self.filename_folder + '/' + self.tmp_folder_name + '*' + file_basename + '.png')], shell=True, stdout=devnull)
        return self.df, self.network

    def __draw_graph_animation_pic(self, color_map=colormap.get_cmap('gist_rainbow'), size_map=None, fraction_map=None, draw_edges=True, just_copy_last=False, smoothing=1,
                                   dynamic_pos=False, infer_size_from_fraction=True):
        generated_files = []
        if just_copy_last:
            min_filenum = self.output_filenum
            orig_filename = self.generate_filename(min_filenum - 1)
            for smoothing_step in range(smoothing):
                filename = self.generate_filename(self.output_filenum)
                shutil.copy(orig_filename, filename)
                generated_files.append(filename)
                self.output_filenum += 1
            self.print_f('Copy file:', orig_filename, ' X ', smoothing)
            return generated_files
        default_edge_alpha = min(1, (1 / np.log2(self.network.num_edges()) if self.network.num_edges() > 0 else 1))
        default_edge_color = [0.179, 0.203, 0.210, default_edge_alpha]
        deactivated_color_edges = [0.179, 0.203, 0.210, (1 / self.network.num_edges()) if self.network.num_edges() > 0 else 0]

        if self.pos is None:
            self.print_f('calc init positions')
            self.pos = sfdp_layout(self.network, mu=self.mu)
        if self.pos_abs is None:
            self.print_f('calc init abs-positions')
            self.pos_abs = self.calc_absolute_positions(self.pos, network=self.network)
        min_vertex_size_shrinking_factor = 2

        size = self.network.new_vertex_property('float')
        if not infer_size_from_fraction and size_map is None or (infer_size_from_fraction and fraction_map is None):
            infer_size_from_fraction = False
            if size_map is None:
                for v in self.network.vertices():
                    size[v] = 1
            else:
                size = copy.copy(size_map)
        try:
            colors = self.network.vp['node_color']
        except KeyError:
            if fraction_map is not None:
                colors = self.network.new_vertex_property('object')
            else:
                colors = self.network.new_vertex_property('vector<float>')
            self.network.vp['node_color'] = colors
        try:
            _ = self.network.vp['node_fractions']
        except KeyError:
            self.network.vp['node_fractions'] = self.network.new_vertex_property('vector<float>')

        edge_color = self.network.new_edge_property('vector<float>')
        active_edges = self.network.new_edge_property('bool')
        for e in self.network.edges():
            edge_color[e] = default_edge_color
            active_edges[e] = 1

        nodes_graph = GraphView(self.network, efilt=self.network.ep['no_edges_filt'])
        edges_graph = None
        if draw_edges or dynamic_pos:
            edges_graph = self.network
        else:
            if not os.path.isfile(self.edges_filename):
                self.print_f('Edge picture file does not exist:', self.edges_filename)
                edges_graph = self.network

        current_size = self.network.new_vertex_property('float')
        active_nodes = self.network.new_vertex_property('bool')

        if fraction_map is not None:
            try:
                if self.output_filenum > 0:
                    last_fraction_map = self.network.vp['last_fraction_map']
                else:
                    raise KeyError
            except KeyError:
                last_fraction_map = copy.copy(fraction_map)
                self.network.vp['last_fraction_map'] = last_fraction_map

            current_fraction_map = nodes_graph.new_vertex_property('object')
            vanish_fraction = nodes_graph.new_vertex_property('object')
            emerge_fraction = nodes_graph.new_vertex_property('object')
            vanish_fraction_reduce = nodes_graph.new_vertex_property('float')
            emerge_fraction_increase = nodes_graph.new_vertex_property('float')
            stay_fraction_change = nodes_graph.new_vertex_property('float')
            current_fraction_values = nodes_graph.new_vertex_property('vector<float>')
            fraction_mods = nodes_graph.new_vertex_property('vector<int>')
            for v in self.network.vertices():
                new_frac = fraction_map[v]
                last_frac = last_fraction_map[v]
                new_frac_len = len(new_frac)
                last_frac_len = len(last_frac)
                if last_frac_len == 0:
                    last_frac = {-1}
                    # last_frac_len = 1
                if new_frac_len == 0:
                    new_frac = {-1}
                    new_frac_len = 1
                if infer_size_from_fraction:
                    size[v] = new_frac_len
                current_frac = last_frac | new_frac
                current_fraction_map[v] = current_frac
                vanish = last_frac - new_frac
                vanish_fraction[v] = vanish
                emerge = new_frac - last_frac
                emerge_fraction[v] = emerge
                old_slice_size = 1 / len(last_frac) if len(last_frac) > 0 else 1
                new_slice_size = 1 / len(new_frac) if len(new_frac) > 0 else 1
                vanish_fraction_reduce[v] = -old_slice_size / smoothing
                emerge_fraction_increase[v] = new_slice_size / smoothing
                stay_fraction_change[v] = (new_slice_size - old_slice_size) / smoothing
                colors[v] = zip(*sorted([color_map[i] for i in current_frac], key=operator.itemgetter(1)))[0]
                tmp_current_fraction_values = []
                sorted_fractions = sorted(current_frac, key=lambda x: color_map[x][1])
                tmp_fraction_mod = []
                for i in sorted_fractions:
                    if i in emerge:
                        tmp_current_fraction_values.append(0)
                        tmp_fraction_mod.append(1)
                    else:
                        if i in vanish:
                            tmp_fraction_mod.append(-1)
                        else:
                            tmp_fraction_mod.append(0)
                        tmp_current_fraction_values.append(old_slice_size)
                fraction_mods[v] = tmp_fraction_mod
                current_fraction_values[v] = tmp_current_fraction_values
                if self.inactive_fraction_f(new_frac):
                    if edges_graph is not None:
                        for e in edges_graph.vertex(v).all_edges():
                            edge_color[e] = deactivated_color_edges
                            active_edges[e] = False
                else:
                    active_nodes[v] = True
        else:
            for v in self.network.vertices():
                val = size[v]
                inactive = self.inactive_value_f(val)
                colors[v] = color_map(val) if not inactive else (self.deactivated_color_nodes if not dynamic_pos else (self.deactivated_color_nodes[:3] + [0]))
                if inactive:
                    if edges_graph is not None:
                        for e in self.network.vertex(v).all_edges():
                            edge_color[e] = deactivated_color_edges if not dynamic_pos else (deactivated_color_edges[:3] + [0])
                            active_edges[e] = False
                else:
                    active_nodes[v] = True

        try:
            last_edge_color = self.network.ep['edge_color']
        except KeyError:
            last_edge_color = edge_color

        num_nodes = nodes_graph.num_vertices() if dynamic_pos else self.network.num_vertices()
        tmp_output_size = self.output_size
        if self.network.num_edges() == 0:
            tmp_output_size *= 0.9
        max_vertex_size = np.sqrt((np.pi * ((tmp_output_size / 4) ** 2)) / num_nodes)
        if max_vertex_size < min_vertex_size_shrinking_factor:
            max_vertex_size = min_vertex_size_shrinking_factor
        min_vertex_size = max_vertex_size / min_vertex_size_shrinking_factor
        if len(set(size.a)) == 1:
            max_vertex_size -= ((max_vertex_size - min_vertex_size) / 2)
            if max_vertex_size < 1:
                max_vertex_size = 1
            min_vertex_size = max_vertex_size

        output_size = (self.output_size, self.output_size)

        self.print_f('cal pos', verbose=2)
        all_active_nodes = self.network.new_vertex_property('bool')
        if self.active_nodes is None:
            all_active_nodes = active_nodes
            self.active_nodes = active_nodes
        else:
            all_active_nodes.a = self.active_nodes.a | active_nodes.a
        if dynamic_pos:
            pos_tmp_net = GraphView(self.network, vfilt=all_active_nodes, efilt=active_edges)
            count_new_active = 0
            calced_mean_pos = 0
            for v in filter(lambda m: not self.active_nodes[m] and active_nodes[m], self.network.vertices()):
                x, y = [], []
                x_abs, y_abs = [], []
                count_new_active += 1
                for n in filter(lambda neigh: self.active_nodes[neigh], v.all_neighbours()):
                    n_pos = self.pos[n].a
                    n_pos_abs = self.pos_abs[n].a
                    if len(n_pos) and all(np.invert(np.isnan(n_pos))) and all(n_pos != 0):
                        x.append(n_pos[0])
                        y.append(n_pos[1])
                    if len(n_pos_abs) and all(np.invert(np.isnan(n_pos_abs))) and all(n_pos_abs != 0):
                        x_abs.append(n_pos_abs[0])
                        y_abs.append(n_pos_abs[1])
                if len(x):
                    self.pos[v] = [np.mean(x), np.mean(y)]
                if len(x_abs):
                    self.pos_abs[v] = [np.mean(x_abs), np.mean(y_abs)]
                self.print_f('mean pos:', self.pos[v], verbose=2)
                calced_mean_pos += 1
            self.print_f('calced mean pos:', calced_mean_pos, verbose=2)
            self.print_f('new active nodes:', count_new_active, '(', count_new_active / pos_tmp_net.num_vertices() * 100, '%)', verbose=2)
            try:
                new_pos = self.calc_grouped_sfdp_layout(network=pos_tmp_net, groups_vp='groups', pos=self.pos)
                self.print_f('dyn pos: updated grouped sfdp', verbose=2)
            except KeyError:
                self.print_f('dyn pos: update sfdp', verbose=2)
                # tmp_deg_map = pos_tmp_net.degree_property_map('total')
                # vweights = pos_tmp_net.new_vertex_property('int')
                l_cp = label_largest_component(pos_tmp_net, directed=False)
                pin = pos_tmp_net.new_vertex_property('bool')
                # tmp_deg_map = prop_to_size(tmp_deg_map, mi=1, ma=10)
                # count_new_active = all_active_nodes.a.sum() - (self.active_nodes.a & active_nodes.a).sum()
                pin.a = self.active_nodes.a
                pin.a = pin.a & l_cp.a
                new_pos = sfdp_layout(pos_tmp_net, pin=pin, pos=self.pos, mu=self.mu)
                new_pos = sfdp_layout(pos_tmp_net, pos=new_pos, mu=self.mu, max_iter=10)

            # calc absolute positions
            new_pos_abs = self.calc_absolute_positions(new_pos, network=pos_tmp_net)
            # for v in self.network.vertices():
            # if not active_nodes[v]:
            # new_pos_abs[v] = self.pos_abs[v]
            # new_pos[v] = self.pos[v]
            edges_graph = pos_tmp_net
            nodes_graph = GraphView(pos_tmp_net, efilt=self.network.ep['no_edges_filt'])
        else:
            new_pos_abs = self.pos_abs
            new_pos = self.pos

        size = prop_to_size(size, mi=min_vertex_size, ma=max_vertex_size, power=1)
        copy_new_size = False
        old_size = None
        try:
            if self.output_filenum > 0:
                old_size = self.network.vp['last_node_size']
            else:
                copy_new_size = True
        except KeyError:
            copy_new_size = True
        if copy_new_size:
            old_size = prop_to_size(size, mi=min_vertex_size, ma=max_vertex_size, power=1)
            self.network.vp['last_node_size'] = old_size
        # print 'nodes graph:', nodes_graph.num_vertices(), nodes_graph.num_edges()
        # print 'edges graph:', edges_graph.num_vertices(), edges_graph.num_edges()

        for smoothing_step in range(smoothing):
            fac = (smoothing_step + 1) / smoothing
            old_fac = 1 - fac
            new_fac = fac
            if fraction_map is not None:
                for v in nodes_graph.vertices():
                    tmp = []
                    for mod, val in zip(list(fraction_mods[v]), list(current_fraction_values[v])):
                        if mod == 0:
                            val += stay_fraction_change[v]
                        elif mod == 1:
                            val += emerge_fraction_increase[v]
                        elif mod == -1:
                            val += vanish_fraction_reduce[v]
                        else:
                            self.print_f('ERROR: Fraction modification unknown')
                            raise Exception
                        tmp.append(val)
                    current_fraction_values[v] = tmp
                vertex_shape = "pie"
            else:
                current_fraction_values = [0.75, 0.25]
                vertex_shape = "circle"

            current_size.a = old_fac * old_size.a + new_fac * size.a
            if dynamic_pos:
                current_pos = nodes_graph.new_vertex_property('vector<float>')
                for v in nodes_graph.vertices():
                    if len(self.pos_abs[v].a) == 0 or all(i == 0 for i in self.pos_abs[v].a):
                        self.pos_abs[v] = new_pos_abs[v]
                        self.pos[v] = new_pos[v]
                    current_pos[v] = old_fac * self.pos_abs[v].a + new_fac * new_pos_abs[v].a
            else:
                current_pos = self.pos_abs

            if edges_graph is not None and (smoothing_step == 0 or dynamic_pos):
                eorder = edges_graph.new_edge_property('float')
                for e in edges_graph.edges():
                    eorder[e] = edge_color[e].a[-1]
                if dynamic_pos:
                    current_edge_color = edges_graph.new_edge_property('vector<float>')
                    for e in edges_graph.edges():
                        current_edge_color[e] = last_edge_color[e].a * old_fac + edge_color[e].a * new_fac
                else:
                    current_edge_color = edge_color
                self.print_f('draw edgegraph', verbose=2)
                graph_draw(edges_graph, output=self.edges_filename, output_size=output_size, pos=current_pos, fit_view=False, vorder=current_size, vertex_size=0.0,
                           vertex_fill_color=self.bg_color, vertex_color=self.bg_color, edge_pen_width=1, edge_color=current_edge_color, eorder=eorder, vertex_pen_width=0.0)
                self.print_f('ok', verbose=2)
                plt.close('all')
                if self.bg_color is not None:
                    bg_img = Image.new("RGB", output_size, self.bg_color)
                    fg_img = Image.open(self.edges_filename)
                    bg_img.paste(fg_img, None, fg_img)
                    bg_img.save(self.edges_filename, 'PNG')

            filename = self.generate_filename(self.output_filenum)
            self.output_filenum += 1
            self.print_f('draw nodegraph', verbose=2)
            graph_draw(nodes_graph, fit_view=False, pos=current_pos, vorder=current_size, vertex_size=current_size, vertex_pie_fractions=current_fraction_values,
                       vertex_pie_colors=colors, vertex_fill_color=colors, vertex_shape=vertex_shape, edge_pen_width=1, edge_color=edge_color,
                       output=filename, output_size=output_size, vertex_pen_width=0.0)
            self.print_f('ok', verbose=2)
            generated_files.append(filename)
            plt.close('all')

            bg_img = Image.open(self.edges_filename)
            fg_img = Image.open(filename)
            bg_img.paste(fg_img, None, fg_img)
            bg_img.save(filename, 'PNG')

        self.network.vp['last_node_size'] = size
        self.network.ep['edge_color'] = edge_color
        if fraction_map is not None:
            self.network.vp['last_fraction_map'] = copy.copy(fraction_map)
        if dynamic_pos:
            self.pos = new_pos
            self.pos_abs = new_pos_abs
            self.active_nodes.a = all_active_nodes.a
        return generated_files


# Generator Class works with GraphTool generators, as they provide more functionality than NetworkX Generators
class GraphGenerator():
    # init generator
    def __init__(self, num_nodes=5000, directed=False):
        self.directed = directed
        self.num_nodes = num_nodes
        self.graph = None
        self.node_id_map = None
        self.return_and_reset()

    def return_and_reset(self):
        result = self.graph
        self.graph = Graph(directed=self.directed)
        self.node_id_map = defaultdict(lambda x: self.graph.add_vertex())
        return result

    # start creating blockmodel graph
    def create_blockmodel_graph(self, blocks=7, connectivity=10, model="blockmodel-traditional"):
        def corr(a, b):
            if a == b:
                return 0.999
            else:
                return 0.001

        self.print_f("Starting to create Blockmodel Graph with {} nodes and {} blocks".format(self.num_nodes, blocks))

        self.graph, vertex_colors = random_graph(self.num_nodes, lambda: poisson(connectivity), directed=False, model=model, block_membership=lambda: random.randint(1, blocks),
                                                 vertex_corr=corr)
        self.graph.vertex_properties["colorsComm"] = vertex_colors
        return self.return_and_reset()

    def create_fully_connected_graph(self, size=1000, directed=False, self_edges=False):
        return self.create_stochastic_blockmodel_graph(blocks=1, size=size, directed=directed, self_edges=self_edges, self_block_connectivity=1.0, other_block_connectivity=1.0)

    @staticmethod
    def create_sbm_lined_up_matrix(blocks=10, self_block_connectivity=None, other_block_connectivity=None):
        if self_block_connectivity is None:
            self_block_connectivity = [0.9]
        elif isinstance(self_block_connectivity, (int, float)):
            self_block_connectivity = [self_block_connectivity]
        if other_block_connectivity is None:
            other_block_connectivity = [0.1]
        elif isinstance(other_block_connectivity, (int, float)):
            other_block_connectivity = [other_block_connectivity]
        connectivity_matrix = []
        blocks_range = range(blocks)
        for idx in blocks_range:
            row = []
            outer_prob = other_block_connectivity[idx % len(other_block_connectivity)]
            inner_prob = self_block_connectivity[idx % len(self_block_connectivity)]
            for jdx in blocks_range:
                if idx != jdx:
                    row.append(outer_prob / pow(abs(idx - jdx), 2))
                else:
                    row.append(inner_prob)
            connectivity_matrix.append(row)
        return connectivity_matrix

    # scale = None
    # scale = relative
    # scale = absolute
    def create_stochastic_blockmodel_graph(self, blocks=10, size=100, self_block_connectivity=0.9, other_block_connectivity=0.1, connectivity_matrix=None, directed=False,
                                           self_edges=False, power_exp=None, scale=None, plot_stat=False):
        size = size if isinstance(size, list) else [size]
        self_block_connectivity = self_block_connectivity if isinstance(self_block_connectivity, list) else [self_block_connectivity]
        other_block_connectivity = other_block_connectivity if isinstance(other_block_connectivity, list) else [other_block_connectivity]

        num_nodes = sum([size[i % len(size)] for i in xrange(blocks)])
        if power_exp is None:
            self.print_f("Starting to create Stochastic Blockmodel Graph with {} nodes and {} blocks".format(num_nodes, blocks))
        else:
            self.print_f("Starting to create degree-corrected (alpha=" + str(power_exp) + ") Stochastic Blockmodel Graph with {} nodes and {} blocks".format(num_nodes, blocks))
        self.print_f('convert/transform probabilities')
        blocks_range = range(blocks)
        block_sizes = np.array([size[i % len(size)] for i in blocks_range])

        # create connectivity matrix of self- and other-block-connectivity
        if connectivity_matrix is None:
            connectivity_matrix = []
            self.print_f('inner conn: ' + str(self_block_connectivity) + '\tother conn: ' + str(other_block_connectivity))
            for idx in blocks_range:
                row = []
                for jdx in blocks_range:
                    if idx == jdx:
                        row.append(self_block_connectivity[idx % len(self_block_connectivity)])
                    else:
                        if scale is not None:
                            prob = other_block_connectivity[idx % len(other_block_connectivity)] / (num_nodes - block_sizes[idx]) * block_sizes[jdx]
                            if directed:
                                row.append(prob)
                            else:
                                row.append(prob / 2)
                        else:
                            row.append(other_block_connectivity[idx % len(other_block_connectivity)])
                connectivity_matrix.append(row)

        # convert con-matrix to np.array
        if connectivity_matrix is not None and isinstance(connectivity_matrix, np.matrix):
            connectivity_matrix = np.asarray(connectivity_matrix)

        # convert con-matrix to np.array
        if connectivity_matrix is not None and not isinstance(connectivity_matrix, np.ndarray):
            connectivity_matrix = np.array(connectivity_matrix)

        self.print_f('conn mat')
        printing.print_matrix(connectivity_matrix)

        if scale == 'relative' or scale == 'absolute':
            new_connectivity_matrix = []
            for i in blocks_range:
                connectivity_row = connectivity_matrix[i, :] if connectivity_matrix is not None else None
                nodes_in_src_block = block_sizes[i]
                multp = 1 if scale == 'absolute' else (nodes_in_src_block * (nodes_in_src_block - 1))
                row_prob = [(connectivity_row[idx] * multp) / (nodes_in_src_block * (nodes_in_block - 1)) for idx, nodes_in_block in enumerate(block_sizes)]
                new_connectivity_matrix.append(np.array(row_prob))
            connectivity_matrix = np.array(new_connectivity_matrix)
            self.print_f(scale + ' scaled conn mat:')
            printing.print_matrix(connectivity_matrix)

        # create nodes and store corresponding block-id
        self.print_f('insert nodes')
        vertex_to_block = []
        appender = vertex_to_block.append
        colors = self.graph.new_vertex_property("float")
        for i in xrange(blocks):
            block_size = size[i % len(size)]
            for j in xrange(block_size):
                appender((self.graph.add_vertex(), i))
                node = vertex_to_block[-1][0]
                colors[node] = i

        # create edges
        get_rand = np.random.random
        add_edge = self.graph.add_edge

        self.print_f('create edge probs')
        degree_probs = defaultdict(lambda: dict())
        for vertex, block_id in vertex_to_block:
            if power_exp is None:
                degree_probs[block_id][vertex] = 1
            else:
                degree_probs[block_id][vertex] = math.exp(power_exp * np.random.random())

        tmp = dict()
        self.print_f('normalize edge probs')
        all_prop = []
        for block_id, node_to_prop in degree_probs.iteritems():
            sum_of_block_norm = 1 / sum(node_to_prop.values())
            tmp[block_id] = {key: val * sum_of_block_norm for key, val in node_to_prop.iteritems()}
            all_prop.append(tmp[block_id].values())
        degree_probs = tmp
        if plot_stat:
            plt.clf()
            plt.hist(all_prop, bins=15)
            plt.savefig("prop_dist.png")
            plt.close('all')

        self.print_f('count edges between blocks')
        edges_between_blocks = defaultdict(lambda: defaultdict(int))
        for idx, (src_node, src_block) in enumerate(vertex_to_block):
            conn_mat_row = connectivity_matrix[src_block, :]
            for dest_node, dest_block in vertex_to_block:
                if get_rand() < conn_mat_row[dest_block]:
                    edges_between_blocks[src_block][dest_block] += 1

        self.print_f('create edges')
        for src_block, dest_dict in edges_between_blocks.iteritems():
            self.print_f(' -- Processing Block {}. Creating links to: {}'.format(src_block, dest_dict))
            for dest_block, num_edges in dest_dict.iteritems():
                self.print_f('   ++ adding {} edges to {}'.format(num_edges, dest_block))
                for i in xrange(num_edges):
                    # find src node
                    prob = np.random.random()
                    prob_sum = 0
                    src_node = None
                    for vertex, v_prob in degree_probs[src_block].iteritems():
                        prob_sum += v_prob
                        if prob_sum >= prob:
                            src_node = vertex
                            break
                    # find dest node
                    prob = np.random.random()
                    prob_sum = 0
                    dest_node = None
                    for vertex, v_prob in degree_probs[dest_block].iteritems():
                        prob_sum += v_prob
                        if prob_sum >= prob:
                            dest_node = vertex
                            break
                    if src_node is None or dest_node is None:
                        print 'Error selecting node:', src_node, dest_node
                    if self.graph.edge(src_node, dest_node) is None:
                        if self_edges or not src_node == dest_node:
                            add_edge(src_node, dest_node)
        self.graph.vertex_properties["colorsComm"] = colors
        return self.return_and_reset()

    def create_preferential_attachment(self, communities=10):
        self.graph = price_network(self.num_nodes, directed=False, c=0, gamma=1, m=1)
        self.graph.vertex_properties['colorsComm'] = community_structure(self.graph, 1000, communities)
        return self.return_and_reset()

    # add node to graph and check if node is in node_dict
    def add_node(self, node_id, further_mappings=None):
        v = self.node_id_map[node_id]
        self.graph.vp['NodeId'][v] = node_id
        if further_mappings is not None:
            assert isinstance(further_mappings, dict)
            for key, val in further_mappings.iteritems():
                self.graph.vp[key][v] = val
        return v

    def load_smw_collab_network(self, filename, communities=10):
        self.print_f("Creating Graph")
        id_prop = self.graph.new_vertex_property("int")
        self.graph.vp["label"] = self.graph.new_vertex_property("string")

        f = open(filename, "rb")
        for idx, line in enumerate(f):
            if idx % 1000 == 0:
                self.print_f("--> parsing line %d" % idx)
            split_line = line.strip("\n").split("\t")
            source_v = self.add_node(split_line[0], id_prop)
            if split_line[1] != "":
                target_v = self.add_node(split_line[1], id_prop)
                self.graph.add_edge(source_v, target_v)

        self.print_f("Detecting Communities")
        self.graph.vp['colorsComm'] = community_structure(self.graph, 1000, communities)
        remove_self_loops(self.graph)
        remove_parallel_edges(self.graph)
        return self.return_and_reset()

    @staticmethod
    def increment_neighbours(vertices, b):
        for n in vertices:
            b[int(n)] += 1

    # start creating random graph
    # NOTE:
    # If min_degree is too small, graph will be disconnected and consist of many smaller graphs!
    # This could make diffusion problematic!
    def create_random_graph(self, min_degree=2, max_degree=40, model="probabilistic", communities=10):
        # Function to sample edges between nodes!
        self.print_f('create random graph')

        def sample_k(min_val, max_val, k=None):
            accept = False
            while not accept:
                k = random.randint(min_val, max_val + 1)
                accept = random.random() < 1.0 / k
            return k

        self.graph = random_graph(self.num_nodes, lambda: sample_k(min_degree, max_degree), model=model, vertex_corr=lambda i, k: 1.0 / (1 + abs(i - k)), directed=self.directed,
                                  n_iter=100)
        self.graph.vp['colorsComm'] = community_structure(self.graph, 10000, max_degree / communities)
        return self.return_and_reset()

    # start loading  graph
    def create_karate_graph(self):
        self.graph = collection.data["karate"]
        # Removing descriptions and readme, as they screw with the GML parser of networkx!
        self.graph.gp['description'] = ''
        self.graph.gp['readme'] = ''
        # Calculating Colors and updating members
        self.graph.vp['colorsComm'] = community_structure(self.graph, 10000, 2)
        self.directed = self.graph.is_directed()
        self.num_nodes = self.graph.num_vertices()
        return self.return_and_reset()

    def loaded_post_action(self):
        self.directed = self.graph.is_directed()
        self.num_nodes = self.graph.num_vertices()
        self.print_f("Graph loaded with {} nodes and {} edges".format(self.graph.num_vertices(), self.graph.num_edges()))

    # load graph from gml
    def load_gml(self, fn):
        self.print_f("Loading GML")
        self.graph = load_graph(fn)
        self.loaded_post_action()
        return self.return_and_reset()

    # load graph from file
    def load_gt(self, fn):
        self.print_f("Loading GT")
        self.graph = load_graph(fn)
        self.loaded_post_action()
        return self.return_and_reset()

    @staticmethod
    def print_f(*args, **kwargs):
        kwargs.update({'class_name': 'GraphGenerator'})
        print_f(*args, **kwargs)


def calc_eigenvalues(graph, num_ev=100):
    num_ev = min(100, num_ev)
    print_f("Extracting adjacency matrix!")
    adj_mat = adjacency(graph, weight=None)
    print_f("Starting calculation of {} Eigenvalues".format(num_ev))
    evals_large_sparse, evecs_large_sparse = largest_eigsh(adj_mat, num_ev * 2, which='LM')
    print_f("Finished calculating Eigenvalues")
    weights = sorted([float(x) for x in evals_large_sparse], reverse=True)[:num_ev]
    graph.gp["top_eigenvalues"] = graph.new_graph_property("vector<float>", weights)
    return graph


def cleanup_graph(graph, largest_comp=True, parallel_edges=False, self_loops=False):
    if largest_comp:
        reduce_to_largest_component(graph)
    if not parallel_edges:
        remove_parallel_edges(graph)
    if not self_loops:
        remove_self_loops(graph)
    return graph


def reduce_to_largest_component(graph):
    print_f("Reducing graph to largest connected component!")
    l = label_largest_component(graph)
    graph = GraphView(graph, vfilt=l)
    graph.purge_vertices(in_place=True)
    return graph


def calc_vertex_properties(graph, max_iter_ev=1000, max_iter_hits=1000):
    print_f("Calculating PageRank")
    graph.vp["pagerank"] = pagerank(graph)

    print_f("Calculating Clustering Coefficient")
    graph.vp["clustercoeff"] = local_clustering(graph)

    print_f("Calculating Eigenvector Centrality")
    ev, ev_centrality = eigenvector(graph, weight=None, max_iter=max_iter_ev)
    graph.vp["evcentrality"] = ev_centrality

    print_f("Calculating HITS")
    eig, authorities, hubs = hits(graph, weight=None, max_iter=max_iter_hits)
    graph.vp["authorities"] = authorities
    graph.vp["hubs"] = hubs

    print_f("Calculating Degree Property Map")
    graph.vertex_properties["degree"] = graph.degree_property_map("total")
    return graph

# plot graph to file
# TODO: check if code of GraphAnimator can be used
'''
def draw_graph(run=0, min_nsize=None, max_nsize=None, size_property=None, file_format="png", output_size=4000, appendix="", label_color="orange", draw_labels=False):
    if size_property == "degree":
        size_map = self.graph.new_vertex_property('float')
        for v in self.graph.vertices():
            size_map[v] = v.out_degree() + v.in_degree()

    if not (isinstance(size_property, int) or isinstance(size_property, float), isinstance(size_property, str)):
        size_map = size_property

    if min_nsize is None or max_nsize is None:
        val = math.sqrt(self.graph.num_vertices()) / self.graph.num_vertices() * (output_size / 4)
        mi = val if min_nsize is None else min_nsize
        ma = val * 2 if max_nsize is None else max_nsize

    if draw_labels:
        try:
            labels = self.graph.vertex_properties["label"]
        except:
            ls = self.graph.new_vertex_property("int")
            for ndx, n in enumerate(self.graph.vertices()):
                ls[n] = str(ndx)
            self.graph.vertex_properties["label"] = ls
            labels = self.graph.vertex_properties["label"]
    else:
        labels = self.graph.new_vertex_property("string")

    if size_property is not None:
        try:
            self.draw_specific_graph(self.graph.vertex_properties["colorsComm"], "communities", output_size, label_color, "black", mi, ma, labels, run, appendix, file_format, label_pos=0, v_size_prop_map=size_map)
        except Exception as e:
            self.debug_msg("\x1b[31m" + str(e) + "\x1b[00m")
    else:
        try:
            self.draw_specific_graph(self.graph.vertex_properties["colorsComm"], "communities", output_size, label_color, "black", mi, ma, labels, run, appendix, file_format, label_pos=0)
        except Exception as e:
            self.debug_msg("\x1b[31m" + str(e) + "\x1b[00m")

        try:
            self.draw_specific_graph(self.graph.vertex_properties["colorsMapping"], "mapping", output_size, label_color, "black", mi, ma, labels, run, appendix, file_format, label_pos=0)
        except Exception as e:
            self.debug_msg("\x1b[31m" + str(e) + "\x1b[00m")

        try:
            self.draw_specific_graph(self.graph.vertex_properties["colorsActivity"], "activity", output_size, label_color, "black", mi, ma, labels, run, appendix, file_format, label_pos=0)
        except Exception as e:
            self.debug_msg("\x1b[31m" + str(e) + "\x1b[00m")
'''

'''
def draw_specific_graph(self, colors, color_type_in_outfname, output_size, label_color, edge_color, mi, ma, labels, run, appendix, file_format, label_pos=0, pos=None, v_size_prop_map=None):
    if pos is None:
        try:
            pos = self.graph.vertex_properties["pos"]
        except KeyError:
            self.debug_msg("  --> Calculating SFDP layout positions!")
            pos = sfdp_layout(self.graph)
            self.graph.vertex_properties["pos"] = pos
            self.debug_msg("  --> Done!")

    if v_size_prop_map is None:
        try:
            v_size_prop_map = self.graph.vertex_properties["activity"]
        except KeyError:
            self.add_node_weights(0.0, 0.1)
            v_size_prop_map = self.graph.vertex_properties["activity"]

    graph_draw(self.graph, vertex_fill_color=colors, edge_color=edge_color, output_size=(output_size, output_size), vertex_text_color=label_color, pos=pos, vertex_size=(prop_to_size(v_size_prop_map, mi=mi, ma=ma)), vertex_text=labels, vertex_text_position=label_pos,
               output=config.graph_dir + "{}_{}_run_{}{}.{}".format(self.graph_name, color_type_in_outfname, run, appendix, file_format))
'''

'''
    def collect_colors(self, alpha=0.75):
        self.debug_msg("Collecting Colors for Graphs")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=self.graph.num_vertices())
        cmap = plt.get_cmap('gist_rainbow')
        norma = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
        camap = plt.get_cmap("Blues")
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        ma = matplotlib.cm.ScalarMappable(norm=norma, cmap=camap)

        clist = self.graph.new_vertex_property("vector<float>")
        calist = self.graph.new_vertex_property("vector<float>")
        for x in xrange(self.graph.num_vertices()):
            # color for node id
            l = list(m.to_rgba(x))
            l[3] = alpha
            node = self.graph.vertex(x)
            clist[node] = l

            # color for activity / weight of node
            weight = self.graph.vp["activity"][node]
            la = list(ma.to_rgba(weight))
            la[3] = alpha
            calist[node] = la
        self.graph.vp["colorsMapping"] = clist
        self.graph.vp["colorsActivity"] = calist
        self.graph.vp['colorsComm'] = community_structure(self.graph, 1000, 10)
'''
