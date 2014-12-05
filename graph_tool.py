from graph_tool.all import *
import os
import matplotlib.cm as colormap
import pandas as pd
import Image
import subprocess
from printing import print_f
import random
import datetime

class GraphAnimator():
    def __init__(self, dataframe, categories, network, filename='output/network_evolution.png', verbose=1, df_iteration_key='iteration', df_vertex_key='vertex',
                 df_cat_key='categories', plot_each=1, fps=10, output_size=1080, bg_color='white', fraction_groups=None, smoothing=1, rate=30):
        self.df = dataframe
        self.categories = categories
        self.network = network
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
        if not os.path.isdir(self.filename_folder + '/' + self.tmp_folder_name):
            try:
                os.mkdir(self.filename_folder + '/' + self.tmp_folder_name)
            except:
                self.print_f('Could not create tmp-folder:', self.filename_folder + '/' + self.tmp_folder_name)
                raise Exception
        self.verbose = verbose
        self.df_iteration_key = df_iteration_key
        self.df_vertex_key = df_vertex_key
        self.df_cat_key = df_cat_key
        self.plot_each = plot_each
        self.fps = fps
        self.output_size = output_size
        self.bg_color = bg_color
        self.fraction_groups = fraction_groups
        self.smoothing = smoothing
        self.rate = rate
        self.pos = None

    def generate_filename(self, filenum):
        return self.filename_folder + '/' + self.tmp_folder_name + str(int(filenum)).rjust(6, '0') + self.filename_basename

    @property
    def network(self):
        return self.network

    @network.setter
    def network(self, network):
        self.pos = None
        self.network = network

    @staticmethod
    def get_categories_color_mapping(categories, groups=None):
        GraphAnimator.print_f('get color mapping')
        cmap = colormap.get_cmap('gist_rainbow')
        if groups:
            try:
                g_cat = set.union(*[groups[i] for i in categories])
                g_cat_map = {i: idx for idx, i in enumerate(g_cat)}
                num_g_cat = len(g_cat)
                color_mapping = {i: g_cat_map[random.sample(groups[i], 1)[0]] / num_g_cat for i in categories}
            except:
                GraphAnimator.print_f('Error in getting categories color mapping.', sys.exc_info())
                return GraphAnimator.get_categories_color_mapping(categories)
        else:
            num_categories = len(categories)
            color_mapping = {i: idx / num_categories for idx, i in enumerate(categories)}
        result = {key: (cmap(val), val) for key, val in color_mapping.iteritems()}
        deactivated_color_nodes = [0.179, 0.179, 0.179, 0.05]
        result.update({-1: (deactivated_color_nodes, -1)})
        return result

    @staticmethod
    def print_f(*args, **kwargs):
        kwargs.update({'class_name': 'GraphAnimator'})
        print_f(*args, **kwargs)

    def calc_absolute_positions(self, pos=None, reposition=False, **kwargs):
        if pos is not None:
            if reposition:
                pos = sfdp_layout(self.network, pos=pos, **kwargs)
            else:
                pos = sfdp_layout(self.network, **kwargs)
        pos_ar = np.array([np.array(pos[v]) for v in self.network.vertices()])
        max_x, max_y = pos_ar.max(axis=0)
        min_x, min_y = pos_ar.min(axis=0)
        max_x -= min_x
        max_y -= min_y
        spacing = 0.15 if self.network.num_vertices() > 10 else 0.3
        for v in self.network.vertices():
            pos[v] = [(pos[v][0] - min_x) / max_x * self.output_size * (1 - spacing) + (self.output_size * (spacing / 2)),
                      (pos[v][1] - min_y) / max_y * self.output_size * (1 - spacing) + (self.output_size * (spacing / 2))]
        return pos

    def calc_grouped_sfdp_layout(self, groups_vp='groups', pos=None, mu=3, **kwargs):
        orig_groups_map = self.network.vp[groups_vp] if isinstance(groups_vp, str) else groups_vp
        e_weights = self.network.new_edge_property('float')
        for e in self.network.edges():
            src_g, dest_g = orig_groups_map[e.source()], orig_groups_map[e.target()]
            try:
                e_weights[e] = len(src_g & dest_g) / len(src_g | dest_g)
            except ZeroDivisionError:
                e_weights[e] = 0
        groups_map = self.network.new_vertex_property('int')
        for v in self.network.vertices():
            v_orig_groups = orig_groups_map[v]
            if len(v_orig_groups) > 0:
                groups_map[v] = random.sample(v_orig_groups, 1)[0]
            else:
                groups_map[v] = -1
        return sfdp_layout(self.network, pos=pos, groups=groups_map, eweight=e_weights, mu=mu, **kwargs)

    def plot_network_evolution(self, dynamic_pos=False):
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
        if self.verbose > 0:
            self.print_f('Framerate:', fps)
            self.print_f('Meetings per second:', fps / smoothing)
            self.print_f('Smoothing:', smoothing)
            self.print_f('Init pause:', init_pause_time)

        # get colors
        categories_colors = self.get_categories_color_mapping(self.categories, self.fraction_groups)
        # get positions &
        if self.verbose >= 1:
            self.print_f('calc graph layout')
        try:
            self.pos = self.calc_grouped_sfdp_layout(groups_vp='groups')
        except KeyError:
            self.pos = sfdp_layout(self.network)
        # calc absolute positions
        self.pos = self.calc_absolute_positions(self.pos)

        # PLOT
        total_iterations = int(self.df[self.df_iteration_key].max())
        if self.verbose >= 1:
            self.print_f('iterations:', total_iterations)
        self.network.vertex_properties[self.df_cat_key] = self.network.new_vertex_property('object')
        fractions_vp = self.network.vertex_properties[self.df_cat_key]
        for v in self.network.vertices():
            fractions_vp[v] = set()
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

        last_iteration = -1
        draw_edges = True
        just_copy = True
        pos = self.pos
        last_progress_perc = -1
        start = datetime.datetime.now()
        for iteration, data in grouped_by_iteration:
            for one_iteration in range(last_iteration + 1, iteration + 1):
                last_iteration = one_iteration
                if self.verbose >= 2:
                    self.print_f('iteration:', one_iteration)
                if one_iteration == iteration:
                    for idx, row in data.iterrows():
                        vertex = row[self.df_vertex_key]
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
                        if self.verbose >= 2:
                            Plotting.print_f(one_iteration, vertex, 'has', fractions_vp[vertex])
                if one_iteration % self.plot_each == 0 or one_iteration == 0 or one_iteration == total_iterations:
                    current_perc = int(one_iteration / total_iterations * 100)
                    if one_iteration > 0:
                        avg_time = (datetime.datetime.now() - start).total_seconds() / one_iteration
                        est_time = datetime.timedelta(seconds=int(avg_time * (total_iterations - one_iteration)))
                    else:
                        est_time = '-'
                    if self.verbose >= 1:
                        if self.verbose >= 2 or current_perc > last_progress_perc:
                            last_progress_perc = current_perc
                            ext = 'draw edges' if draw_edges else ''
                            self.print_f('plot network evolution iteration:', one_iteration, '(' + str(current_perc) + '%)', 'est remain:', est_time, ext)
                    if one_iteration == 0 or one_iteration == total_iterations:
                        for i in xrange(init_pause_time):
                            offset = i
                            if one_iteration == total_iterations:
                                offset += init_pause_time
                            self.__draw_graph_animation_pic(fractions_vp, categories_colors, one_iteration, pos=pos, draw_edges=draw_edges, just_copy_last=i != 0,
                                                            smoothing=smoothing)
                        init_pause_time -= 1
                    else:
                        self.__draw_graph_animation_pic(fractions_vp, categories_colors, one_iteration, pos=pos, draw_edges=draw_edges, smoothing=smoothing,
                                                        just_copy_last=just_copy)
                    draw_edges = False
                    just_copy = True

        if self.filename_basename.endswith('.png'):
            file_basename = self.filename_basename[:-4]
        if _platform == "linux" or _platform == "linux2":
            with open(os.devnull, "w") as devnull:
                if self.verbose >= 1:
                    self.print_f('create movie...')
                exit_status = subprocess.check_call(
                    ['ffmpeg', '-i', self.filename_folder + '/' + self.tmp_folder_name + '%06d' + file_basename + '.png', '-framerate', str(fps), '-r', str(self.rate),
                     '-y', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', self.filename_folder + '/' + file_basename.strip('_') + '.avi'], stdout=devnull, stderr=devnull)
                if exit_status == 0:
                    if self.verbose >= 1:
                        self.print_f('delete pictures...')
                        exit_status = subprocess.check_call(['rm ' + str(self.filename_folder + '/' + self.tmp_folder_name + '*' + file_basename + '.png')], shell=True,
                                                            stdout=devnull)
        return self.df, self.network

    def __draw_graph_animation_pic(self, fraction_map, color_map, iteration, pos=None, draw_edges=True, just_copy_last=False, smoothing=1):
        if just_copy_last:
            min_filenum = self.output_filenum
            orig_filename = self.generate_filename(min_filenum - 1)
            for smoothing_step in range(smoothing):
                filename = self.generate_filename(self.output_filenum)
                shutil.copy(orig_filename, filename)
                self.output_filenum += 1
            self.print_f('Copy file:', orig_filename, ' X ', smoothing)
            return
        default_edge_alpha = min(1, (1 / np.log2(self.network.num_edges()) if self.network.num_edges() > 0 else 1))
        default_edge_color = [0.179, 0.203, 0.210, default_edge_alpha]
        deactivated_color_edges = [0.179, 0.203, 0.210, (1 / self.network.num_edges()) if self.network.num_edges() > 0 else 0]

        pos = sfdp_layout(self.network) if pos is None else pos
        min_vertex_size_shrinking_factor = 2

        size = self.network.new_vertex_property('float')

        try:
            colors = self.network.vp['node_color']
        except KeyError:
            colors = self.network.new_vertex_property('object')
            self.network.vp['node_color'] = colors
        try:
            fractions = self.network.vp['node_fractions']
        except KeyError:
            fractions = self.network.new_vertex_property('vector<double>')
            self.network.vp['node_fractions'] = fractions

        edge_color = self.network.new_edge_property('vector<double>')
        for e in self.network.edges():
            edge_color[e] = default_edge_color

        last_fraction_map = None
        try:
            if self.output_filenum > 0:
                last_fraction_map = self.network.vp['last_fraction_map']
            else:
                raise KeyError
        except KeyError:
            last_fraction_map = copy.copy(fraction_map)
            self.network.vp['last_fraction_map'] = last_fraction_map

        nodes_graph = GraphView(self.network, efilt=lambda x: False)
        edges_graph = None
        if draw_edges:
            edges_graph = self.network
        else:
            if not os.path.isfile(self.edges_filename):
                self.print_f('Edge picture file does not exist:', self.edges_filename)
                edges_graph = self.network

        current_size = nodes_graph.new_vertex_property('float')
        current_fraction_map = nodes_graph.new_vertex_property('object')
        vanish_fraction = nodes_graph.new_vertex_property('object')
        emerge_fraction = nodes_graph.new_vertex_property('object')
        vanish_fraction_reduce = nodes_graph.new_vertex_property('float')
        emerge_fraction_increase = nodes_graph.new_vertex_property('float')
        stay_fraction_change = nodes_graph.new_vertex_property('float')
        current_fraction_values = nodes_graph.new_vertex_property('vector<double>')
        fraction_mods = nodes_graph.new_vertex_property('vector<int>')
        for v in nodes_graph.vertices():
            new_frac = fraction_map[v]
            last_frac = last_fraction_map[v]
            new_frac_len = len(new_frac)
            last_frac_len = len(last_frac)
            if last_frac_len == 0:
                last_frac = {-1}
                last_frac_len = 1
            if new_frac_len == 0:
                new_frac = {-1}
                new_frac_len = 1
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
            if new_frac == {-1} and edges_graph is not None:
                for e in edges_graph.vertex(v).all_edges():
                    edge_color[e] = deactivated_color_edges

        num_nodes = self.network.num_vertices()
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
        tmp_pos = nodes_graph.new_vertex_property('vector<double>')
        for v in nodes_graph.vertices():
            tmp_pos[v] = pos[v]
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

        bg_img = None
        if edges_graph is not None:
            graph_draw(edges_graph, fit_view=False, pos=tmp_pos, vorder=size, vertex_size=0, vertex_color=self.bg_color, edge_pen_width=1, edge_color=edge_color,
                       output=self.edges_filename, output_size=output_size, nodesfirst=True, vertex_pen_width=0.0)
            if self.bg_color is not None:
                bg_img = Image.new("RGB", output_size, self.bg_color)
                fg_img = Image.open(self.edges_filename)
                bg_img.paste(fg_img, None, fg_img)
                bg_img.save(self.edges_filename, 'PNG')

        for smoothing_step in range(smoothing):
            fac = (smoothing_step + 1) / smoothing
            old_fac = 1 - fac
            new_fac = fac
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

            current_size.a = old_fac * old_size.a + new_fac * size.a
            filename = self.generate_filename(self.output_filenum)
            self.output_filenum += 1
            graph_draw(nodes_graph, fit_view=False, pos=tmp_pos, vorder=current_size, vertex_size=current_size, vertex_pie_fractions=current_fraction_values,
                       vertex_pie_colors=colors, vertex_shape="pie", edge_pen_width=1, edge_color=edge_color, output=filename, output_size=output_size,
                       vertex_pen_width=0.0)
            bg_img = Image.open(self.edges_filename)
            fg_img = Image.open(filename)
            bg_img.paste(fg_img, None, fg_img)
            bg_img.save(filename, 'PNG')
        self.network.vp['last_node_size'] = size
        self.network.vp['last_fraction_map'] = copy.copy(fraction_map)