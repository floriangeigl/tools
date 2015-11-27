from __future__ import division
from sys import platform as _platform
import matplotlib
import matplotlib.cm as colormap

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import sys
import os

sys.path.append('./../')

import unittest
import pd_tools
import pandas as pd
import random
import string


class Test_sunburst(unittest.TestCase):
    def setUp(self):
        pass

    def test_rnd_gen(self):
        alphabet = string.ascii_lowercase
        ser = pd.Series(
            data=['-'.join([alphabet[random.randint(0, 3)] for j in range(random.randint(1, 3))]) for i in range(200)])
        test_file = 'test.csv'
        pd_tools.to_sunburst_csv(ser, test_file, exec_r=True)
        os.remove(test_file)
        #os.remove(test_file.rsplit('.csv', 1)[0] + '.html')
