#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  simu_dtree.py
#
#  Copyright 2018 Bruno S <bruno@oac.unc.edu.ar>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#


import os
from sqlalchemy import create_engine

import numpy as np
import seaborn as sns
import pandas as pd

from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt

import custom_funs as cf


storefile = '/mnt/clemente/bos0109/table_store.h5'

store = pd.HDFStore(storefile)
store.open()


simus = store['simulations']

cols = ['id', 'code', 'executed', 'loaded', 'crossmatched',
        'failed_to_subtract', 'possible_saturation', 'ref_starzp',
        'ref_starslope', 'ref_fwhm', 'new_fwhm', 'm1_diam', 'm2_diam',
        'eff_col', 'px_scale', 'ref_back_sbright', 'new_back_sbright',
        'exp_time']

y = simus['failed_to_subtract'].values.astype(int)

x = ['ref_starzp', 'ref_starslope', 'ref_fwhm', 'new_fwhm', 'm1_diam', 'm2_diam',
     'eff_col', 'px_scale', 'ref_back_sbright', 'new_back_sbright', 'exp_time']
X = simus[x].values

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=20)
rslts_c45 = experiment(clf, X, y)

tree = rslts_c45['model']


import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=x,
                         class_names=['simulated', 'failed'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('simluations.pdf')

