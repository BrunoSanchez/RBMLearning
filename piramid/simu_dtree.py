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
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import tree

from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import graphviz

import custom_funs as cf


storefile = '/mnt/clemente/bos0109/table_store.h5'

store = pd.HDFStore(storefile)
store.open()


simus = store['simulations']

simus['new_fwhm_px'] = simus['new_fwhm'] / simus['px_scale']
simus['ref_fwhm_px'] = simus['ref_fwhm'] / simus['px_scale']
simus['new_back_px'] = simus['new_back_sbright'] / simus['px_scale']
simus['ref_back_px'] = simus['ref_back_sbright'] / simus['px_scale']

simus['m1_exp'] = simus['m1_diam'] * simus['exp_time']
simus['m2_exp'] = simus['m2_diam'] * simus['exp_time']
simus['eff_col_exp'] = simus['eff_col'] * simus['exp_time']

simus['new_back_px_exp'] = simus['exp_time'] / simus['new_back_px']
simus['ref_back_px_exp'] = simus['exp_time'] / simus['ref_back_px']



cols = ['id', 'code', 'executed', 'loaded', 'crossmatched',
        'failed_to_subtract', 'possible_saturation', 'ref_starzp',
        'ref_starslope', 'ref_fwhm', 'new_fwhm', 'm1_diam', 'm2_diam',
        'eff_col', 'px_scale', 'ref_back_sbright', 'new_back_sbright',
        'exp_time']
y = simus['failed_to_subtract'].values.astype(int)
x = ['ref_fwhm', 'new_fwhm', 'm1_diam', 'ref_starslope', 'm2_diam',
     'eff_col', 'px_scale', 'ref_back_sbright', 'new_back_sbright', 'exp_time',
     'new_fwhm_px', 'ref_fwhm_px', 'new_back_px', 'ref_back_px',
     'm1_exp', 'm2_exp', 'eff_col_exp', 'new_back_px_exp', 'ref_back_px_exp']
X = simus[x].values

clf = tree.DecisionTreeClassifier(criterion='entropy',
                                  min_impurity_decrease=0.000001,
                                  class_weight=None,
                                  max_depth=6,
                                  presort=True)
rslts_c45 = cf.experiment(clf, X, y, printing=True, nfolds=20)
clf = rslts_c45['model']

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=x,
                         class_names=['simulated', 'failed'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('simulations')

# =============================================================================
# Ahora con goyet
# =============================================================================
merged = store['merged']

pars = ['simulation_id', 'has_goyet_sps', 'has_goyet_zps', 'has_goyet_ois',
       'has_goyet_hot', 'mix_goyet', 'selected', 'mean_goyet_zps',
       'mean_goyet_sps', 'mean_goyet_hot', 'mean_goyet_ois']

dat = pd.merge(left=merged[pars], right=simus,
               left_on='simulation_id', right_on='id',
               how='right')
dat.drop(columns=['code', 'crossmatched', 'loaded', 'executed', 'possible_saturation'], inplace=True)
dat = cf.optimize_df(dat)
dat.drop_duplicates(inplace=True)



cols = ['simulation_id', 'has_goyet_sps', 'has_goyet_zps', 'has_goyet_ois',
       'has_goyet_hot', 'mix_goyet', 'selected', 'mean_goyet_zps',
       'mean_goyet_sps', 'mean_goyet_hot', 'mean_goyet_ois', 'id',
       'failed_to_subtract', 'ref_starzp', 'ref_starslope', 'ref_fwhm',
       'new_fwhm', 'm1_diam', 'm2_diam', 'eff_col', 'px_scale',
       'ref_back_sbright', 'new_back_sbright', 'exp_time', 'new_fwhm_px',
       'ref_fwhm_px', 'new_back_px', 'ref_back_px', 'm1_exp', 'm2_exp',
       'eff_col_exp', 'new_back_px_exp', 'ref_back_px_exp']

y = (dat['selected'] | dat['failed_to_subtract']).values.astype(int)

x = ['ref_starzp', 'ref_starslope', 'ref_fwhm',
     'new_fwhm', 'm1_diam', 'm2_diam', 'eff_col', 'px_scale',
     'ref_back_sbright', 'new_back_sbright', 'exp_time', 'new_fwhm_px',
     'ref_fwhm_px', 'new_back_px', 'ref_back_px', 'm1_exp', 'm2_exp',
     'eff_col_exp', 'new_back_px_exp', 'ref_back_px_exp']


dat['new_fwhm_px'] = dat['new_fwhm'] / dat['px_scale']
dat['ref_fwhm_px'] = dat['ref_fwhm'] / dat['px_scale']
dat['new_back_px'] = dat['new_back_sbright'] / dat['px_scale']
dat['ref_back_px'] = dat['ref_back_sbright'] / dat['px_scale']

dat['m1_exp'] = dat['m1_diam'] * dat['exp_time']
dat['m2_exp'] = dat['m2_diam'] * dat['exp_time']
dat['eff_col_exp'] = dat['eff_col'] * dat['exp_time']

dat['new_back_px_exp'] = dat['exp_time'] / dat['new_back_px']
dat['ref_back_px_exp'] = dat['exp_time'] / dat['ref_back_px']

X = dat[x].values


clf = tree.DecisionTreeClassifier(criterion='entropy',
                                  min_impurity_decrease=0.0001,
                                  class_weight=None,
                                  max_depth=6,
                                  presort=True)
rslts_c45 = cf.experiment(clf, X, y, printing=True, nfolds=20)

clf = rslts_c45['model']


dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=x,
                         class_names=['selected', 'disposed'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('selected')



