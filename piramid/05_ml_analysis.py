#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  05_ml_analysis.py
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
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

from astropy.stats import sigma_clipped_stats

zps = pd.read_csv('zps_grouping_table.csv')
ois = pd.read_csv('ois_grouping_table.csv')
sps = pd.read_csv('sps_grouping_table.csv')
hot = pd.read_csv('./plots/hot_grouping_table.csv')

cols = hot.columns

group_cols = cols[1:4]

knn_cols = [col for col in cols if col.startswith('knn')]
rfo_cols = [col for col in cols if col.startswith('rfo')]
svc_cols = [col for col in cols if col.startswith('svc')]

labels=['exp0', 'exp', 'test0', 'test']
colors=['red', 'darkred', 'blue', 'darkblue']

bins = np.arange(0.8, 1., 0.02)
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.title('KNN')
plt.hist([hot['knn_exp0_f1'],hot['knn_exp_f1'],
          hot['knn_test0_f1'], hot['knn_test_f1']],
          alpha=0.5, label=labels, bins=bins, color=colors)
plt.xlabel('F1 metric')
plt.legend(loc='best')

plt.subplot(132)
plt.title('SVM')
plt.hist([hot['svc_exp0_f1'],hot['svc_exp_f1'],
          hot['svc_test0_f1'], hot['svc_test_f1']],
          alpha=0.5, label=labels, bins=bins, color=colors)
plt.xlabel('F1 metric')
plt.legend(loc='best')

plt.subplot(133)
plt.title('Random Forest')
plt.hist([hot['rfo_exp0_f1'],hot['rfo_exp_f1'],
          hot['rfo_test0_f1'], hot['rfo_test_f1']],
          alpha=0.5, label=labels, bins=bins, color=colors)
plt.xlabel('F1 metric')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# =============================================================================
#
# =============================================================================
bins = np.arange(0.8, 1., 0.025)
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.title('KNN')
plt.hist(hot['knn_exp0_bacc'],  alpha=0.5, label='exp0',  bins=bins, histtype='bar')
plt.hist(hot['knn_exp_bacc'],   alpha=0.5, label='exp',   bins=bins, histtype='bar')
plt.hist(hot['knn_test0_bacc'], alpha=0.5, label='test0', bins=bins, histtype='bar')
plt.hist(hot['knn_test_bacc'],  alpha=0.5, label='test',  bins=bins, histtype='bar')
plt.xlabel('Bal. Accuracy metric')
plt.legend(loc='best')

plt.subplot(132)
plt.title('SVM')
plt.hist(hot['svc_exp0_bacc'],  alpha=0.5, label='exp0',  bins=bins)
plt.hist(hot['svc_exp_bacc'],   alpha=0.5, label='exp',   bins=bins)
plt.hist(hot['svc_test0_bacc'], alpha=0.5, label='test0', bins=bins)
plt.hist(hot['svc_test_bacc'],  alpha=0.5, label='test',  bins=bins)
plt.xlabel('Bal. Accuracy metric')
plt.legend(loc='best')

plt.subplot(133)
plt.title('Random Forest')
plt.hist(hot['rfo_exp0_bacc'],  alpha=0.5, label='exp0',  bins=bins)
plt.hist(hot['rfo_exp_bacc'],   alpha=0.5, label='exp',   bins=bins)
plt.hist(hot['rfo_test0_bacc'], alpha=0.5, label='test0', bins=bins)
plt.hist(hot['rfo_test_bacc'],  alpha=0.5, label='test',  bins=bins)
plt.xlabel('Bal. Accuracy metric')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


delta0_f1 = hot['knn_exp0_f1'] - hot['knn_test0_f1']
delta_f1 = hot['knn_exp_f1'] - hot['knn_test_f1']
delta0_bacc = hot['knn_exp0_bacc'] - hot['knn_test0_bacc']
delta_bacc = hot['knn_exp_bacc'] - hot['knn_test_bacc']

plt.hist(delta0_f1)
plt.hist(delta_f1)
plt.show()

plt.hist(delta0_bacc)
plt.hist(delta_bacc)
plt.show()


