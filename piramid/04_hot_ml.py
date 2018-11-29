#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  03_feature_selection.py
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
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from astropy.stats import sigma_clipped_stats

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

import custom_funs as cf


storefile = '/mnt/clemente/bos0109/table_store2.h5'

store = pd.HDFStore(storefile)
store.open()

#sns.set_context(font_scale=16)
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams['text.usetex'] = False


def main(m1_diam=1.54, plots_path='./plots/.'):
    plot_dir = os.path.abspath(plots_path)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    #~ simulated = store['simulated']
    #~ simulations = store['simulations']

    #~ simulations = simulations[simulations.failed_to_subtract==False]
    #~ if m1_diam is not None:
        #~ simulations = simulations[simulations.m1_diam==m1_diam]

    #~ simus = pd.merge(left=simulations, right=simulated,
                     #~ right_on='simulation_id', left_on='id', how='outer')

    # =============================================================================
    # tables
    # =============================================================================
    dt_zps = store['dt_hot']
    if m1_diam is not None:
        dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
    dt_zps = cf.optimize_df(dt_zps)
    dt_ois = dt_zps
    del(dt_zps)

    dt_ois['MU'] = dt_ois.mag/(dt_ois.A_IMAGE*dt_ois.B_IMAGE)
    dt_ois['SN'] = dt_ois.FLUX_APER/dt_ois.FLUXERR_APER

    merged = store['merged']
    selected = merged[merged.selected==True]

# =============================================================================
# Usar los seleccionados desde la tabla merged
# =============================================================================
    ids = selected['image_id_hot'].drop_duplicates().values
    dt_ois = dt_ois.loc[dt_ois['image_id'].isin(ids)].drop_duplicates()

    store.close()

# =============================================================================
# Columnas usables
# =============================================================================
    cols = ['FLUX_ISO', 'FLUXERR_ISO', 'MAG_ISO', 'MAGERR_ISO',
            'FLUX_APER', 'FLUXERR_APER', 'MAG_APER', 'MAGERR_APER', 'FLUX_AUTO',
            'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', 'BACKGROUND', 'THRESHOLD',
            'FLUX_MAX', 'XMIN_IMAGE', 'YMIN_IMAGE', 'XMAX_IMAGE', 'YMAX_IMAGE',
            'XPEAK_IMAGE', 'YPEAK_IMAGE', 'X_IMAGE', 'Y_IMAGE', 'X2_IMAGE',
            'Y2_IMAGE', 'XY_IMAGE', 'CXX_IMAGE', 'CYY_IMAGE', 'CXY_IMAGE',
            'A_IMAGE', 'B_IMAGE', 'THETA_IMAGE', 'MU_MAX', 'FLAGS', 'FWHM_IMAGE',
            'ELONGATION', 'ELLIPTICITY', 'CLASS_STAR', 'MU_THRESHOLD', 'SNR_WIN',
            'DELTAX', 'DELTAY', 'RATIO', 'ROUNDNESS', 'PEAK_CENTROID',
            'ref_fwhm', 'new_fwhm', 'px_scale', 'ref_back_sbright',
            'new_back_sbright', 'exp_time', 'VALID_MAG', 'mean_offset',
            'slope', 'mag', 'VALID_MAG_iso', 'mean_offset_iso',
            'slope_iso', 'mag_iso', 'mean_goyet', 'mean_goyet_iso', 'MU', 'SN']

    target = ['IS_REAL']

# =============================================================================
# Para que entre en memoria hacemos un sampling de esto
# =============================================================================
    train_ois, ftest_ois = train_test_split(dt_ois, test_size=0.7, stratify=dt_ois.IS_REAL)

# =============================================================================
# Aca separo en grupos... Agrupo por distintas cosas
# =============================================================================
    ois_grouping = cf.group_ml(train_ois, cols=cols, method='Alard')

    ois_grouping.to_csv(os.path.join(plots_path, 'hot_grouping_table.csv'))
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--m1_diam", help="diameter to filter",
                        default=None, type=float)
    parser.add_argument("path", help="path to plot files", default='./plots')
    args = parser.parse_args()

    import sys
    sys.exit(main(args.m1_diam, args.path))
