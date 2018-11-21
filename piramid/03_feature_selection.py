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
from sklearn import decomposition
from sklearn import feature_selection
from sklearn import svm
from sklearn import metrics
from sklearn import neighbors
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2


from rfpimp import *

import custom_funs as cf


storefile = '/mnt/clemente/bos0109/table_store.h5'

store = pd.HDFStore(storefile)
store.open()

#sns.set_context(font_scale=16)
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams['text.usetex'] = False


def main(m1_diam=1.54, plots_path='./plots/.'):
    plot_dir = os.path.abspath(plots_path)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    simulated = store['simulated']
    simulations = store['simulations']

    simulations = simulations[simulations.failed_to_subtract==False]
    if m1_diam is not None:
        simulations = simulations[simulations.m1_diam==m1_diam]

    simus = pd.merge(left=simulations, right=simulated,
                     right_on='simulation_id', left_on='id', how='outer')

    # =============================================================================
    # tables
    # =============================================================================
    dt_zps = store['dt_ois']
    if m1_diam is not None:
        dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
    dt_zps = cf.optimize_df(dt_zps)
    dt_ois = dt_zps

    dt_zps = store['dt_sps']
    if m1_diam is not None:
        dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
    dt_zps = cf.optimize_df(dt_zps)
    dt_sps = dt_zps

    dt_zps = store['dt_hot']
    if m1_diam is not None:
        dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
    dt_zps = cf.optimize_df(dt_zps)
    dt_hot = dt_zps

    dt_zps = store['dt_zps']
    if m1_diam is not None:
        dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
    dt_zps = cf.optimize_df(dt_zps)

    dt_ois['MU'] = dt_ois.mag/(dt_ois.A_IMAGE*dt_ois.B_IMAGE)
    dt_zps['MU'] = dt_zps.mag/(dt_zps.A_IMAGE*dt_zps.B_IMAGE)
    dt_hot['MU'] = dt_hot.mag/(dt_hot.A_IMAGE*dt_hot.B_IMAGE)
    dt_sps['MU'] = dt_sps.mag/(dt_sps.a*dt_sps.b)

    dt_ois['SN'] = dt_ois.FLUX_APER/dt_ois.FLUXERR_APER
    dt_zps['SN'] = dt_zps.FLUX_APER/dt_zps.FLUXERR_APER
    dt_hot['SN'] = dt_hot.FLUX_APER/dt_hot.FLUXERR_APER
    dt_sps['SN'] = dt_sps.cflux/np.sqrt(dt_sps.cflux)

    merged = store['merged']
    selected = merged[merged.selected==True]

# =============================================================================
# Usar los seleccionados desde la tabla merged
# =============================================================================

    ids = selected['image_id_zps'].drop_duplicates().values
    dt_zps = dt_zps.loc[dt_zps['image_id'].isin(ids)].drop_duplicates()

    ids = selected['image_id_sps'].drop_duplicates().values
    dt_sps = dt_sps.loc[dt_sps['image_id'].isin(ids)].drop_duplicates()

    ids = selected['image_id_ois'].drop_duplicates().values
    dt_ois = dt_ois.loc[dt_ois['image_id'].isin(ids)].drop_duplicates()

    ids = selected['image_id_hot'].drop_duplicates().values
    dt_hot = dt_hot.loc[dt_hot['image_id'].isin(ids)].drop_duplicates()

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

    scols = ['thresh', 'npix', 'tnpix', 'xmin_col', 'xmax_col', 'ymin', 'ymax',
            'x', 'y', 'x2', 'y2', 'xy', 'errx2', 'erry2', 'errxy', 'a', 'b',
            'theta', 'cxx', 'cyy', 'cxy', 'cflux', 'flux', 'cpeak', 'peak',
            'xcpeak', 'ycpeak', 'xpeak', 'ypeak', 'flag', 'DELTAX', 'DELTAY',
            'RATIO', 'ROUNDNESS', 'PEAK_CENTROID', 'ref_fwhm', 'new_fwhm',
            'px_scale', 'ref_back_sbright', 'new_back_sbright',
            'exp_time', 'MAG_APER', 'MAG_ISO', 'VALID_MAG',
            'mean_offset', 'slope', 'mag', 'VALID_MAG_iso', 'mag_iso',
            'goyet_iso', 'mean_goyet', 'mean_goyet_iso', 'MU', 'SN']

    target = ['IS_REAL']

# =============================================================================
# Para que entre en memoria hacemos un sampling de esto
# =============================================================================
    n_samples = 45000

    d_ois = dt_ois[cols+target].sample(n_samples).dropna()
    d_zps = dt_zps[cols+target].sample(n_samples).dropna()
    d_hot = dt_hot[cols+target].sample(n_samples).dropna()
    d_sps = dt_sps[scols+target].sample(n_samples).dropna()

    y_zps = d_zps[target]
    y_ois = d_ois[target]
    y_sps = d_sps[target]
    y_hot = d_hot[target]

    d_ois = d_ois[cols]
    d_zps = d_zps[cols]
    d_sps = d_sps[scols]
    d_hot = d_hot[cols]

# =============================================================================
# Ahora que tengo los datos seleccionados hago preprocessing general
# =============================================================================
    scaler_ois = preprocessing.StandardScaler().fit(d_ois)
    scaler_zps = preprocessing.StandardScaler().fit(d_zps)
    scaler_hot = preprocessing.StandardScaler().fit(d_hot)
    scaler_sps = preprocessing.StandardScaler().fit(d_sps)

    X_ois = scaler_ois.transform(d_ois)
    X_zps = scaler_zps.transform(d_zps)
    X_hot = scaler_hot.transform(d_hot)
    X_sps = scaler_sps.transform(d_sps)


# =============================================================================
# Analisis univariado
# =============================================================================

# %%%%%  Variance threshold
    from sklearn.feature_selection import VarianceThreshold

    thresh = 0.1
    sel = VarianceThreshold(threshold=thresh)

    X_ois = sel.fit_transform(X_ois)
    newcols_ois = d_ois.columns[sel.get_support()]
    print('Dropped columns = {}'.format(d_ois.columns[~sel.get_support()]))

    X_zps = sel.fit_transform(X_zps)
    newcols_zps = d_zps.columns[sel.get_support()]
    print('Dropped columns = {}'.format(d_zps.columns[~sel.get_support()]))

    X_sps = sel.fit_transform(X_sps)
    newcols_sps = d_sps.columns[sel.get_support()]
    print('Dropped columns = {}'.format(d_sps.columns[~sel.get_support()]))

    X_hot = sel.fit_transform(X_hot)
    newcols_hot = d_hot.columns[sel.get_support()]
    print('Dropped columns = {}'.format(d_hot.columns[~sel.get_support()]))


# =============================================================================
# KNN Neighbors
# =============================================================================
# %%%%%  Univariate f_mutual_info_classif

    percentile = 20.

    scores, selector, selected_cols = cf.select(X_ois, y_ois, percentile)
    scoring_ois = pd.DataFrame(scores, index=newcols_ois, columns=['ois'])
    selection_ois = scoring_ois.loc[newcols_ois.values[selected_cols][0]]

    scores, selector, selected_cols = cf.select(X_zps, y_zps, percentile)
    scoring_zps = pd.DataFrame(scores, index=newcols_zps, columns=['zps'])
    selection_zps = scoring_zps.loc[newcols_zps.values[selected_cols][0]]

    scores, selector, selected_cols = cf.select(X_hot, y_hot, percentile)
    scoring_hot = pd.DataFrame(scores, index=newcols_hot, columns=['hot'])
    selection_hot = scoring_hot.loc[newcols_hot.values[selected_cols][0]]

    scores, selector, selected_cols = cf.select(X_sps, y_sps, percentile)
    scoring_sps = pd.DataFrame(scores, index=newcols_sps, columns=['sps'])
    selection_sps = scoring_sps.loc[newcols_sps.values[selected_cols][0]]
    scoring_sps = scoring_sps.rename(index=cf.transl)
    selection_sps = selection_sps.rename(index=cf.transl)

    scoring = pd.concat([scoring_ois, scoring_zps, scoring_hot, scoring_sps], axis=1)
    scoring = scoring.fillna(0)
    selected = pd.concat([selection_ois, selection_zps, selection_hot, selection_sps], axis=1)
    selected = selected.reindex(scoring.index)

    sns.heatmap(scoring, vmin=0., vmax=1., cmap='Blues',
                annot=np.round(selected.fillna(-1).values, 2), cbar=True)
    plt.savefig(os.path.join(plots_path, 'select_percentile_mutual_info_heatmap.png'))
    plt.clf()

    #  Selected features are then
    features = list(selection_ois.index)
    features += list(selection_zps.index)
    features += list(selection_sps.index)
    features += list(selection_hot.index)
    features = np.unique(features)


# =============================================================================
# RandomForests
# =============================================================================
# =============================================================================
# Get the correlations out first, as suggested by referee
# =============================================================================
    corr = pd.DataFrame(X_ois, columns=newcols_ois).corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    corr_plot = sns.heatmap(corr, mask=mask, cmap='RdBu', center=0,
                    square=True, linewidths=.2, cbar_kws={"shrink": .5})
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'corr_ois.png'))
    plt.clf()

    # remove corr columns
    correlated_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                colname = corr.columns[i]
                correlated_features.add(colname)

    decorr_ois = pd.DataFrame(X_ois, columns=newcols_ois).drop(correlated_features, axis=1)

    corr = decorr_ois.corr()
        # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    corr_plot = sns.heatmap(corr, mask=mask, cmap='RdBu', center=0,
                    square=True, linewidths=.2, cbar_kws={"shrink": .5})
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'decorr_ois.png'))
    plt.clf()

    # -------------------------- #
    corr = pd.DataFrame(X_hot, columns=newcols_hot).corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    corr_plot = sns.heatmap(corr, mask=mask, cmap='RdBu', center=0,
                    square=True, linewidths=.2, cbar_kws={"shrink": .5})
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'corr_hot.png'))
    plt.clf()

    # remove corr columns
    correlated_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                colname = corr.columns[i]
                correlated_features.add(colname)

    decorr_hot = pd.DataFrame(X_hot, columns=newcols_hot).drop(correlated_features, axis=1)

    corr = decorr_hot.corr()
        # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    corr_plot = sns.heatmap(corr, mask=mask, cmap='RdBu', center=0,
                    square=True, linewidths=.2, cbar_kws={"shrink": .5})
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'decorr_hot.png'))
    plt.clf()

    # -------------------------- #
    corr = pd.DataFrame(X_zps, columns=newcols_zps).corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    corr_plot = sns.heatmap(corr, mask=mask, cmap='RdBu', center=0,
                    square=True, linewidths=.2, cbar_kws={"shrink": .5})
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'corr_zps.png'))
    plt.clf()

    # remove corr columns
    correlated_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                colname = corr.columns[i]
                correlated_features.add(colname)

    decorr_zps = pd.DataFrame(X_zps, columns=newcols_zps).drop(correlated_features, axis=1)

    corr = decorr_zps.corr()
        # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    corr_plot = sns.heatmap(corr, mask=mask, cmap='RdBu', center=0,
                    square=True, linewidths=.2, cbar_kws={"shrink": .5})
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'decorr_zps.png'))
    plt.clf()

    # -------------------------- #
    corr = pd.DataFrame(X_sps, columns=newcols_sps).corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    corr_plot = sns.heatmap(corr, mask=mask, cmap='RdBu', center=0,
                    square=True, linewidths=.2, cbar_kws={"shrink": .5})
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'corr_sps.png'))
    plt.clf()

    # remove corr columns
    correlated_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                colname = corr.columns[i]
                correlated_features.add(colname)

    decorr_sps = pd.DataFrame(X_sps, columns=newcols_sps).drop(correlated_features, axis=1)

    corr = decorr_sps.corr()
        # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    corr_plot = sns.heatmap(corr, mask=mask, cmap='RdBu', center=0,
                    square=True, linewidths=.2, cbar_kws={"shrink": .5})
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'decorr_sps.png'))
    plt.clf()

    full_cols = list(decorr_ois.index)
    full_cols += list(decorr_zps.index)
    full_cols += list(decorr_hot.index)
    full_cols += list(decorr_sps.index)
    full_cols = np.unique(full_cols)

    # datasets are now in decorr frames...

    ois_importance = cf.importance_perm_kfold(decorr_ois.values, y_ois.values.ravel(),
        RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
        cols=decorr_ois.columns, method='Bramich')

    zps_importance = cf.importance_perm_kfold(decorr_zps.values, y_zps.values.ravel(),
        RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
        cols=decorr_zps.columns, method='Zackay')

    hot_importance = cf.importance_perm_kfold(decorr_hot.values, y_hot.values.ravel(),
        RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
        cols=decorr_hot.columns, method='Alard-Lupton')

    sps_importance = cf.importance_perm_kfold(decorr_sps.values, y_sps.values.ravel(),
        RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
        cols=decorr_sps.columns, method='Scorr')

    sps_newimp = []
    for animp in sps_importance:
        sps_newimp.append(animp.rename(index=cf.transl))

    res_ois = pd.concat(ois_importance, axis=1)
    res_zps = pd.concat(zps_importance, axis=1)
    res_hot = pd.concat(hot_importance, axis=1)
    res_sps = pd.concat(sps_newimp, axis=1)

    full_cols = list(full_cols).extend(['Random'])
    m = pd.concat([res_ois.mean(axis=1),
                   res_zps.mean(axis=1),
                   res_hot.mean(axis=1),
                   res_sps.mean(axis=1)], axis=1).reindex(full_cols)
    s = pd.concat([res_ois.std(axis=1),
                   res_zps.std(axis=1),
                   res_hot.std(axis=1),
                   res_sps.std(axis=1)], axis=1).reindex(full_cols)

    m2 = m.copy()
    for i in range(4):
        m2[i] = m2[i] - m2[i]['Random']
        m2[i] = m2[i]/np.max(m2[i])

    plt.figure(figsize=(6, 12))
    plt.imshow(m2.fillna(0).values, vmin=0, vmax=1., cmap='gray_r', aspect='auto')
    plt.yticks(np.arange(len(m)), m.index, rotation='horizontal')
    plt.xticks(np.arange(4)+0.5, ['Bramich', 'Zackay', 'A.-Lupton', 'Scorr'], rotation=45)
    plt.colorbar()
    plt.savefig(os.path.join(plots_path, 'feature_heatmap_simdata.svg'),
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(3, 12))
    sns.heatmap(m.fillna(0).values, robust=True, cmap='Greys', cbar=True)
    plt.yticks(np.arange(len(m))+0.5, rotation='horizontal')
    plt.xticks(np.arange(4)+0.5, ['Bramich', 'Zackay', 'A.-Lupton', 'Scorr'], rotation=45)
    #plt.colorbar()
    plt.savefig(os.path.join(plots_path, './feature_heatmap_simdata.pdf'),
                bbox_inches='tight')
    plt.close()



# =============================================================================
# Support Vector Machines
# =============================================================================

    #  here we will use recursive feature elimination
    ## ois
    svc = SVC(kernel='linear',
              cache_size=2048,
              class_weight='balanced',
              probability=False)

    svc = svm.LinearSVC(dual=False, tol=1e-5)

    rfecv = feature_selection.RFECV(estimator=svc, step=1, cv=StratifiedKFold(6),
                  scoring='accuracy', n_jobs=-1)

    rfecv.fit(np.ascontiguousarray(X_ois), y_ois)
    print("Optimal number of features : %d" % rfecv.n_features_)
    print(newcols_ois[rfecv.support_])

    # hot
    rfecv.fit(np.ascontiguousarray(X_hot), y_hot)
    print("Optimal number of features : %d" % rfecv.n_features_)
    print(newcols_hot[rfecv.support_])

    # zps
    rfecv.fit(np.ascontiguousarray(X_zps), y_zps)
    print("Optimal number of features : %d" % rfecv.n_features_)
    print(newcols_zps[rfecv.support_])

    # sps
    rfecv.fit(np.ascontiguousarray(X_sps), y_sps)
    print("Optimal number of features : %d" % rfecv.n_features_)
    print(newcols_sps[rfecv.support_])




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--m1_diam", help="diameter to filter",
                        default=None)
    parser.add_argument("path", help="path to plot files")
    args = parser.parse_args()

    import sys
    sys.exit(main(args.m1_diam, args.path))
