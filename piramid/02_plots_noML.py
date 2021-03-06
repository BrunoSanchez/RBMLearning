#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  02_plots_noML.py
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
import gc
import matplotlib
matplotlib.use('Agg')
from sqlalchemy import create_engine

import numpy as np
import seaborn as sns
import pandas as pd

from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt

import custom_funs as cf


storefile = '/mnt/clemente/bos0109/table_store2.h5'

store = pd.HDFStore(storefile, mode='r+', complevel=5)
store.open()

#sns.set_context(font_scale=16)
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams['text.usetex'] = False


def main(m1_diam=1.54, plots_path='./plots/.', store_flush=False,
         goyet_vs_pars=False):
    plot_dir = os.path.abspath(plots_path)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    # make platescale less than 1.3 (discard px_scale=1.4)

    simulated = store['simulated']
    simulations = store['simulations']

    simulations = simulations[simulations.failed_to_subtract==False]
    #simulations = simulations[simulations.px_scale<1.3]

    if m1_diam is not None:
        simulations = simulations[simulations.m1_diam==m1_diam]

    simus = pd.merge(left=simulations, right=simulated,
                     right_on='simulation_id', left_on='id', how='inner')
    simus.drop_duplicates(inplace=True)
    simus.drop('executed', axis=1, inplace=True)
    simus.drop('scorrimage_id', axis=1, inplace=True)
    simus.drop('loaded', axis=1, inplace=True)
    simus.drop('crossmatched', axis=1, inplace=True)
    simus.drop('possible_saturation', axis=1, inplace=True)
    # =============================================================================
    # tables
    # =============================================================================
    dt_zps = store['dt_ois']
    if m1_diam is not None:
        dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
    #dt_zps = dt_zps[dt_zps.px_scale<1.3]
    #dt_zps = cf.optimize_df(dt_zps)
    dt_ois = dt_zps

    dt_zps = store['dt_sps']
    if m1_diam is not None:
        dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
    #dt_zps = dt_zps[dt_zps.px_scale<1.3]
    #dt_zps = cf.optimize_df(dt_zps)
    dt_sps = dt_zps

    dt_zps = store['dt_hot']
    if m1_diam is not None:
        dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
    #dt_zps = dt_zps[dt_zps.px_scale<1.3]
    #dt_zps = cf.optimize_df(dt_zps)
    dt_hot = dt_zps

    dt_zps = store['dt_zps']
    if m1_diam is not None:
        dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
    #dt_zps = dt_zps[dt_zps.px_scale<1.3]
    #dt_zps = cf.optimize_df(dt_zps)
    # import ipdb; ipdb.set_trace()
# =============================================================================
# plot de deltas de magnitud
# =============================================================================
    plt.figure(figsize=(9,3))
    plt.title('mag offsets from mag simulated')
    plt.subplot(141)
    dmag = dt_zps.sim_mag - dt_zps.mag
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag zps')

    plt.subplot(142)
    dmag = dt_ois.sim_mag - dt_ois.mag
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag ois')

    plt.subplot(143)
    dmag = dt_hot.sim_mag - dt_hot.mag
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag hot')

    plt.subplot(144)
    dmag = dt_sps.sim_mag - dt_sps.mag
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag sps')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'delta_mags.svg'), dpi=400)
    plt.clf()
    plt.close()
#~ # =============================================================================
#~ # plot de deltas de magnitud sobre magnitud (goyet)
#~ # =============================================================================
    #~ plt.figure(figsize=(9,3))
    #~ plt.title('mag offsets over mag simulated')
    #~ plt.subplot(141)
    #~ dmag = dt_zps.goyet
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('delta mag zps')

    #~ plt.subplot(142)
    #~ dmag = dt_ois.goyet
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('delta mag ois')

    #~ plt.subplot(143)
    #~ dmag = dt_hot.goyet
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('delta mag hot')

    #~ plt.subplot(144)
    #~ dmag = dt_sps.goyet
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('delta mag sps')

    #~ plt.tight_layout()
    #~ plt.savefig(os.path.join(plot_dir, 'goyet_full.svg'), dpi=400)
    #~ plt.clf()

#~ # =============================================================================
#~ # plot de deltas de magnitud sobre magnitud (goyet)
#~ # =============================================================================
    #~ plt.figure(figsize=(9,3))
    #~ plt.title('mag offsets over mag simulated')
    #~ plt.subplot(141)
    #~ dmag = dt_zps.goyet_iso
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('delta mag zps')

    #~ plt.subplot(142)
    #~ dmag = dt_ois.goyet_iso
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('delta mag ois')

    #~ plt.subplot(143)
    #~ dmag = dt_hot.goyet_iso
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('delta mag hot')

    #~ plt.subplot(144)
    #~ dmag = dt_sps.goyet_iso
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('delta mag sps')

    #~ plt.tight_layout()
    #~ plt.savefig(os.path.join(plot_dir, 'goyet_iso_full.svg'), dpi=400)
    #~ plt.clf()


# =============================================================================
#   Percentiles de la calibracion
# =============================================================================
    #~ pars = ['image_id', 'p05', 'p95']

    #~ cals = cf.cal_mags(dt_zps)
    #~ dt_zps = pd.merge(dt_zps, cals[pars], on='image_id', how='left')

    #~ cals = cf.cal_mags(dt_sps)
    #~ dt_sps = pd.merge(dt_sps, cals[pars], on='image_id', how='left')

    #~ cals = cf.cal_mags(dt_hot)
    #~ dt_hot = pd.merge(dt_hot, cals[pars], on='image_id', how='left')

    #~ cals = cf.cal_mags(dt_ois)
    #~ dt_ois = pd.merge(dt_ois, cals[pars], on='image_id', how='left')

    bins = np.arange(7, 26.5, 0.5)
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('zackay')
    in_range = (dt_zps.mag > dt_zps.p05) & (dt_zps.mag < dt_zps.p95)
    in_mags = dt_zps.loc[in_range].sim_mag.dropna()
    out_mags = dt_zps.loc[~in_range].sim_mag.dropna()
    plt.hist(simus.app_mag, bins=bins, histtype='step', color='k',
             label='simulated', log=True)
    plt.hist(in_mags, bins=bins, alpha=0.5, label='inliers', stacked=True)
    plt.hist(out_mags, bins=bins, alpha=0.5, label='outliers', stacked=True)
    plt.legend(loc='best')

    plt.subplot(222)
    plt.title('scorr')
    in_range = (dt_sps.mag > dt_sps.p05) & (dt_sps.mag < dt_sps.p95)
    in_mags = dt_sps.loc[in_range].sim_mag.dropna()
    out_mags = dt_sps.loc[~in_range].sim_mag.dropna()
    plt.hist(simus.app_mag, bins=bins, histtype='step', color='k',
             label='simulated', log=True)
    plt.hist(in_mags, bins=bins, alpha=0.5, label='inliers', stacked=True)
    plt.hist(out_mags, bins=bins, alpha=0.5, label='outliers', stacked=True)
    plt.legend(loc='best')

    plt.subplot(223)
    plt.title('hotpants')
    in_range = (dt_hot.mag > dt_hot.p05) & (dt_hot.mag < dt_hot.p95)
    in_mags = dt_hot.loc[in_range].sim_mag.dropna()
    out_mags = dt_hot.loc[~in_range].sim_mag.dropna()
    plt.hist(simus.app_mag, bins=bins, histtype='step', color='k',
             label='simulated', log=True)
    plt.hist(in_mags, bins=bins, alpha=0.5, label='inliers', stacked=True)
    plt.hist(out_mags, bins=bins, alpha=0.5, label='outliers', stacked=True)
    plt.legend(loc='best')

    plt.subplot(224)
    plt.title('bramich')
    in_range = (dt_ois.mag > dt_ois.p05) & (dt_ois.mag < dt_ois.p95)
    in_mags = dt_ois.loc[in_range].sim_mag.dropna()
    out_mags = dt_ois.loc[~in_range].sim_mag.dropna()
    plt.hist(simus.app_mag, bins=bins, histtype='step', color='k',
             label='simulated', log=True)
    plt.hist(in_mags, bins=bins, alpha=0.5, label='inliers', stacked=True)
    plt.hist(out_mags, bins=bins, alpha=0.5, label='outliers', stacked=True)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'inliers_range.svg'), dpi=400)
    plt.clf()
    plt.close()
# =============================================================================
# Como quedan los diagramas de error de magnitud vs magnitud simulada
# =============================================================================
# =============================================================================
# inliers
# =============================================================================
    dm = 0
    plt.figure(figsize=(8,4))
    bins = np.arange(6.5, 26.5, .5)
    #~ ff = subset_hot_lo.FLAGS<=0
    ff = (dt_hot.mag > dt_hot.p05+dm) & (dt_hot.mag < dt_hot.p95)
    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_hot.loc[ff], bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='g--', label='Hotpants')

    #ff = subset_sps_hi.FLAGS<=1
    ff = (dt_sps.mag > dt_sps.p05+dm) & (dt_sps.mag < dt_sps.p95)
    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_sps.loc[ff], bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='m:', label='Scorr')

    #~ ff = subset_zps_lo.FLAGS<=0
    ff = (dt_zps.mag > dt_zps.p05+dm) & (dt_zps.mag < dt_zps.p95)
    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_zps.loc[ff], bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='b.-', label='Zackay')

    #~ ff = subset_ois_lo.FLAGS<=0
    ff = (dt_ois.mag > dt_ois.p05+dm) & (dt_ois.mag < dt_ois.p95)
    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_ois.loc[ff], bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='ro-', label='Bramich')

    plt.tick_params(labelsize=16)
    plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    plt.xlabel('Sim Mag', fontsize=16)
    plt.title('Simulated Data', fontsize=14)
    plt.legend(loc='best', fontsize=14)

    plt.xlim(10, 22.5)
    plt.ylim(-2, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_inliers.svg'),
                format='svg', dpi=480)
    plt.close()
#~ # =============================================================================
#~ # 300 segundos
#~ # =============================================================================
    #~ dm = 1.5
    #~ plt.figure(figsize=(8,4))
    #~ bins = np.arange(6.5, 26.5, .5)
    #ff = subset_hot_lo.FLAGS<=0
    #~ ff = (dt_hot.mag > dt_hot.p05+dm) & (dt_hot.mag < dt_hot.p95) & (dt_hot.exp_time==300)
    #~ mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_hot.loc[ff], bins=bins)
    #~ plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='g--', label='Hotpants')

    #~ #ff = subset_sps_hi.FLAGS<=1
    #~ ff = (dt_sps.mag > dt_sps.p05+dm) & (dt_sps.mag < dt_sps.p95) & (dt_sps.exp_time==300)
    #~ mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_sps.loc[ff], bins=bins)
    #~ plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='m:', label='Scorr')

    #ff = subset_zps_lo.FLAGS<=0
    #~ ff = (dt_zps.mag > dt_zps.p05+dm) & (dt_zps.mag < dt_zps.p95) & (dt_zps.exp_time==300)
    #~ mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_zps.loc[ff], bins=bins)
    #~ plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='b.-', label='Zackay')

    #ff = subset_ois_lo.FLAGS<=0
    #~ ff = (dt_ois.mag > dt_ois.p05+dm) & (dt_ois.mag < dt_ois.p95) & (dt_ois.exp_time==300)
    #~ mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_ois.loc[ff], bins=bins)
    #~ plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='ro-', label='Bramich')

    #~ plt.tick_params(labelsize=16)
    #~ plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    #~ plt.xlabel('Sim Mag', fontsize=16)
    #~ plt.title('Simulated Data', fontsize=14)
    #~ plt.legend(loc='best', fontsize=14)

    #~ plt.xlim(10, 22.5)
    #~ plt.ylim(-2, 3)
    #~ plt.tight_layout()
    #~ plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_inliers_300s.svg'),
                #~ format='svg', dpi=480)

#~ # =============================================================================
#~ # 300 segundos
#~ # =============================================================================
    #~ dm = 0
    #~ means = []
    #~ plt.figure(figsize=(8,4))
    #~ bins = np.arange(6.5, 26.5, .5)
    #ff = subset_hot_lo.FLAGS<=0
    #~ ff = (dt_hot.mag > dt_hot.p05+dm) & (dt_hot.mag < dt_hot.p95) & (dt_hot.exp_time==300)
    #~ means.append(cf.binning_res(dt_hot.loc[ff], bins=bins))

    #~ #ff = subset_sps_hi.FLAGS<=1
    #~ ff = (dt_sps.mag > dt_sps.p05+dm) & (dt_sps.mag < dt_sps.p95) & (dt_sps.exp_time==300)
    #~ means.append(cf.binning_res(dt_sps.loc[ff], bins=bins))

    #ff = subset_zps_lo.FLAGS<=0
    #~ ff = (dt_zps.mag > dt_zps.p05+dm) & (dt_zps.mag < dt_zps.p95) & (dt_zps.exp_time==300)
    #~ means.append(cf.binning_res(dt_zps.loc[ff], bins=bins))

    #ff = subset_ois_lo.FLAGS<=0
    #~ ff = (dt_ois.mag > dt_ois.p05+dm) & (dt_ois.mag < dt_ois.p95) & (dt_ois.exp_time==300)
    #~ mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_ois.loc[ff], bins=bins)

    #~ mm = np.ma.masked_invalid(means)
    #~ bin_centers = mm.max(axis=0)[3]
    #~ mean = np.sum(mm[:, 0, :]*mm[:, 2, :]**2, axis=0)/np.sum(mm[:, 2, :]**2, axis=0)
    #~ stds = np.sqrt(np.sum((mm[:, 1, :]**2)/(mm[:, 2, :]**2), axis=0))
    #~ plt.errorbar(bin_centers, mean, yerr=stds, fmt='--', label='mean')

    #~ plt.tick_params(labelsize=16)
    #~ plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    #~ plt.xlabel('Sim Mag', fontsize=16)
    #~ plt.title('Simulated Data', fontsize=14)
    #~ plt.legend(loc='best', fontsize=14)

    #~ plt.xlim(10, 22.5)
    #~ plt.ylim(-2, 3)
    #~ plt.tight_layout()
    #~ plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_inliers_300s_averaged.svg'),
                #~ format='svg', dpi=480)


#~ # =============================================================================
#~ # 120 segundos
#~ # =============================================================================
    #~ dm = 0
    #~ plt.figure(figsize=(8,4))
    #~ bins = np.arange(6.5, 26.5, .5)
    #ff = subset_hot_lo.FLAGS<=0
    #~ ff = (dt_hot.mag > dt_hot.p05+dm) & (dt_hot.mag < dt_hot.p95) & (dt_hot.exp_time==120)
    #~ mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_hot.loc[ff], bins=bins)
    #~ plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='g--', label='Hotpants')

    #~ #ff = subset_sps_hi.FLAGS<=1
    #~ ff = (dt_sps.mag > dt_sps.p05+dm) & (dt_sps.mag < dt_sps.p95) & (dt_sps.exp_time==120)
    #~ mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_sps.loc[ff], bins=bins)
    #~ plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='m:', label='Scorr')

    #ff = subset_zps_lo.FLAGS<=0
    #~ ff = (dt_zps.mag > dt_zps.p05+dm) & (dt_zps.mag < dt_zps.p95) & (dt_zps.exp_time==120)
    #~ mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_zps.loc[ff], bins=bins)
    #~ plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='b.-', label='Zackay')

    #ff = subset_ois_lo.FLAGS<=0
    #~ ff = (dt_ois.mag > dt_ois.p05+dm) & (dt_ois.mag < dt_ois.p95) & (dt_ois.exp_time==120)
    #~ mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_ois.loc[ff], bins=bins)
    #~ plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='ro-', label='Bramich')

    #~ plt.tick_params(labelsize=16)
    #~ plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    #~ plt.xlabel('Sim Mag', fontsize=16)
    #~ plt.title('Simulated Data', fontsize=14)
    #~ plt.legend(loc='best', fontsize=14)

    #~ plt.xlim(10, 22.5)
    #~ plt.ylim(-2, 3)
    #~ plt.tight_layout()
    #~ plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_inliers_120s.svg'),
                #~ format='svg', dpi=480)

#~ # =============================================================================
#~ # 120 segundos
#~ # =============================================================================
    #~ dm = 0
    #~ means = []
    #~ plt.figure(figsize=(8,4))
    #~ bins = np.arange(6.5, 26.5, .5)
    #ff = subset_hot_lo.FLAGS<=0
    #~ ff = (dt_hot.mag > dt_hot.p05+dm) & (dt_hot.mag < dt_hot.p95) & (dt_hot.exp_time==120)
    #~ means.append(cf.binning_res(dt_hot.loc[ff], bins=bins))

    #~ #ff = subset_sps_hi.FLAGS<=1
    #~ ff = (dt_sps.mag > dt_sps.p05+dm) & (dt_sps.mag < dt_sps.p95) & (dt_sps.exp_time==120)
    #~ means.append(cf.binning_res(dt_sps.loc[ff], bins=bins))

    #ff = subset_zps_lo.FLAGS<=0
    #~ ff = (dt_zps.mag > dt_zps.p05+dm) & (dt_zps.mag < dt_zps.p95) & (dt_zps.exp_time==120)
    #~ means.append(cf.binning_res(dt_zps.loc[ff], bins=bins))

    #ff = subset_ois_lo.FLAGS<=0
    #~ ff = (dt_ois.mag > dt_ois.p05+dm) & (dt_ois.mag < dt_ois.p95) & (dt_ois.exp_time==120)
    #~ mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_ois.loc[ff], bins=bins)

    #~ mm = np.ma.masked_invalid(means)
    #~ bin_centers = mm.max(axis=0)[3]
    #~ mean = np.sum(mm[:, 0, :]*mm[:, 2, :]**2, axis=0)/np.sum(mm[:, 2, :]**2, axis=0)
    #~ stds = np.sqrt(np.sum((mm[:, 1, :]**2)/(mm[:, 2, :]**2), axis=0))
    #~ plt.errorbar(bin_centers, mean, yerr=stds, fmt='--', label='mean')

    #~ plt.tick_params(labelsize=16)
    #~ plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    #~ plt.xlabel('Sim Mag', fontsize=16)
    #~ plt.title('Simulated Data', fontsize=14)
    #~ plt.legend(loc='best', fontsize=14)

    #~ plt.xlim(10, 22.5)
    #~ plt.ylim(-2, 3)
    #~ plt.tight_layout()
    #~ plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_inliers_120s_averaged.svg'),
                #~ format='svg', dpi=480)

# =============================================================================
# 60 segundos
# =============================================================================
    dm = 0
    plt.figure(figsize=(8,4))
    bins = np.arange(6.5, 26.5, .5)
    #~ ff = subset_hot_lo.FLAGS<=0
    ff = (dt_hot.mag > dt_hot.p05+dm) & (dt_hot.mag < dt_hot.p95) & (dt_hot.exp_time==60)
    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_hot.loc[ff], bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='g--', label='Hotpants')

    #ff = subset_sps_hi.FLAGS<=1
    ff = (dt_sps.mag > dt_sps.p05+dm) & (dt_sps.mag < dt_sps.p95) & (dt_sps.exp_time==60)
    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_sps.loc[ff], bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='m:', label='Scorr')

    #~ ff = subset_zps_lo.FLAGS<=0
    ff = (dt_zps.mag > dt_zps.p05+dm) & (dt_zps.mag < dt_zps.p95) & (dt_zps.exp_time==60)
    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_zps.loc[ff], bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='b.-', label='Zackay')

    #~ ff = subset_ois_lo.FLAGS<=0
    ff = (dt_ois.mag > dt_ois.p05+dm) & (dt_ois.mag < dt_ois.p95) & (dt_ois.exp_time==60)
    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_ois.loc[ff], bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='ro-', label='Bramich')

    plt.tick_params(labelsize=16)
    plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    plt.xlabel('Sim Mag', fontsize=16)
    plt.title('Simulated Data', fontsize=14)
    plt.legend(loc='best', fontsize=14)

    plt.xlim(10, 22.5)
    plt.ylim(-2, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_inliers_60s.svg'),
                format='svg', dpi=480)
    plt.close()
# =============================================================================
# 60 segundos
# =============================================================================
    dm = 0
    means = []
    plt.figure(figsize=(8,4))
    bins = np.arange(6.5, 26.5, .5)
    #~ ff = subset_hot_lo.FLAGS<=0
    ff = (dt_hot.mag > dt_hot.p05+dm) & (dt_hot.mag < dt_hot.p95) & (dt_hot.exp_time==60)
    means.append(cf.binning_res(dt_hot.loc[ff], bins=bins))

    #ff = subset_sps_hi.FLAGS<=1
    ff = (dt_sps.mag > dt_sps.p05+dm) & (dt_sps.mag < dt_sps.p95) & (dt_sps.exp_time==60)
    means.append(cf.binning_res(dt_sps.loc[ff], bins=bins))

    #~ ff = subset_zps_lo.FLAGS<=0
    ff = (dt_zps.mag > dt_zps.p05+dm) & (dt_zps.mag < dt_zps.p95) & (dt_zps.exp_time==60)
    means.append(cf.binning_res(dt_zps.loc[ff], bins=bins))

    #~ ff = subset_ois_lo.FLAGS<=0
    ff = (dt_ois.mag > dt_ois.p05+dm) & (dt_ois.mag < dt_ois.p95) & (dt_ois.exp_time==60)
    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_ois.loc[ff], bins=bins)

    mm = np.ma.masked_invalid(means)
    bin_centers = mm.max(axis=0)[3]
    mean = np.sum(mm[:, 0, :]*mm[:, 2, :]**2, axis=0)/np.sum(mm[:, 2, :]**2, axis=0)
    stds = np.sqrt(np.sum((mm[:, 1, :]**2)/(mm[:, 2, :]**2), axis=0))
    plt.errorbar(bin_centers, mean, yerr=stds, fmt='--', label='mean')

    plt.tick_params(labelsize=16)
    plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    plt.xlabel('Sim Mag', fontsize=16)
    plt.title('Simulated Data', fontsize=14)
    plt.legend(loc='best', fontsize=14)

    plt.xlim(10, 22.5)
    plt.ylim(-2, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_inliers_60s_averaged.svg'),
                format='svg', dpi=480)
    plt.close()
# =============================================================================
# 60 segundos
# =============================================================================
    dm = 0
    means = []
    plt.figure(figsize=(8,4))
    bins = np.arange(6.5, 26.5, .5)
    #~ ff = subset_hot_lo.FLAGS<=0
    ff = (dt_hot.mag > dt_hot.p05+dm) & (dt_hot.mag < dt_hot.p95) & (dt_hot.exp_time==60)
    means.append(cf.binning_res(dt_hot.loc[ff], bins=bins))

    #ff = subset_sps_hi.FLAGS<=1
    ff = (dt_sps.mag > dt_sps.p05+dm) & (dt_sps.mag < dt_sps.p95) & (dt_sps.exp_time==60)
    means.append(cf.binning_res(dt_sps.loc[ff], bins=bins))

    #~ ff = subset_zps_lo.FLAGS<=0
    ff = (dt_zps.mag > dt_zps.p05+dm) & (dt_zps.mag < dt_zps.p95) & (dt_zps.exp_time==60)
    means.append(cf.binning_res(dt_zps.loc[ff], bins=bins))

    #~ ff = subset_ois_lo.FLAGS<=0
    ff = (dt_ois.mag > dt_ois.p05+dm) & (dt_ois.mag < dt_ois.p95) & (dt_ois.exp_time==60)
    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(dt_ois.loc[ff], bins=bins)

    mm = np.ma.masked_invalid(means)
    bin_centers = mm.max(axis=0)[3]
    mean = np.sum(mm[:, 0, :]*mm[:, 2, :]**2, axis=0)/np.sum(mm[:, 2, :]**2, axis=0)
    stds = np.sqrt(np.sum((mm[:, 1, :]**2)/(mm[:, 2, :]**2), axis=0))

    #plt.errorbar(bin_centers, mean, yerr=stds, fmt='--', label='mean')
    plt.plot(bin_centers, mean, '--', label='mean')
    plt.fill_between(bin_centers, mean+stds, mean-stds, alpha=0.5, label='mean')

    plt.tick_params(labelsize=16)
    plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    plt.xlabel('Sim Mag', fontsize=16)
    plt.title('Simulated Data', fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.grid()
    plt.xlim(9, 22.5)
    plt.ylim(-2, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_inliers_60s_averaged_filled.svg'),
                format='svg', dpi=480)
    plt.close()

# =============================================================================
# Mean goyet alto
# =============================================================================
# =============================================================================
# Seleccionamos los mean_goyet
# =============================================================================
    pars = ['id', 'mean_goyet', 'image_id', 'id_simulation', 'mag', 'sim_mag', 'sim_id',
            'goyet', 'goyet_iso', 'mean_goyet_iso', 'IS_REAL', 'p05', 'p95']
    subset_zps = dt_zps[pars+['FLAGS', 'FLUX_APER', 'MAG_APER']]
    subset_ois = dt_ois[pars+['FLAGS', 'FLUX_APER', 'MAG_APER']]
    subset_sps = dt_sps[pars]
    subset_hot = dt_hot[pars+['FLAGS', 'FLUX_APER', 'MAG_APER']]

    del(dt_zps)
    del(dt_sps)
    del(dt_ois)
    del(dt_hot)
    gc.collect()

#~ # =============================================================================
#~ # Veamos la distrubicion general de las medias de los goyet
#~ # =============================================================================
    #~ plt.figure(figsize=(9,3))
    #~ plt.title('mean goyets for each technique')
    #~ plt.subplot(141)
    #~ dmag = subset_zps[['mean_goyet', 'image_id']].drop_duplicates().mean_goyet
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('mean goyet zps')

    #~ plt.subplot(142)
    #~ dmag = subset_ois[['mean_goyet', 'image_id']].drop_duplicates().mean_goyet
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('mean goyet ois')

    #~ plt.subplot(143)
    #~ dmag = subset_hot[['mean_goyet', 'image_id']].drop_duplicates().mean_goyet
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('mean goyet hot')

    #~ plt.subplot(144)
    #~ dmag = subset_sps[['mean_goyet', 'image_id']].drop_duplicates().mean_goyet
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('mean goyet sps')

    #~ plt.tight_layout()
    #~ plt.savefig(os.path.join(plot_dir, 'mean_goyets.svg'), dpi=400)
    #~ plt.clf()

#~ # =============================================================================
#~ # Veamos la distrubicion general de las medias de los goyet ISO
#~ # =============================================================================
    #~ plt.figure(figsize=(9,3))
    #~ plt.title('mean goyets_iso for each technique')
    #~ plt.subplot(141)
    #~ dmag = subset_zps[['mean_goyet_iso', 'image_id']].drop_duplicates().mean_goyet_iso
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('mean goyet_iso zps')

    #~ plt.subplot(142)
    #~ dmag = subset_ois[['mean_goyet_iso', 'image_id']].drop_duplicates().mean_goyet_iso
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('mean goyet_iso ois')

    #~ plt.subplot(143)
    #~ dmag = subset_hot[['mean_goyet_iso', 'image_id']].drop_duplicates().mean_goyet_iso
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('mean goyet_iso hot')

    #~ plt.subplot(144)
    #~ dmag = subset_sps[['mean_goyet_iso', 'image_id']].drop_duplicates().mean_goyet_iso
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('mean goyet_iso sps')

    #~ plt.tight_layout()
    #~ plt.savefig(os.path.join(plot_dir, 'mean_goyets_iso.svg'), dpi=400)
    #~ plt.clf()


# =============================================================================
# vetamos por mean goyet
# =============================================================================
    #subset_zps_hi = subset_zps[subset_zps.mean_goyet>=0.05]
    #subset_hot_hi = subset_hot[subset_hot.mean_goyet>=0.05]
    #subset_sps_hi = subset_sps[subset_sps.mean_goyet>=0.05]
    #subset_ois_hi = subset_ois[subset_ois.mean_goyet>=0.05]

#~ # =============================================================================
#~ # Como quedan las distros de goyet individuales
#~ # =============================================================================
    #~ plt.figure(figsize=(9,3))
    #~ plt.title('mag offsets over mag simulated')
    #~ plt.subplot(141)
    #~ dmag = subset_zps_hi.goyet
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('delta mag zps')

    #~ plt.subplot(142)
    #~ dmag = subset_ois_hi.goyet
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('delta mag ois')

    #~ plt.subplot(143)
    #~ dmag = subset_hot_hi.goyet
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('delta mag hot')

    #~ plt.subplot(144)
    #~ dmag = subset_sps_hi.goyet
    #~ dmag = dmag.dropna()
    #~ plt.hist(dmag, log=True)
    #~ plt.xlabel('delta mag sps')

    #~ plt.tight_layout()
    #~ plt.savefig(os.path.join(plot_dir, 'delta_over_mags_hi_goyet.svg'), dpi=400)
    #~ plt.clf()

# =============================================================================
# Como quedan los diagramas de error de magnitud vs magnitud simulada
# =============================================================================
    #plt.figure(figsize=(8,4))
    #bins = np.arange(6.5, 26.5, .5)
    #ff = subset_hot_hi.FLAGS<=4
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res_robust(subset_hot_hi[ff], bins=bins)
    #plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='g--', label='Hotpants')

    #ff = subset_sps_hi.FLAGS<=1
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res_robust(subset_sps_hi, bins=bins)
    #plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='m:', label='Scorr')

    #ff = subset_zps_hi.FLAGS<=4
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res_robust(subset_zps_hi[ff], bins=bins)
    #plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='b.-', label='Zackay')

    #ff = subset_ois_hi.FLAGS<=4
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res_robust(subset_ois_hi[ff], bins=bins)
    #plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='ro-', label='Bramich')

    #plt.tick_params(labelsize=16)
    #plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    #plt.xlabel('Sim Mag', fontsize=16)
    #plt.title('Simulated Data', fontsize=14)
    #plt.legend(loc='best', fontsize=14)

    #plt.xlim(8, 22.5)
    #plt.ylim(-2, 3)
    #plt.tight_layout()
    #plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_hi_goyet.svg'),
    #            format='svg', dpi=480)
    #plt.close()
# =============================================================================
# Liberamos algo de memoria
# =============================================================================

    #del(subset_hot_hi)
    #del(subset_sps_hi)
    #del(subset_zps_hi)
    #del(subset_ois_hi)

    #gc.collect()

# =============================================================================
# Mean goyet bajo
# =============================================================================
# =============================================================================
# vetamos por mean goyet
# =============================================================================
    subset_zps_lo = subset_zps[subset_zps.mean_goyet<0.05]
    subset_hot_lo = subset_hot[subset_hot.mean_goyet<0.05]
    subset_sps_lo = subset_sps[subset_sps.mean_goyet<0.05]
    subset_ois_lo = subset_ois[subset_ois.mean_goyet<0.05]

# =============================================================================
# Como quedan las distros de goyet individuales
# =============================================================================
    #plt.figure(figsize=(9,3))
    #plt.title('mag offsets over mag simulated')
    #plt.subplot(141)
    #dmag = subset_zps_lo.goyet
    #dmag = dmag.dropna()
    #plt.hist(dmag, log=True)
    #plt.xlabel('delta mag zps')

    #plt.subplot(142)
    #dmag = subset_ois_lo.goyet
    #dmag = dmag.dropna()
    #plt.hist(dmag, log=True)
    #plt.xlabel('delta mag ois')

    #plt.subplot(143)
    #dmag = subset_hot_lo.goyet
    #dmag = dmag.dropna()
    #plt.hist(dmag, log=True)
    #plt.xlabel('delta mag hot')

    #plt.subplot(144)
    #dmag = subset_sps_lo.goyet
    #dmag = dmag.dropna()
    #plt.hist(dmag, log=True)
    #plt.xlabel('delta mag sps')

    #plt.tight_layout()
    #plt.savefig(os.path.join(plot_dir, 'delta_over_mags_lo_goyet.svg'), dpi=400)
    #plt.clf()
    #plt.close()
# =============================================================================
# Como quedan los diagramas de error de magnitud vs magnitud simulada
# =============================================================================
    #dm = 0.0
    #plt.figure(figsize=(8,4))
    #bins = np.arange(6.5, 26.5, .5)
    #ff = subset_hot_lo.FLAGS<=0
    #ff = ff & (subset_hot_lo.mag > subset_hot_lo.p05+dm) & (subset_hot_lo.mag < subset_hot_lo.p95)
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_hot_lo.loc[ff], bins=bins)
    #plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='g--', label='Hotpants')

    # #ff = subset_sps_lo.FLAGS<=0
    #ff = ff & (subset_sps_lo.mag > subset_sps_lo.p05+dm) & (subset_sps_lo.mag < subset_sps_lo.p95)
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_sps_lo.loc[ff], bins=bins)
    #plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='m:', label='Scorr')

    #ff = subset_zps_lo.FLAGS<=0
    #ff = ff & (subset_zps_lo.mag > subset_zps_lo.p05+dm) & (subset_zps_lo.mag < subset_zps_lo.p95)
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_zps_lo.loc[ff], bins=bins)
    #plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='b.-', label='Zackay')

    #ff = subset_ois_lo.FLAGS<=0
    #ff = ff & (subset_ois_lo.mag > subset_ois_lo.p05+dm) & (subset_ois_lo.mag < subset_ois_lo.p95)
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_ois_lo.loc[ff], bins=bins)
    #plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='ro-', label='Bramich')

    #plt.tick_params(labelsize=16)
    #plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    #plt.xlabel('Sim Mag', fontsize=16)
    #plt.title('Simulated Data', fontsize=14)
    #plt.legend(loc='best', fontsize=14)

    #plt.xlim(8, 22.5)
    #plt.ylim(-2, 3)
    #plt.tight_layout()
    #plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_lo_goyet.svg'),
    #            format='svg', dpi=480)
    #plt.close()
# =============================================================================
# Como quedan los diagramas de error de magnitud vs magnitud simulada
# =============================================================================
    dm = 0.0
    means = []
    plt.figure(figsize=(8,4))
    bins = np.arange(6.5, 26.5, .5)

    ff = subset_hot_lo.FLAGS<=0
    ff = ff & (subset_hot_lo.mag > subset_hot_lo.p05+dm) & (subset_hot_lo.mag < subset_hot_lo.p95)
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_hot_lo[ff], bins=bins)
    means.append(cf.binning_res(subset_hot_lo.loc[ff], bins=bins))

    ff = ff & (subset_sps_lo.mag > subset_sps_lo.p05+dm) & (subset_sps_lo.mag < subset_sps_lo.p95)
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_sps_lo[ff], bins=bins)
    means.append(cf.binning_res(subset_sps_lo.loc[ff], bins=bins))

    ff = subset_zps_lo.FLAGS<=0
    ff = ff & (subset_zps_lo.mag > subset_zps_lo.p05+dm) & (subset_zps_lo.mag < subset_zps_lo.p95)
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_zps_lo[ff], bins=bins)
    means.append(cf.binning_res(subset_zps_lo.loc[ff], bins=bins))

    ff = subset_ois_lo.FLAGS<=0
    ff = ff & (subset_ois_lo.mag > subset_ois_lo.p05+dm) & (subset_ois_lo.mag < subset_ois_lo.p95)
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_ois_lo[ff], bins=bins)
    means.append(cf.binning_res(subset_ois_lo.loc[ff], bins=bins))
    mm = np.ma.masked_invalid(means)

    bin_centers = mm.max(axis=0)[3]
    mean = np.sum(mm[:, 0, :]*mm[:, 2, :]**2, axis=0)/np.sum(mm[:, 2, :]**2, axis=0)
    stds = np.sqrt(np.sum((mm[:, 1, :]**2)/(mm[:, 2, :]**2), axis=0))

    plt.plot(bin_centers, mean, '--', label='mean')
    plt.fill_between(bin_centers, mean-stds, mean+stds, alpha=0.5)
    #plt.errorbar(bin_centers, mean, yerr=stds, fmt='--', label='mean')

    plt.tick_params(labelsize=16)
    plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    plt.xlabel('Sim Mag', fontsize=16)
    plt.title('Simulated Data', fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.grid()
    plt.xlim(9, 22.5)
    plt.ylim(-2, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_averaged_filled.svg'),
                format='svg', dpi=480)
    plt.close()
# =============================================================================
# Como quedan los diagramas de error de magnitud vs magnitud simulada
# =============================================================================
    dm = 0.0
    means = []
    plt.figure(figsize=(8,4))
    bins = np.arange(6.5, 26.5, .5)

    ff = subset_hot_lo.FLAGS<=0
    ff = ff & (subset_hot_lo.mag > subset_hot_lo.p05+dm) & (subset_hot_lo.mag < subset_hot_lo.p95)
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_hot_lo[ff], bins=bins)
    means.append(cf.binning_res(subset_hot_lo[ff], bins=bins))

    ff = ff & (subset_sps_lo.mag > subset_sps_lo.p05+dm) & (subset_sps_lo.mag < subset_sps_lo.p95)
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_sps_lo[ff], bins=bins)
    means.append(cf.binning_res(subset_sps_lo[ff], bins=bins))

    ff = subset_zps_lo.FLAGS<=0
    ff = ff & (subset_zps_lo.mag > subset_zps_lo.p05+dm) & (subset_zps_lo.mag < subset_zps_lo.p95)
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_zps_lo[ff], bins=bins)
    means.append(cf.binning_res(subset_zps_lo[ff], bins=bins))

    ff = subset_ois_lo.FLAGS<=0
    ff = ff & (subset_ois_lo.mag > subset_ois_lo.p05+dm) & (subset_ois_lo.mag < subset_ois_lo.p95)
    #mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_ois_lo[ff], bins=bins)
    means.append(cf.binning_res(subset_ois_lo[ff], bins=bins))
    mm = np.ma.masked_invalid(means)

    bin_centers = mm.max(axis=0)[3]
    mean = np.sum(mm[:, 0, :]*mm[:, 2, :]**2, axis=0)/np.sum(mm[:, 2, :]**2, axis=0)
    stds = np.sqrt(np.sum((mm[:, 1, :]**2)/(mm[:, 2, :]**2), axis=0))

    #plt.fill_between(bin_centers, mean-stds, mean+stds)
    plt.errorbar(bin_centers, mean, yerr=stds, fmt='--', label='mean')

    plt.tick_params(labelsize=16)
    plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    plt.xlabel('Sim Mag', fontsize=16)
    plt.title('Simulated Data', fontsize=14)
    plt.legend(loc='best', fontsize=14)

    plt.xlim(9, 22.5)
    plt.ylim(-2, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_averaged.svg'),
                format='svg', dpi=480)
    plt.close()
# =============================================================================
#  Queremos los image id con buen goyet y ver quienes son
# =============================================================================
    ids_mix = store['ids_mix']
    cond = ids_mix['image_id']==ids_mix['simage_id']
    cond = cond & (ids_mix['image_id']==ids_mix['image_id_hot'])
    cond = cond & (ids_mix['image_id']==ids_mix['image_id_ois'])
    ids_mix = ids_mix.loc[cond]

    pars = ['image_id', 'mean_goyet', 'mean_goyet_iso', 'id_simulation']

    sel_zps = pd.merge(left=subset_zps[pars].drop_duplicates(),
                       right=ids_mix[['simulation_id', 'image_id']],
                       left_on='image_id', right_on='image_id',
                       #left_on='id_simulation', right_on='simulation_id',
                       suffixes=('_id_mix', '_zps'))

    sel_ois = pd.merge(left=subset_ois[pars].drop_duplicates(),
                       right=ids_mix[['simulation_id', 'image_id_ois']],
                       left_on='image_id', right_on='image_id_ois',
                       suffixes=('_id_mix', '_ois'))

    sel_sps = pd.merge(left=subset_sps[pars].drop_duplicates(),
                       right=ids_mix[['simulation_id', 'simage_id']],
                       left_on='image_id', right_on='simage_id',
                       suffixes=('_id_mix', '_sps'))

    sel_hot = pd.merge(left=subset_hot[pars].drop_duplicates(),
                       right=ids_mix[['simulation_id', 'image_id_hot']],
                       left_on='image_id', right_on='image_id_hot',
                       suffixes=('_id_mix', '_hot'))

    merged = pd.merge(left=sel_zps, right=sel_sps,
                      left_on='simulation_id', right_on='simulation_id',
                      how='inner', suffixes=('_zps', '_sps'))

    merged = pd.merge(left=merged, right=sel_ois,
                      left_on='simulation_id', right_on='simulation_id',
                      how='inner', suffixes=('', '_ois'))

    merged = pd.merge(left=merged, right=sel_hot,
                      left_on='simulation_id', right_on='simulation_id',
                      how='inner', suffixes=('', '_hot2'))

    cond = merged['image_id_zps']==merged['image_id_sps']
    cond = cond & (merged['image_id_zps']==merged['image_id_hot'])
    cond = cond & (merged['image_id_zps']==merged['image_id_ois'])

    merged = merged[cond]

    del(sel_zps)
    del(sel_sps)
    del(sel_ois)
    del(sel_hot)
    gc.collect()
# =============================================================================
# Simplemente usamos los thresholds definidos antes
# =============================================================================

    merged['has_goyet_zps'] = merged['mean_goyet_zps'] < 0.05
    merged['has_goyet_sps'] = merged['mean_goyet_sps'] < 0.05
    merged['has_goyet_ois'] = merged['mean_goyet'] < 0.05
    merged['has_goyet_hot'] = merged['mean_goyet_hot2'] < 0.05

    merged['mix_goyet'] = merged.has_goyet_zps.astype(int) + \
                          merged.has_goyet_sps.astype(int) + \
                          merged.has_goyet_ois.astype(int) + \
                          merged.has_goyet_hot.astype(int)
# =============================================================================
#  Cortamos por la suma... tiene que valer 3 o mas
# =============================================================================

    merged = merged.loc[merged[['simulation_id', 'mix_goyet']].drop_duplicates().index]
    merged['selected'] = merged.mix_goyet>=3
    merged = merged.loc[merged[['simulation_id', 'selected']].drop_duplicates().index]

# =============================================================================
# Distribucion de esta sumatoria
# =============================================================================
    plt.figure(figsize=(4,3))
    plt.title('images passing goyet threshold')
    plt.hist(merged[['simulation_id', 'mix_goyet']].drop_duplicates().mix_goyet)
    plt.xlabel('combined goyet')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mix_goyets.svg'), dpi=400)

# =============================================================================
#  Cantidades extra, para storage
# =============================================================================
    merged = merged[['simulation_id', 'image_id_zps', 'image_id_sps',
                     'image_id_ois', 'image_id_hot', 'has_goyet_sps',
                     'has_goyet_zps', 'has_goyet_ois', 'has_goyet_hot',
                     'mix_goyet', 'selected', 'mean_goyet_zps',
                     'mean_goyet_sps', 'mean_goyet', 'mean_goyet_hot2']]
    merged['mean_goyet_hot'] = merged['mean_goyet_hot2']
    merged['mean_goyet_ois'] = merged['mean_goyet']
    merged.drop(columns=['mean_goyet_hot2', 'mean_goyet'],inplace=True)

    if store_flush:
        try:
            store.remove('merged')
        except:
            pass
        store['merged'] = merged
        store.flush(fsync=True)

# =============================================================================
# Ahora vamos a usar los seleccionados para las funciones de luminosidad
# =============================================================================

    ## Primero necesitamos las inyecciones y los perdidos, seleccionados por
    ## mean_goyet
    selected = merged[merged.selected==True]

    ids = selected['simulation_id'].drop_duplicates().values
    simus = simus.loc[simus['simulation_id'].isin(ids)].drop_duplicates()

    ids = selected['image_id_zps'].drop_duplicates().values
    subset_zps = subset_zps.loc[subset_zps['image_id'].isin(ids)].drop_duplicates()

    ids = selected['image_id_sps'].drop_duplicates().values
    subset_sps = subset_sps.loc[subset_sps['image_id'].isin(ids)].drop_duplicates()

    ids = selected['image_id_ois'].drop_duplicates().values
    subset_ois = subset_ois.loc[subset_ois['image_id'].isin(ids)].drop_duplicates()

    ids = selected['image_id_hot'].drop_duplicates().values
    subset_hot = subset_hot.loc[subset_hot['image_id'].isin(ids)].drop_duplicates()


    ids = selected['image_id_zps'].drop_duplicates().values
    und_z = store['und_z']
    und_z = und_z.loc[und_z['image_id'].isin(ids)].drop_duplicates()

    ids = selected['image_id_sps'].drop_duplicates().values
    und_s = store['und_s']
    und_s = und_s.loc[und_s['image_id'].isin(ids)].drop_duplicates()

    ids = selected['image_id_hot'].drop_duplicates().values
    und_h = store['und_h']
    und_h = und_h.loc[und_h['image_id'].isin(ids)].drop_duplicates()

    ids = selected['image_id_ois'].drop_duplicates().values
    und_o = store['und_b']
    und_o = und_o.loc[und_o['image_id'].isin(ids)].drop_duplicates()

# =============================================================================
# Check that some objects are both in detected and in undetected
# =============================================================================
    print('length of the undetected before')
    print(len(und_z), len(und_s), len(und_h), len(und_o))

    und_z = und_z.loc[~und_z['simulated_id'].isin(subset_zps.sim_id.dropna().drop_duplicates())].drop_duplicates()
    und_s = und_s.loc[~und_s['simulated_id'].isin(subset_sps.sim_id.dropna().drop_duplicates())].drop_duplicates()
    und_h = und_h.loc[~und_h['simulated_id'].isin(subset_hot.sim_id.dropna().drop_duplicates())].drop_duplicates()
    und_o = und_o.loc[~und_o['simulated_id'].isin(subset_ois.sim_id.dropna().drop_duplicates())].drop_duplicates()

    print('length of the undetected after duplicates drops')
    print(len(und_z), len(und_s), len(und_h), len(und_o))

# =============================================================================
# Check the other way of this
# ===================================================+==========================

    print('length of the sources before')
    print(len(subset_zps), len(subset_sps), len(subset_hot), len(subset_ois))
    print(sum(subset_zps.IS_REAL), sum(subset_sps.IS_REAL),
          sum(subset_hot.IS_REAL), sum(subset_ois.IS_REAL))

    subset_zps = subset_zps.loc[~subset_zps['sim_id'].isin(und_z.simulated_id.dropna().drop_duplicates())].drop_duplicates()
    subset_sps = subset_sps.loc[~subset_sps['sim_id'].isin(und_s.simulated_id.dropna().drop_duplicates())].drop_duplicates()
    subset_hot = subset_hot.loc[~subset_hot['sim_id'].isin(und_h.simulated_id.dropna().drop_duplicates())].drop_duplicates()
    subset_ois = subset_ois.loc[~subset_ois['sim_id'].isin(und_o.simulated_id.dropna().drop_duplicates())].drop_duplicates()

    print('length of the sources after duplicates drops')
    print(len(subset_zps), len(subset_sps), len(subset_hot), len(subset_ois))
    print(sum(subset_zps.IS_REAL), sum(subset_sps.IS_REAL),
          sum(subset_hot.IS_REAL), sum(subset_ois.IS_REAL))

# =============================================================================
# Check that we have no simulations with image_ids different than in dt_'s
# =============================================================================

    zz = simus.loc[simus['image_id'].isin(subset_zps.image_id.dropna().drop_duplicates())]
    ss = simus.loc[simus['simage_id'].isin(subset_sps.image_id.dropna().drop_duplicates())]
    hh = simus.loc[simus['image_id_hot'].isin(subset_hot.image_id.dropna().drop_duplicates())]
    oo = simus.loc[simus['image_id_ois'].isin(subset_ois.image_id.dropna().drop_duplicates())]

    jump = np.array([np.sum(np.sum(ss!=hh)),
                     np.sum(np.sum(zz!=hh)),
                     np.sum(np.sum(oo!=hh)),
                     np.sum(np.sum(ss!=zz)),
                     np.sum(np.sum(ss!=oo)),
                     np.sum(np.sum(zz!=oo))])
    if np.any(jump):
        import ipdb; ipdb.set_trace()
    else:
        simus = zz.drop_duplicates()
        del(zz)
        del(ss)
        del(hh)
        del(oo)

# =============================================================================
# There is a discrepancy of ~1900 sources, not present at simus... Weird
# =============================================================================
    discreps = [np.sum(subset_zps.IS_REAL)+len(und_z) - len(simus),
                np.sum(subset_sps.IS_REAL)+len(und_s) - len(simus),
                np.sum(subset_hot.IS_REAL)+len(und_h) - len(simus),
                np.sum(subset_ois.IS_REAL)+len(und_o) - len(simus)]
    print(discreps)
    ids = np.unique(np.hstack([subset_zps.sim_id.dropna().drop_duplicates().values,
                               und_z.simulated_id.dropna().drop_duplicates()]))
    print('Are the ids of detected + und unique?')
    print(len(ids) == sum(subset_zps.IS_REAL) + len(und_z))

    newsimus = simulated.loc[simulated['id'].isin(ids)]
    lost_z = newsimus.loc[~newsimus['id'].isin(simus.id_y)]

    ids = np.unique(np.hstack([subset_sps.sim_id.dropna().drop_duplicates().values,
                               und_s.simulated_id.dropna().drop_duplicates()]))
    print('Are the ids of detected + und unique?')
    print(len(ids) == sum(subset_sps.IS_REAL) + len(und_s))

    newsimus = simulated.loc[simulated['id'].isin(ids)]
    lost_s = newsimus.loc[~newsimus['id'].isin(simus.id_y)]

    ids = np.unique(np.hstack([subset_ois.sim_id.dropna().drop_duplicates().values,
                               und_o.simulated_id.dropna().drop_duplicates()]))
    print('Are the ids of detected + und unique?')
    print(len(ids) == sum(subset_ois.IS_REAL) + len(und_o))

    newsimus = simulated.loc[simulated['id'].isin(ids)]
    lost_o = newsimus.loc[~newsimus['id'].isin(simus.id_y)]

    ids = np.unique(np.hstack([subset_hot.sim_id.dropna().drop_duplicates().values,
                               und_h.simulated_id.dropna().drop_duplicates()]))
    print('Are the ids of detected + und unique?')
    print(len(ids), sum(subset_hot.IS_REAL), len(und_h))
    print(len(ids) == sum(subset_hot.IS_REAL) + len(und_h))

    newsimus = simulated.loc[simulated['id'].isin(ids)]
    lost_h = newsimus.loc[~newsimus['id'].isin(simus.id_y)]

    # there seems to be a lack of simulation id recorded in this hot table. 
    # How to imput them? We first should check on the simulation ids...
    # The problem is the bogus objects... How to place them in the whole situation?
    # The answer could be by image_id
    
    subhot = pd.merge(left=subset_hot, right=ids_mix[['simulation_id', 'image_id_hot']],
                      left_on='image_id', right_on='image_id_hot', how='left')
    subhot['id_simulation'] = subhot['simulation_id']
    subset_hot = subhot
    del(subhot)
    #import ipdb; ipdb.set_trace()
    #und_h = pd.merge(left=und_h, right=ids_mix[['simulation_id', 'image_id_hot']],
                   # left_on='image_id', right_on='image_id_hot', how='left')
    #und_h['id_simulation'] = und_h['simulation_id']
    
    if np.any(np.array([np.sum(np.sum(lost_h.x!=lost_o.x)),
                        np.sum(np.sum(lost_h.x!=lost_s.x)),
                        np.sum(np.sum(lost_o.x!=lost_s.x)),
                        np.sum(np.sum(lost_z.x!=lost_s.x)),
                        np.sum(np.sum(lost_z.x!=lost_o.x)),
                        np.sum(np.sum(lost_z.x!=lost_h.x))])):
        print('differences in lost')
    else:
        print('we lost the same simulated objects')
        lost = lost_z
        #del(lost_z)
        del(lost_h)
        del(lost_s)
        del(lost_o)

# =============================================================================
# Need to check the simulations on this lost objects
# =============================================================================
    lost = pd.merge(left=simulations, right=lost,
                     right_on='simulation_id', left_on='id', how='inner')
    print(lost.simulation_id.describe())

# =============================================================================
# Clean everything from nas in simulations id
# =============================================================================
    #simus = simus.loc[~simus.id_y.isna()]
    # what if???
    simus = newsimus

    subset_zps = subset_zps.loc[~subset_zps.id_simulation.isna()]
    subset_sps = subset_sps.loc[~subset_sps.id_simulation.isna()]
    subset_ois = subset_ois.loc[~subset_ois.id_simulation.isna()]
    subset_hot = subset_hot.loc[~subset_hot.id_simulation.isna()]

    und_z = und_z.loc[~und_z.simulated_id.isna()]
    und_s = und_s.loc[~und_s.simulated_id.isna()]
    und_o = und_o.loc[~und_o.simulated_id.isna()]
    und_h = und_h.loc[~und_h.simulated_id.isna()]

    if store_flush:
        store['c_subset_zps'] = subset_zps
        store['c_subset_sps'] = subset_sps
        store['c_subset_ois'] = subset_ois
        store['c_subset_hot'] = subset_hot
        store.flush(fsync=True)

        store['c_simus'] = simus
        store.flush(fsync=True)

        store['c_und_z'] = und_z
        store['c_und_s'] = und_s
        store['c_und_h'] = und_h
        store['c_und_o'] = und_o
        store.flush(fsync=True)

# =============================================================================
# imputar magnitudes de hot
# =============================================================================
    import ipdb; ipdb.set_trace()
    nomags = subset_hot.loc[subset_hot['mag'].isna()]
    nomags['mag_offset'] = nomags['sim_mag'] - nomags['MAG_APER']
    cals = cf.cal_mags(nomags)
    subset_hot = pd.merge(left=subset_hot, right=cals, on='image_id', how='left', suffixes=('', '_cl'))
    fails = subset_hot['mag'].isna()
    subset_hot.loc[fails,'mag'] = subset_hot.loc[fails,'slope']*subset_hot.loc[fails,'MAG_APER']+subset_hot.loc[fails,'mean_offset']
    fails = subset_hot['mag'].isna()
    subset_hot.loc[fails,'mag'] = sigma_clipped_stats(subset_hot.slope.values)[1]*subset_hot.loc[fails,'MAG_APER']+sigma_clipped_stats(subset_hot.mean_offset.values)[1]
    
    #subset_hot.loc[subset_hot['mag'].isna()]['mag'] = 
    #subset_hot['mag'] = subset_hot['mag_h']
# =============================================================================
# plot de funcion de luminosidad inyectada
# =============================================================================
    plt.figure(figsize=(6,3))
    plt.hist(simus['app_mag'], cumulative=False, bins=25, log=True)
    plt.xlabel(r'$mag$', fontsize=16)
    plt.tick_params(labelsize=15)
    plt.ylabel(r'$N(m) dm$', fontsize=16)
    #plt.ylabel(r'$\int_{-\infty}^{mag}\phi(m\prime)dm\prime$', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'lum_fun_simulated.svg'), dpi=400)

# =============================================================================
# Funciones de luminosidad combinadas
# =============================================================================
    plt.figure(figsize=(12,4))
    plt.title('Luminosity function', fontsize=14)
    cumulative=True
    #magnitude bins
    bins = np.arange(7, 26.5, 0.5)
    plt.rcParams['text.usetex'] = False

    plt.subplot(131)
    x_bins, vals = cf.custom_histogram(simus.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'black', label='Injected')

    x_bins, vals = cf.custom_histogram(subset_ois[subset_ois.IS_REAL==True].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'ro-', label='Bramich')

    x_bins, vals = cf.custom_histogram(subset_zps[subset_zps.IS_REAL==True].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'b.-', label='Zackay')

    x_bins, vals = cf.custom_histogram(subset_sps[subset_sps.IS_REAL==True].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'm:', label='$S_{corr}$')

    x_bins, vals = cf.custom_histogram(subset_hot[subset_hot.IS_REAL==True].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'g--', label='A-Lupton')

    if cumulative:
        plt.ylabel(r'$N(>r)$', fontsize=16)
    else:
        plt.ylabel(r'$N(m)dm$', fontsize=16)
    plt.ylim(1, 1e8)
    plt.xlim(7., 25.5)
    plt.title('Real', fontsize=16)
    #plt.ylabel(r'$N(m)dm$', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.xlabel(r'$r \ [mag]$', fontsize=16)
    #plt.ylim(50, 280000)
    plt.tick_params(labelsize=16)
    plt.grid()
    plt.subplot(132)
    x_bins, vals = cf.custom_histogram(simus.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'black', label='Injected')
    x_bins, vals = cf.custom_histogram(subset_ois[subset_ois.IS_REAL==False].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'ro-', label='Bramich')
    x_bins, vals = cf.custom_histogram(subset_zps[subset_zps.IS_REAL==False].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'b.-', label='Zackay')
    x_bins, vals = cf.custom_histogram(subset_sps[subset_sps.IS_REAL==False].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'm:', label='$S_{corr}$')
    x_bins, vals = cf.custom_histogram(subset_hot[subset_hot.IS_REAL==False].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'g--', label='A-Lupton')
    plt.xlim(7., 25.5)
    plt.ylim(1, 1e8)
    #plt.ylabel(r'$N(m)dm$', fontsize=16)
    #plt.legend(loc='best', fontsize=16)
    plt.xlabel(r'$r \ [mag]$', fontsize=16)
    plt.title('Bogus', fontsize=16)
    #plt.ylim(50, 280000)
    plt.tick_params(labelsize=16)
    plt.grid()
    plt.subplot(133)
    plt.title('False Negatives', fontsize=16)
    x_bins, vals = cf.custom_histogram(simus.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'black', label='Injected')
    x_bins, vals = cf.custom_histogram(und_o.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'ro-', label='Bramich')
    x_bins, vals = cf.custom_histogram(und_z.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'b.-', label='Zackay')
    x_bins, vals = cf.custom_histogram(und_s.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'm:', label='$S_{corr}$')
    x_bins, vals = cf.custom_histogram(und_h.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'g--', label='A-Lupton')
    #plt.legend(loc='lower right', fontsize=16)
    plt.xlabel(r'$r \ [mag]$', fontsize=16)
    #plt.title('Cummulative Luminosity Function of False Negatives', fontsize=14)
    plt.tick_params(labelsize=16)
    plt.xlim(7., 25.5)
    #plt.show()
    plt.ylim(1, 1e8)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'combined_luminosities_functions_simu.pdf'),
                format='pdf', dpi=720)
    plt.close()

# =============================================================================
# Funciones de luminosidad combinadas
# =============================================================================
    plt.figure(figsize=(12,4))
    plt.title('Luminosity function', fontsize=14)
    cumulative=True
    #magnitude bins
    bins = np.arange(7, 26.5, 0.5)
    plt.rcParams['text.usetex'] = False

    plt.subplot(131)
    x_bins, vals = cf.custom_histogram(simus.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'black', label='Injected')

    x_bins, vals = cf.custom_histogram(subset_ois[subset_ois.IS_REAL==True].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'ro-', label='Bramich')

    x_bins, vals = cf.custom_histogram(subset_zps[subset_zps.IS_REAL==True].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'b.-', label='Zackay')

    x_bins, vals = cf.custom_histogram(subset_sps[subset_sps.IS_REAL==True].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'm:', label='$S_{corr}$')

    x_bins, vals = cf.custom_histogram(subset_hot[subset_hot.IS_REAL==True].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'g--', label='A-Lupton')

    if cumulative:
        plt.ylabel(r'$N(>r)$', fontsize=16)
    else:
        plt.ylabel(r'$N(m)dm$', fontsize=16)
    plt.ylim(1, 1e8)
    plt.xlim(7., 25.5)
    plt.title('Real', fontsize=16)
    #plt.ylabel(r'$N(m)dm$', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.xlabel(r'$r \ [mag]$', fontsize=16)
    #plt.ylim(50, 280000)
    plt.tick_params(labelsize=16)

    plt.subplot(132)
    x_bins, vals = cf.custom_histogram(simus.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'black', label='Injected')
    x_bins, vals = cf.custom_histogram(subset_ois[subset_ois.IS_REAL==False].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'ro-', label='Bramich')
    x_bins, vals = cf.custom_histogram(subset_zps[subset_zps.IS_REAL==False].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'b.-', label='Zackay')
    x_bins, vals = cf.custom_histogram(subset_sps[subset_sps.IS_REAL==False].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'm:', label='$S_{corr}$')
    x_bins, vals = cf.custom_histogram(subset_hot[subset_hot.IS_REAL==False].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'g--', label='A-Lupton')
    plt.xlim(7., 25.5)
    plt.ylim(1, 1e8)
    #plt.ylabel(r'$N(m)dm$', fontsize=16)
    #plt.legend(loc='best', fontsize=16)
    plt.xlabel(r'$r \ [mag]$', fontsize=16)
    plt.title('Bogus', fontsize=16)
    #plt.ylim(50, 280000)
    plt.tick_params(labelsize=16)

    plt.subplot(133)
    plt.title('False Negatives', fontsize=16)
    x_bins, vals = cf.custom_histogram(simus.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'black', label='Injected')
    x_bins, vals = cf.custom_histogram(und_o.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'ro-', label='Bramich')
    x_bins, vals = cf.custom_histogram(und_z.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'b.-', label='Zackay')
    x_bins, vals = cf.custom_histogram(und_s.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'm:', label='$S_{corr}$')
    x_bins, vals = cf.custom_histogram(und_h.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'g--', label='A-Lupton')
    #plt.legend(loc='lower right', fontsize=16)
    plt.xlabel(r'$r \ [mag]$', fontsize=16)
    #plt.title('Cummulative Luminosity Function of False Negatives', fontsize=14)
    plt.tick_params(labelsize=16)
    plt.xlim(7., 25.5)
    #plt.show()
    plt.ylim(1, 1e8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'combined_luminosities_functions.svg'),
                format='svg', dpi=720)
    plt.close()
    store.close()
# =============================================================================
# Tabla de recoveries
# =============================================================================
    print(len(simus))
    ois = ['Bramich', np.sum(subset_ois.IS_REAL), len(und_o), len(subset_ois.IS_REAL)-np.sum(subset_ois.IS_REAL)]
    zps = ['Zackay', np.sum(subset_zps.IS_REAL), len(und_z), len(subset_zps.IS_REAL)-np.sum(subset_zps.IS_REAL)]
    hot = ['ALupton', np.sum(subset_hot.IS_REAL), len(und_h), len(subset_hot.IS_REAL)-np.sum(subset_hot.IS_REAL)]
    sps = ['Ssep', np.sum(subset_sps.IS_REAL), len(und_s), len(subset_sps.IS_REAL)-np.sum(subset_sps.IS_REAL)]
    #scr = [np.sum(dt_scr.IS_REAL), len(und_sc), len(dt_scr.IS_REAL)-np.sum(dt_scr.IS_REAL)]

    df2 = pd.DataFrame([zps, ois, hot, sps], #, scr],
                   columns=['Method', 'Real', 'False Neg', 'Bogus'])
    df2['TruePos'] = df2['Real']/(df2['Real']+df2['False Neg'])
    df2['FalseNeg'] = df2['False Neg']/(df2['Real']+df2['False Neg'])
    df2['FalsePos'] = df2['Bogus']/(df2['Real']+df2['False Neg'])
    #import ipdb; ipdb.set_trace()
    print(df2['Real']+df2['False Neg'])
    with open(os.path.join(plot_dir, 'table_of_dets.txt'), 'w') as f:
        f.write(df2.to_latex())
        f.write(str(len(simus)))
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--m1_diam", help="diameter to filter",
                        default=None, type=float)

    parser.add_argument("path", help="path to plot files")

    parser.add_argument("-s", "--store_flush",
                        help="flush the merged to the store h5",
                        type=bool, default=False)

    args = parser.parse_args()

    print([args.m1_diam, args.path, args.store_flush])

    import sys
    sys.exit(main(args.m1_diam, args.path, store_flush=args.store_flush))
