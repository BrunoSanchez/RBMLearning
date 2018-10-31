#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  borrador.py
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

#sns.set_context(font_scale=16)
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams['text.usetex'] = True


#~ def main(m1_diam=1.54, plots_path='./plots/.'):
m1_diam = 1.54
plots_path='./plots_1540'
plot_dir = os.path.abspath(plots_path)
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)


simulated = store['simulated']
simulations = store['simulations']

simulations = simulations[simulations.failed_to_subtract==False]
simulations = simulations[simulations.m1_diam==m1_diam]

simus = pd.merge(left=simulations, right=simulated,
                 right_on='simulation_id', left_on='id', how='outer')

# =============================================================================
# tables
# =============================================================================
dt_zps = store['dt_ois']
dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
dt_zps = cf.optimize_df(dt_zps)
dt_zps['VALID_MAG'] = dt_zps['MAG_APER']<30
dt_zps['mag_offset'] = dt_zps['sim_mag'] - dt_zps['MAG_APER']
mean_offset, median_offset, std_offset = sigma_clipped_stats(dt_zps.mag_offset)
dt_zps['mag'] = dt_zps['MAG_APER'] + mean_offset
dt_zps['goyet'] = np.abs(dt_zps['sim_mag'] - dt_zps['mag'])/dt_zps['sim_mag']
grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet'])[0])
dd.name = 'mean_goyet'
dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
dt_ois = dt_zps

dt_zps = store['dt_sps']
dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
dt_zps = cf.optimize_df(dt_zps)
dt_zps['MAG_APER'] = -2.5*np.log10(dt_zps.cflux)
dt_zps['VALID_MAG'] = dt_zps['MAG_APER']<30
dt_zps['mag_offset'] = dt_zps['sim_mag'] - dt_zps['MAG_APER']
mean_offset, median_offset, std_offset = sigma_clipped_stats(dt_zps.mag_offset)
dt_zps['mag'] = dt_zps['MAG_APER'] + mean_offset
dt_zps['goyet'] = np.abs(dt_zps['sim_mag'] - dt_zps['mag'])/dt_zps['sim_mag']
grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet'])[0])
dd.name = 'mean_goyet'
dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
dt_sps = dt_zps

dt_zps = store['dt_hot']
dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
dt_zps = cf.optimize_df(dt_zps)
dt_zps['VALID_MAG'] = dt_zps['MAG_APER']<30
dt_zps['mag_offset'] = dt_zps['sim_mag'] - dt_zps['MAG_APER']
mean_offset, median_offset, std_offset = sigma_clipped_stats(dt_zps.mag_offset)
dt_zps['mag'] = dt_zps['MAG_APER'] + mean_offset
dt_zps['goyet'] = np.abs(dt_zps['sim_mag'] - dt_zps['mag'])/dt_zps['sim_mag']
grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet'])[0])
dd.name = 'mean_goyet'
dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
dt_hot = dt_zps

dt_zps = store['dt_zps']
dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
dt_zps = cf.optimize_df(dt_zps)
dt_zps['VALID_MAG'] = dt_zps['MAG_APER']<30
dt_zps['mag_offset'] = dt_zps['sim_mag'] - dt_zps['MAG_APER']
mean_offset, median_offset, std_offset = sigma_clipped_stats(dt_zps.mag_offset)
dt_zps['mag'] = dt_zps['MAG_APER'] + mean_offset
dt_zps['goyet'] = np.abs(dt_zps['sim_mag'] - dt_zps['mag'])/dt_zps['sim_mag']
grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet'])[0])
dd.name = 'mean_goyet'
dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')


# =============================================================================
# plot de funcion de luminosidad inyectada
# =============================================================================
plt.figure(figsize=(6,3))
plt.hist(simus['app_mag'], cumulative=False, bins=25, log=True)
plt.xlabel(r'$mag$', fontsize=16)
plt.tick_params(labelsize=15)
plt.ylabel(r'$N(m) dm$', fontsize=16)
#plt.ylabel(r'$\int_{-\infty}^{mag}\phi(m\prime)dm\prime$', fontsize=16)
plt.savefig(os.path.join(plot_dir, 'lum_fun_simulated.svg'), dpi=400)
plt.clf()

# =============================================================================
# plot de deltas de magnitud
# =============================================================================
plt.figure(figsize=(9,3))
plt.title('mag offsets over mag simulated')
plt.subplot(141)
dmag = dt_zps[dt_zps.VALID_MAG==True].mag_offset
dmag = dmag.dropna()
plt.hist(dmag, log=True)
plt.xlabel('delta mag zps')

plt.subplot(142)
dmag = dt_ois[dt_ois.VALID_MAG==True].mag_offset
dmag = dmag.dropna()
plt.hist(dmag, log=True)
plt.xlabel('delta mag ois')

plt.subplot(143)
dmag = dt_hot[dt_hot.VALID_MAG==True].mag_offset
dmag = dmag.dropna()
plt.hist(dmag, log=True)
plt.xlabel('delta mag hot')

plt.subplot(144)
dmag = dt_sps[dt_sps.VALID_MAG==True].mag_offset
dmag = dmag.dropna()
plt.hist(dmag, log=True)
plt.xlabel('delta mag sps')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'delta_mags.svg'), dpi=400)
plt.clf()

# =============================================================================
# plot de deltas de magnitud sobre magnitud (goyet)
# =============================================================================
plt.figure(figsize=(9,3))
plt.title('mag offsets over mag simulated')
plt.subplot(141)
dmag = dt_zps[(dt_zps.VALID_MAG==True)].goyet
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag zps')

plt.subplot(142)
dmag = dt_ois[(dt_ois.VALID_MAG==True)].goyet
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag ois')

plt.subplot(143)
dmag = dt_hot[(dt_hot.VALID_MAG==True)].goyet
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag hot')

plt.subplot(144)
dmag = dt_sps[(dt_sps.VALID_MAG==True)].goyet
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag sps')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'delta_over_mags.svg'), dpi=400)
plt.clf()

# =============================================================================
# Seleccionamos los mean_goyet
# =============================================================================

subset_zps = dt_zps[['mean_goyet', 'image_id', 'id_simulation']]
subset_zps.drop_duplicates(inplace=True)

subset_ois = dt_ois[['mean_goyet', 'image_id', 'id_simulation']]
subset_ois.drop_duplicates(inplace=True)

subset_sps = dt_sps[['mean_goyet', 'image_id', 'id_simulation']]
subset_sps.drop_duplicates(inplace=True)

subset_hot = dt_hot[['mean_goyet', 'image_id', 'id_simulation']]
subset_hot.drop_duplicates(inplace=True)

# =============================================================================
# Distribuciones de goyet
# =============================================================================
plt.figure(figsize=(9,3))
plt.title('mean goyet')
plt.subplot(141)
dmag = subset_zps.mean_goyet
dmag = dmag.dropna()
plt.hist(dmag, log=True)
plt.xlabel('mean goyet zps')

plt.subplot(142)
dmag = subset_ois.mean_goyet
dmag = dmag.dropna()
plt.hist(dmag, log=True)
plt.xlabel('mean goyet ois')

plt.subplot(143)
dmag = subset_hot.mean_goyet
dmag = dmag.dropna()
plt.hist(dmag, log=True)
plt.xlabel('mean goyet hot')

plt.subplot(144)
dmag = subset_sps.mean_goyet
dmag = dmag.dropna()
plt.hist(dmag, log=True)
plt.xlabel('mean goyet sps')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'mean_goyet.svg'), dpi=400)
plt.clf()

# =============================================================================
# vetamos por mean goyet
# =============================================================================
subset_zps = subset_zps[subset_zps.mean_goyet<=0.25]
subset_hot = subset_hot[subset_hot.mean_goyet<=0.25]
subset_sps = subset_sps[subset_sps.mean_goyet<=0.25]
subset_ois = subset_ois[subset_ois.mean_goyet<=0.25]

merged = pd.merge(left=subset_zps, right=simulations,
                  left_on='id_simulation', right_on='id',
                  how='inner')

merged = pd.merge(left=merged, right=subset_ois,
                  left_on='', right_on='id_simulation',
                  how='inner')

merged = pd.merge(left=merged, right=subset_sps,
                  left_on='', right_on='id_simulation',
                  how='inner')

merged = pd.merge(left=merged, right=subset_hot,
                  left_on='', right_on='id_simulation',
                  how='inner')


# =============================================================================
# Distribuciones de goyet vs pars
# =============================================================================

mag_range = [16, 20]
mag_bins = np.arange(16, 20, 0.5)
dataset =
data = dataset[dataset.sim_mag<=20]
data = data[data.sim_mag>=16]
data = data[data.VALID_MAG==True]
cube = data[['r_scales', 'gx_mag', 'm1_diam',
'm2_diam', 'ref_starzp', 'ref_starslope',
'ref_fwhm', 'new_fwhm', 'eff_col', 'px_scale', 'ref_back_sbright',
'new_back_sbright', 'exp_time', 'mag_offset', 'goyet']]

cols = ['r_scales', 'gx_mag', 'm1_diam', 'ref_starzp', 'ref_starslope',
        'ref_fwhm', 'new_fwhm', 'eff_col', 'px_scale', 'ref_back_sbright',
        'new_back_sbright', 'exp_time', 'mag_offset']

for a_par in cols:
    subplot()



def goyet_vs_pars_plot(dataset, dia='zackay'):
    mag_range = [16, 20]
    mag_bins = np.arange(16, 20, 0.5)
    data = dataset[dataset.sim_mag<=20]
    data = data[data.sim_mag>=16]
    data = data[data.VALID_MAG==True]
    cube = data[['r_scales', 'gx_mag', 'm1_diam',
    'm2_diam', 'ref_starzp', 'ref_starslope',
    'ref_fwhm', 'new_fwhm', 'eff_col', 'px_scale', 'ref_back_sbright',
    'new_back_sbright', 'exp_time', 'mag_offset', 'goyet']]

    cols = ['r_scales', 'gx_mag', 'm1_diam', 'ref_starzp', 'ref_starslope',
            'ref_fwhm', 'new_fwhm', 'eff_col', 'px_scale', 'ref_back_sbright',
            'new_back_sbright', 'exp_time', 'mag_offset']

    for a_par in cols:
        subplot()

# =============================================================================
# plot de goyet factor
# =============================================================================

def goyet_vs_pars_plot(dataset, dia='zackay'):
    mag_range = [16, 20]
    mag_bins = np.arange(16, 20, 0.5)
    data = dataset[dataset.sim_mag<=20]
    data = data[data.sim_mag>=16]
    data = data[data.VALID_MAG==True]
    cube = data[['new_fwhm', 'ref_fwhm', 'ref_starzp','ref_starslope',
                 'px_scale', 'ref_back_sbright','new_back_sbright',
                 'exp_time', 'goyet']]

    plt.plot(data.sim_mag, data.goyet, '.')
    plt.xlabel('simulated mag')
    plt.ylabel('goyet')
    plt.savefig(os.path.join(plot_dir, 'simulated_goyet_{}.svg'.format(dia)), dpi=400)

    plt.figure(figsize=(18, 18))

    plt.subplot(7, 4, 1)
    plot_data = []
    for ref_starzp in [128e3, 256e3]:
        subcube = cube[np.abs(cube.ref_starzp-ref_starzp)<0.1]
        for ref_starslope in [0.1, 0.5, 0.9]:
            subcube2 = subcube[np.abs(subcube.ref_starslope-ref_starslope)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starzp, ref_starslope,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{zp}$')
    plt.ylabel('$ref_{slope}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 2)
    plot_data = []
    for ref_starzp in [128e3, 256e3]:
        subcube = cube[np.abs(cube.ref_starzp-ref_starzp)<0.1]
        for ref_fwhm in [0.8, 1., 1.3]:
            subcube2 = subcube[np.abs(subcube.ref_fwhm-ref_fwhm)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starzp, ref_fwhm,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{zp}$')
    plt.ylabel('$ref_{fwhm}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 3)
    plot_data = []
    for ref_starzp in [128e3, 256e3]:
        subcube = cube[np.abs(cube.ref_starzp-ref_starzp)<0.1]
        for exp_time in [120, 300]:
            subcube2 = subcube[np.abs(subcube.exp_time-exp_time)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starzp, exp_time,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{zp}$')
    plt.ylabel('$exptime$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 4)
    plot_data = []
    for ref_starzp in [128e3, 256e3]:
        subcube = cube[np.abs(cube.ref_starzp-ref_starzp)<0.1]
        for new_fwhm in [1.3, 1.9, 2.5]:
            subcube2 = subcube[np.abs(subcube.new_fwhm-new_fwhm)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starzp, new_fwhm,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{zp}$')
    plt.ylabel('$new_{fwhm}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 5)
    plot_data = []
    for ref_starzp in [128e3, 256e3]:
        subcube = cube[np.abs(cube.ref_starzp-ref_starzp)<0.1]
        for px_scale in [0.3, 0.7, 1.4]:
            subcube2 = subcube[np.abs(subcube.px_scale-px_scale)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starzp, px_scale,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{zp}$')
    plt.ylabel('px scale')
    plt.colorbar(label='goyet=$<\delta m /m>$')


    plt.subplot(7, 4, 6)
    plot_data = []
    for ref_starzp in [128e3, 256e3]:
        subcube = cube[np.abs(cube.ref_starzp-ref_starzp)<0.1]
        for ref_back_sbright in [20., 21., 22.]:
            subcube2 = subcube[np.abs(subcube.ref_back_sbright-ref_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starzp, ref_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{zp}$')
    plt.ylabel('$ref_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 7)
    plot_data = []
    for ref_starzp in [128e3, 256e3]:
        subcube = cube[np.abs(cube.ref_starzp-ref_starzp)<0.1]
        for new_back_sbright in [20, 19., 18]:
            subcube2 = subcube[np.abs(subcube.new_back_sbright-new_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starzp, new_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{zp}$')
    plt.ylabel('$new_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 8)
    plot_data = []
    for ref_starslope in [0.1, 0.5, 0.9]:
        subcube = cube[np.abs(cube.ref_starslope-ref_starslope)<0.1]
        for ref_fwhm in [0.8, 1., 1.3]:
            subcube2 = subcube[np.abs(subcube.ref_fwhm-ref_fwhm)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starslope, ref_fwhm,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{slope}$')
    plt.ylabel('$ref_{fwhm}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 9)
    plot_data = []
    for ref_starslope in [0.1, 0.5, 0.9]:
        subcube = cube[np.abs(cube.ref_starslope-ref_starslope)<0.1]
        for exp_time in [120, 300]:
            subcube2 = subcube[np.abs(subcube.exp_time-exp_time)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starslope, exp_time,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{slope}$')
    plt.ylabel('$exptime$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 10)
    plot_data = []
    for ref_starslope in [0.1, 0.5, 0.9]:
        subcube = cube[np.abs(cube.ref_starslope-ref_starslope)<0.1]
        for new_fwhm in [1.3, 1.9, 2.5]:
            subcube2 = subcube[np.abs(subcube.new_fwhm-new_fwhm)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starslope, new_fwhm,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{slope}$')
    plt.ylabel('$new_{fwhm}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 11)
    plot_data = []
    for ref_starslope in [0.1, 0.5, 0.9]:
        subcube = cube[np.abs(cube.ref_starslope-ref_starslope)<0.1]
        for px_scale in [0.3, 0.7, 1.4]:
            subcube2 = subcube[np.abs(subcube.px_scale-px_scale)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starslope, px_scale,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{slope}$')
    plt.ylabel('px scale')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 12)
    plot_data = []
    for ref_starslope in [0.1, 0.5, 0.9]:
        subcube = cube[np.abs(cube.ref_starslope-ref_starslope)<0.1]
        for ref_back_sbright in [20., 21., 22.]:
            subcube2 = subcube[np.abs(subcube.ref_back_sbright-ref_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starslope, ref_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{slope}$')
    plt.ylabel('$ref_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 13)
    plot_data = []
    for ref_starslope in [0.1, 0.5, 0.9]:
        subcube = cube[np.abs(cube.ref_starslope-ref_starslope)<0.1]
        for new_back_sbright in [20, 19., 18]:
            subcube2 = subcube[np.abs(subcube.new_back_sbright-new_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_starslope, new_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{slope}$')
    plt.ylabel('$new_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')


    plt.subplot(7, 4, 14)
    plot_data = []
    for ref_fwhm in [0.8, 1., 1.3]:
        subcube = cube[np.abs(cube.ref_fwhm-ref_fwhm)<0.1]
        for exp_time in [120, 300]:
            subcube2 = subcube[np.abs(subcube.exp_time-exp_time)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_fwhm, exp_time,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{fwhm}$')
    plt.ylabel('$exptime$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 15)
    plot_data = []
    for ref_fwhm in [0.8, 1., 1.3]:
        subcube = cube[np.abs(cube.ref_fwhm-ref_fwhm)<0.1]
        for new_fwhm in [1.3, 1.9, 2.5]:
            subcube2 = subcube[np.abs(subcube.new_fwhm-new_fwhm)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_fwhm, new_fwhm,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{fwhm}$')
    plt.ylabel('$new_{fwhm}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 16)
    plot_data = []
    for ref_fwhm in [0.8, 1., 1.3]:
        subcube = cube[np.abs(cube.ref_fwhm-ref_fwhm)<0.1]
        for px_scale in [0.3, 0.7, 1.4]:
            subcube2 = subcube[np.abs(subcube.px_scale-px_scale)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_fwhm, px_scale,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{fwhm}$')
    plt.ylabel('px scale')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 17)
    plot_data = []
    for ref_fwhm in [0.8, 1., 1.3]:
        subcube = cube[np.abs(cube.ref_fwhm-ref_fwhm)<0.1]
        for ref_back_sbright in [20., 21., 22.]:
            subcube2 = subcube[np.abs(subcube.ref_back_sbright-ref_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_fwhm, ref_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{fwhm}$')
    plt.ylabel('$ref_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 18)
    plot_data = []
    for ref_fwhm in [0.8, 1., 1.3]:
        subcube = cube[np.abs(cube.ref_fwhm-ref_fwhm)<0.1]
        for new_back_sbright in [20, 19., 18]:
            subcube2 = subcube[np.abs(subcube.new_back_sbright-new_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_fwhm, new_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{fwhm}$')
    plt.ylabel('$new_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')


    plt.subplot(7, 4, 19)
    plot_data = []
    for exp_time in [120, 300]:
        subcube = cube[np.abs(cube.exp_time-exp_time)<0.1]
        for new_fwhm in [1.3, 1.9, 2.5]:
            subcube2 = subcube[np.abs(subcube.new_fwhm-new_fwhm)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([exp_time, new_fwhm,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('exptime')
    plt.ylabel('$new_{fwhm}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 20)
    plot_data = []
    for exp_time in [120, 300]:
        subcube = cube[np.abs(cube.exp_time-exp_time)<0.1]
        for px_scale in [0.3, 0.7, 1.4]:
            subcube2 = subcube[np.abs(subcube.px_scale-px_scale)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([exp_time, px_scale,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('exptime')
    plt.ylabel('px scale')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 21)
    plot_data = []
    for exp_time in [120, 300]:
        subcube = cube[np.abs(cube.exp_time-exp_time)<0.1]
        for ref_back_sbright in [20., 21., 22.]:
            subcube2 = subcube[np.abs(subcube.ref_back_sbright-ref_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([exp_time, ref_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('exptime')
    plt.ylabel('$ref_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 22)
    plot_data = []
    for exp_time in [120, 300]:
        subcube = cube[np.abs(cube.exp_time-exp_time)<0.1]
        for new_back_sbright in [20, 19., 18]:
            subcube2 = subcube[np.abs(subcube.new_back_sbright-new_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([exp_time, new_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('exptime')
    plt.ylabel('$new_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')


    plt.subplot(7, 4, 23)
    plot_data = []
    for new_fwhm in [1.3, 1.9, 2.5]:
        subcube = cube[np.abs(cube.new_fwhm-new_fwhm)<0.1]
        for px_scale in [0.3, 0.7, 1.4]:
            subcube2 = subcube[np.abs(subcube.px_scale-px_scale)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([new_fwhm, px_scale,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$new_{fwhm}$')
    plt.ylabel('px scale')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 24)
    plot_data = []
    for new_fwhm in [1.3, 1.9, 2.5]:
        subcube = cube[np.abs(cube.new_fwhm-new_fwhm)<0.1]
        for ref_back_sbright in [20., 21., 22.]:
            subcube2 = subcube[np.abs(subcube.ref_back_sbright-ref_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([new_fwhm, ref_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$new_{fwhm}$')
    plt.ylabel('$ref_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 25)
    plot_data = []
    for new_fwhm in [1.3, 1.9, 2.5]:
        subcube = cube[np.abs(cube.new_fwhm-new_fwhm)<0.1]
        for new_back_sbright in [20, 19., 18]:
            subcube2 = subcube[np.abs(subcube.new_back_sbright-new_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([new_fwhm, new_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$new_{fwhm}$')
    plt.ylabel('$new_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 26)
    plot_data = []
    for px_scale in [0.3, 0.7, 1.4]:
        subcube = cube[np.abs(cube.px_scale-px_scale)<0.1]
        for ref_back_sbright in [20., 21., 22.]:
            subcube2 = subcube[np.abs(subcube.ref_back_sbright-ref_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([px_scale, ref_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('px scale')
    plt.ylabel('$ref_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 27)
    plot_data = []
    for px_scale in [0.3, 0.7, 1.4]:
        subcube = cube[np.abs(cube.px_scale-px_scale)<0.1]
        for new_back_sbright in [20, 19., 18]:
            subcube2 = subcube[np.abs(subcube.new_back_sbright-new_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([px_scale, new_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('px scale')
    plt.ylabel('$new_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.subplot(7, 4, 28)
    plot_data = []
    for ref_back_sbright in [20., 21., 22.]:
        subcube = cube[np.abs(cube.ref_back_sbright-ref_back_sbright)<0.1]
        for new_back_sbright in [20, 19., 18]:
            subcube2 = subcube[np.abs(subcube.new_back_sbright-new_back_sbright)<0.1]
            mean_goyet, med_goyet, std_goyet = sigma_clipped_stats(
                subcube2.goyet.values)
            plot_data.append([ref_back_sbright, new_back_sbright,
                              mean_goyet, med_goyet, std_goyet])
    plot_data = np.asarray(plot_data)
    plt.scatter(x=plot_data[:,0], y=plot_data[:,1],
                c=plot_data[:,2], s=10./plot_data[:,4])
    plt.xlabel('$ref_{backgorund}$')
    plt.ylabel('$new_{backgorund}$')
    plt.colorbar(label='goyet=$<\delta m /m>$')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'goyet_vs_pars_{}.svg'.format(dia)), dpi=400)
    plt.clf()

goyet_vs_pars_plot(dt_sps, dia='scorr')
goyet_vs_pars_plot(dt_zps, dia='zackay')
goyet_vs_pars_plot(dt_ois, dia='bramich')
goyet_vs_pars_plot(dt_hot, dia='alard')



#~ if __name__ == '__main__':
    #~ import sys
    #~ print(sys.argv)
    #~ m1 = sys.argv[1]
    #~ path = sys.argv[2]
    #~ sys.exit(main(float(m1), path))
