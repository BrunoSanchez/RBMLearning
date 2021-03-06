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
dt_zps = store['dt_ois_iso']
dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
dt_zps = cf.optimize_df(dt_zps)
dt_zps['VALID_MAG'] = dt_zps['MAG_APER']<30
#~ dt_zps['mag_offset'] = dt_zps['sim_mag'] - dt_zps['MAG_APER']
#~ grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
#~ dd = grouped.apply(lambda df: sigma_clipped_stats(df['mag_offset'])[0])
#~ dd.name = 'mean_offset'
#~ dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
#~ #mean_offset, median_offset, std_offset = sigma_clipped_stats(dt_zps.mag_offset)
#~ dt_zps['mag'] = dt_zps['MAG_APER'] + dt_zps['mean_offset']
dt_zps['goyet'] = np.abs(dt_zps['sim_mag'] - dt_zps['mag'])/dt_zps['sim_mag']
grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet'])[0])
dd.name = 'mean_goyet'
dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
dt_zps['goyet_iso'] = np.abs(dt_zps['sim_mag'] - dt_zps['mag_iso'])/dt_zps['sim_mag']
grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet_iso'])[0])
dd.name = 'mean_goyet_iso'
dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
dt_ois = dt_zps

dt_zps = store['dt_sps_iso']
dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
dt_zps = cf.optimize_df(dt_zps)
dt_zps['MAG_APER'] = -2.5*np.log10(dt_zps.cflux)
dt_zps['VALID_MAG'] = dt_zps['MAG_APER']<30
#~ dt_zps['mag_offset'] = dt_zps['sim_mag'] - dt_zps['MAG_APER']
#~ grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
#~ dd = grouped.apply(lambda df: sigma_clipped_stats(df['mag_offset'])[0])
#~ dd.name = 'mean_offset'
#~ dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
#~ #mean_offset, median_offset, std_offset = sigma_clipped_stats(dt_zps.mag_offset)
#~ dt_zps['mag'] = dt_zps['MAG_APER'] + dt_zps['mean_offset']
dt_zps['goyet'] = np.abs(dt_zps['sim_mag'] - dt_zps['mag'])/dt_zps['sim_mag']
grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet'])[0])
dd.name = 'mean_goyet'
dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
dt_sps = dt_zps

dt_zps = store['dt_hot_iso']
dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
dt_zps = cf.optimize_df(dt_zps)
dt_zps['VALID_MAG'] = dt_zps['MAG_APER']<30
#~ dt_zps['mag_offset'] = dt_zps['sim_mag'] - dt_zps['MAG_APER']
#~ grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
#~ dd = grouped.apply(lambda df: sigma_clipped_stats(df['mag_offset'])[0])
#~ dd.name = 'mean_offset'
#~ dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
#~ #mean_offset, median_offset, std_offset = sigma_clipped_stats(dt_zps.mag_offset)
#~ dt_zps['mag'] = dt_zps['MAG_APER'] + dt_zps['mean_offset']
dt_zps['goyet'] = np.abs(dt_zps['sim_mag'] - dt_zps['mag'])/dt_zps['sim_mag']
grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet'])[0])
dd.name = 'mean_goyet'
dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
dt_zps['goyet_iso'] = np.abs(dt_zps['sim_mag'] - dt_zps['mag_iso'])/dt_zps['sim_mag']
grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet_iso'])[0])
dd.name = 'mean_goyet_iso'
dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
dt_hot = dt_zps

dt_zps = store['dt_zps_iso']
dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
dt_zps = cf.optimize_df(dt_zps)
dt_zps['VALID_MAG'] = dt_zps['MAG_APER']<30
#~ dt_zps['mag_offset'] = dt_zps['sim_mag'] - dt_zps['MAG_APER']
#~ grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
#~ dd = grouped.apply(lambda df: sigma_clipped_stats(df['mag_offset'])[0])
#~ dd.name = 'mean_offset'
#~ dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
#~ #mean_offset, median_offset, std_offset = sigma_clipped_stats(dt_zps.mag_offset)
#~ dt_zps['mag'] = dt_zps['MAG_APER'] + dt_zps['mean_offset']
dt_zps['goyet'] = np.abs(dt_zps['sim_mag'] - dt_zps['mag'])/dt_zps['sim_mag']
grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet'])[0])
dd.name = 'mean_goyet'
dt_zps = pd.merge(dt_zps, dd.to_frame(), on='image_id', how='left')
dt_zps['goyet_iso'] = np.abs(dt_zps['sim_mag'] - dt_zps['mag_iso'])/dt_zps['sim_mag']
grouped = dt_zps.dropna().groupby(['image_id'], sort=False)
dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet_iso'])[0])
dd.name = 'mean_goyet_iso'
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

    #~ plt.plot(data.sim_mag, data.goyet, '.')
    #~ plt.xlabel('simulated mag')
    #~ plt.ylabel('goyet')
    #~ plt.savefig(os.path.join(plot_dir, 'simulated_goyet_{}.svg'.format(dia)), dpi=400)

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


# =============================================================================
# plot de deltas de magnitud sobre magnitud (goyet)
# =============================================================================
plt.figure(figsize=(9,3))
plt.title('mag offsets over mag simulated')
plt.subplot(141)
#~ dmag = dt_zps[(dt_zps.VALID_MAG==True)].goyet
dmag = dt_zps.goyet
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag zps')

plt.subplot(142)
#~ dmag = dt_ois[(dt_ois.VALID_MAG==True)].goyet
dmag = dt_ois.goyet
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag ois')

plt.subplot(143)
#~ dmag = dt_hot[(dt_hot.VALID_MAG==True)].goyet
dmag = dt_hot.goyet
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag hot')

plt.subplot(144)
#~ dmag = dt_sps[(dt_sps.VALID_MAG==True)].goyet
dmag = dt_sps.goyet
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag sps')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'delta_over_mags.svg'), dpi=400)
plt.clf()

# =============================================================================
# plot de deltas de magnitud sobre magnitud (goyet)
# =============================================================================
plt.figure(figsize=(9,3))
plt.title('mag_iso offsets over mag simulated')
plt.subplot(141)
dmag = dt_zps[(dt_zps.VALID_MAG==True)].goyet_iso
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag zps')

plt.subplot(142)
dmag = dt_ois[(dt_ois.VALID_MAG==True)].goyet_iso
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag ois')

plt.subplot(143)
dmag = dt_hot[(dt_hot.VALID_MAG==True)].goyet_iso
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
plt.savefig(os.path.join(plot_dir, 'delta_over_mags_iso.svg'), dpi=400)
plt.clf()


# =============================================================================
# GOYET ALTO
# =============================================================================

# =============================================================================
# Seleccionamos los mean_goyet
# =============================================================================
pars = ['mean_goyet', 'image_id', 'id_simulation', 'goyet', 'goyet_iso']
subset_zps = dt_zps[pars]
subset_ois = dt_ois[pars]
subset_sps = dt_sps[pars]
subset_hot = dt_hot[pars]

# =============================================================================
# vetamos por mean goyet
# =============================================================================
subset_zps_hi = subset_zps[subset_zps.mean_goyet>=0.01]
subset_hot_hi = subset_hot[subset_hot.mean_goyet>=0.01]
subset_sps_hi = subset_sps[subset_sps.mean_goyet>=0.01]
subset_ois_hi = subset_ois[subset_ois.mean_goyet>=0.01]

# =============================================================================
# Como quedan las distros de goyet individuales
# =============================================================================
plt.figure(figsize=(9,3))
plt.title('mag offsets over mag simulated')
plt.subplot(141)
dmag = subset_zps_hi.goyet
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag zps')

plt.subplot(142)
dmag = subset_ois_hi.goyet
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag ois')

plt.subplot(143)
dmag = subset_hot_hi.goyet
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag hot')

plt.subplot(144)
dmag = subset_sps_hi.goyet
dmag = dmag.dropna()
#dmag = dmag.mag_offset/dmag.sim_mag
plt.hist(dmag, log=True)
plt.xlabel('delta mag sps')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'delta_over_mags_hi_goyet.svg'), dpi=400)
plt.clf()



# =============================================================================
# Drop duplicates
# =============================================================================

subset_zps.drop_duplicates(inplace=True)
subset_ois.drop_duplicates(inplace=True)
subset_sps.drop_duplicates(inplace=True)
subset_hot.drop_duplicates(inplace=True)


# =============================================================================
# merge the tables
# =============================================================================

merged_zps = pd.merge(left=subset_zps_hi, right=simulations,
                  left_on='id_simulation', right_on='id',
                  how='inner', suffixes=('_zps', 'simus'))

merged_sps = pd.merge(left=subset_sps_hi, right=simulations,
                  left_on='id_simulation', right_on='id',
                  how='inner', suffixes=('_sps', 'simus'))

merged_ois = pd.merge(left=subset_ois_hi, right=simulations,
                  left_on='id_simulation', right_on='id',
                  how='inner', suffixes=('_ois', 'simus'))

merged_hot = pd.merge(left=subset_hot_hi, right=simulations,
                  left_on='id_simulation', right_on='id',
                  how='inner', suffixes=('_hot', 'simus'))

combined_merge1 = pd.merge(left=merged_zps, right=merged_sps,
                          left_on='image_id', right_on='image_id',
                          how='inner', suffixes=('_zps','_sps'))

combined_merge2 = pd.merge(left=merged_hot, right=merged_ois,
                           left_on='image_id', right_on='image_id',
                           how='inner', suffixes=('_hot','_ois'))

comb_merge = pd.merge(left=combined_merge1, right=combined_merge2,
                      left_on='image_id', right_on='image_id',
                      how='inner', suffixes=('_1','_2'))

# =============================================================================
# Ahora a ver que muestra quedo
# =============================================================================
plt.rcParams['text.usetex'] = False

#  checkings
print(np.any(comb_merge['id_simulation_zps']!=comb_merge['id_simulation_sps']))
print(np.any(comb_merge['id_simulation_zps']!=comb_merge['id_simulation_ois']))
print(np.any(comb_merge['id_simulation_zps']!=comb_merge['id_simulation_hot']))
print(np.any(comb_merge['ref_starslope_zps']!=comb_merge['ref_starslope_sps']))
print(np.any(comb_merge['ref_starslope_zps']!=comb_merge['ref_starslope_ois']))
print(np.any(comb_merge['ref_starslope_zps']!=comb_merge['ref_starslope_hot']))
print(np.any(comb_merge['px_scale_zps']!=comb_merge['px_scale_ois']))
print(np.any(comb_merge['px_scale_zps']!=comb_merge['px_scale_hot']))
print(np.any(comb_merge['px_scale_zps']!=comb_merge['px_scale_sps']))

comb_merge = comb_merge[['image_id', 'id_simulation_zps',
                         'ref_starzp_zps', 'ref_starslope_zps',
                         'ref_fwhm_zps', 'new_fwhm_zps',
                         'm1_diam_zps', 'm2_diam_zps', 'eff_col_zps',
                         'px_scale_zps', 'ref_back_sbright_zps',
                         'new_back_sbright_zps', 'exp_time_zps', 'mean_goyet_zps',
                         'mean_goyet_sps', 'mean_goyet_hot', 'mean_goyet_ois']]

pd.plotting.scatter_matrix(comb_merge[['ref_starzp_zps', 'ref_starslope_zps',
                         'ref_fwhm_zps', 'new_fwhm_zps','px_scale_zps', 'ref_back_sbright_zps',
                         'new_back_sbright_zps', 'exp_time_zps', 'mean_goyet_zps',
                         'mean_goyet_sps', 'mean_goyet_hot', 'mean_goyet_ois']],
                         alpha=0.1, diagonal='hist', figsize=(12, 12))
plt.savefig(os.path.join(plot_dir, 'merge_goyet_hi_25_scatter_matrix.png'),
            dpi=400)

# =============================================================================
#  Ellegimos una imagen con goyet ALTO y graficamos dm vs m
# =============================================================================

images = comb_merge[comb_merge.mean_goyet_zps==np.max(comb_merge.mean_goyet_zps)]
id_selected_image = images.iloc[np.random.randint(len(images))]

plt.figure(figsize=(6, 3))

data_zps = dt_zps[dt_zps.image_id==id_selected_image.image_id].dropna()
plt.plot(data_zps.sim_mag, data_zps.mag, '.', label='zackay')

data_sps = dt_sps[dt_sps.image_id==id_selected_image.image_id].dropna()
plt.plot(data_sps.sim_mag, data_sps.mag, '.', label='scorr')

data_ois = dt_ois[dt_ois.image_id==id_selected_image.image_id].dropna()
plt.plot(data_ois.sim_mag, data_ois.mag, '.', label='bramich')

data_hot = dt_hot[dt_hot.image_id==id_selected_image.image_id].dropna()
plt.plot(data_hot.sim_mag, data_hot.mag, '.', label='alard')

plt.xlim(14, 22)
plt.ylim(14, 22)
plt.xlabel('sim mag')
plt.ylabel('recovered mag')
plt.legend(loc='best')

plt.savefig(os.path.join(plot_dir, 'high_goyet_sim_mag_vs_mag.png'), dpi=400)

# =============================================================================
#  Ellegimos una imagen con goyet bsjo y graficamos dm vs m
# =============================================================================

images = comb_merge[comb_merge.mean_goyet_zps==np.max(comb_merge.mean_goyet_zps)]
id_selected_image = images.iloc[np.random.randint(len(images))]

plt.figure(figsize=(6, 3))

data_zps = dt_zps[dt_zps.image_id==id_selected_image.image_id].dropna()
plt.plot(data_zps.sim_mag, data_zps.mag_iso, '.', label='zackay')

data_sps = dt_sps[dt_sps.image_id==id_selected_image.image_id].dropna()
plt.plot(data_sps.sim_mag, data_sps.mag, '.', label='scorr')

data_ois = dt_ois[dt_ois.image_id==id_selected_image.image_id].dropna()
plt.plot(data_ois.sim_mag, data_ois.mag_iso, '.', label='bramich')

data_hot = dt_hot[dt_hot.image_id==id_selected_image.image_id].dropna()
plt.plot(data_hot.sim_mag, data_hot.mag_iso, '.', label='alard')

plt.xlim(14, 22)
plt.ylim(14, 22)
plt.xlabel('sim mag')
plt.ylabel('recovered mag_iso')
plt.legend(loc='best')

plt.savefig(os.path.join(plot_dir, 'high_goyet_sim_mag_vs_mag_iso.png'), dpi=400)

# =============================================================================
# GOYET BAJO
# =============================================================================

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
# plot de goyet factor vs pars
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

        #~ plt.plot(data.sim_mag, data.goyet, '.')
        #~ plt.xlabel('simulated mag')
        #~ plt.ylabel('goyet')
        #~ plt.savefig(os.path.join(plot_dir, 'simulated_goyet_{}.svg'.format(dia)), dpi=400)

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

    if goyet_vs_pars:
        goyet_vs_pars_plot(dt_sps, dia='scorr')
        goyet_vs_pars_plot(dt_zps, dia='zackay')
        goyet_vs_pars_plot(dt_ois, dia='bramich')
        goyet_vs_pars_plot(dt_hot, dia='alard')

    gc.collect()


# =============================================================================
# vetamos por mean goyet
# =============================================================================
subset_zps = subset_zps[subset_zps.mean_goyet<=0.01]
subset_hot = subset_hot[subset_hot.mean_goyet<=0.01]
subset_sps = subset_sps[subset_sps.mean_goyet<=0.01]
subset_ois = subset_ois[subset_ois.mean_goyet<=0.01]

merged_zps = pd.merge(left=subset_zps, right=simulations,
                  left_on='id_simulation', right_on='id',
                  how='inner', suffixes=('_zps', 'simus'))

merged_sps = pd.merge(left=subset_sps, right=simulations,
                  left_on='id_simulation', right_on='id',
                  how='inner', suffixes=('_sps', 'simus'))

merged_ois = pd.merge(left=subset_ois, right=simulations,
                  left_on='id_simulation', right_on='id',
                  how='inner', suffixes=('_ois', 'simus'))

merged_hot = pd.merge(left=subset_hot, right=simulations,
                  left_on='id_simulation', right_on='id',
                  how='inner', suffixes=('_hot', 'simus'))

combined_merge1 = pd.merge(left=merged_zps, right=merged_sps,
                          left_on='image_id', right_on='image_id',
                          how='inner', suffixes=('_zps','_sps'))

combined_merge2 = pd.merge(left=merged_hot, right=merged_ois,
                           left_on='image_id', right_on='image_id',
                           how='inner', suffixes=('_hot','_ois'))

comb_merge = pd.merge(left=combined_merge1, right=combined_merge2,
                      left_on='image_id', right_on='image_id',
                      how='inner', suffixes=('_1','_2'))

# =============================================================================
# Ahora a ver que muestra quedo
# =============================================================================
plt.rcParams['text.usetex'] = False

#  checkings
print(np.any(comb_merge['id_simulation_zps']!=comb_merge['id_simulation_sps']))
print(np.any(comb_merge['id_simulation_zps']!=comb_merge['id_simulation_ois']))
print(np.any(comb_merge['id_simulation_zps']!=comb_merge['id_simulation_hot']))
print(np.any(comb_merge['ref_starslope_zps']!=comb_merge['ref_starslope_sps']))
print(np.any(comb_merge['ref_starslope_zps']!=comb_merge['ref_starslope_ois']))
print(np.any(comb_merge['ref_starslope_zps']!=comb_merge['ref_starslope_hot']))
print(np.any(comb_merge['px_scale_zps']!=comb_merge['px_scale_ois']))
print(np.any(comb_merge['px_scale_zps']!=comb_merge['px_scale_hot']))
print(np.any(comb_merge['px_scale_zps']!=comb_merge['px_scale_sps']))

comb_merge = comb_merge[['image_id', 'id_simulation_zps',
                         'ref_starzp_zps', 'ref_starslope_zps',
                         'ref_fwhm_zps', 'new_fwhm_zps',
                         'm1_diam_zps', 'm2_diam_zps', 'eff_col_zps',
                         'px_scale_zps', 'ref_back_sbright_zps',
                         'new_back_sbright_zps', 'exp_time_zps', 'mean_goyet_zps',
                         'mean_goyet_sps', 'mean_goyet_hot', 'mean_goyet_ois']]

pd.plotting.scatter_matrix(comb_merge[['ref_starzp_zps', 'ref_starslope_zps',
                         'ref_fwhm_zps', 'new_fwhm_zps','px_scale_zps', 'ref_back_sbright_zps',
                         'new_back_sbright_zps', 'exp_time_zps', 'mean_goyet_zps',
                         'mean_goyet_sps', 'mean_goyet_hot', 'mean_goyet_ois']],
                         alpha=0.1, diagonal='hist', figsize=(12, 12))
plt.savefig(os.path.join(plot_dir, 'merge_goyet_lw_25_scatter_matrix.png'),
            dpi=400)
# =============================================================================
#  Ellegimos una imagen con goyet bsjo y graficamos dm vs m
# =============================================================================

images = comb_merge[comb_merge.mean_goyet_zps<0.005]
id_selected_image = images.iloc[np.random.randint(len(images))]

plt.figure(figsize=(6, 3))

data_zps = dt_zps[dt_zps.image_id==id_selected_image.image_id].dropna()
plt.plot(data_zps.sim_mag, data_zps.mag, '.', label='zackay')

data_sps = dt_sps[dt_sps.image_id==id_selected_image.image_id].dropna()
plt.plot(data_sps.sim_mag, data_sps.mag, '.', label='scorr')

data_ois = dt_ois[dt_ois.image_id==id_selected_image.image_id].dropna()
plt.plot(data_ois.sim_mag, data_ois.mag, '.', label='bramich')

data_hot = dt_hot[dt_hot.image_id==id_selected_image.image_id].dropna()
plt.plot(data_hot.sim_mag, data_hot.mag, '.', label='alard')

plt.xlim(14, 22)
plt.ylim(14, 22)
plt.xlabel('sim mag')
plt.ylabel('recovered mag')
plt.legend(loc='best')

plt.savefig(os.path.join(plot_dir, 'low_goyet_sim_mag_vs_mag.png'), dpi=400)

# =============================================================================
#  Ellegimos una imagen con goyet bsjo y graficamos dm vs m
# =============================================================================

images = comb_merge[comb_merge.mean_goyet_zps<0.005]
id_selected_image = images.iloc[np.random.randint(len(images))]

plt.figure(figsize=(6, 3))

data_zps = dt_zps[dt_zps.image_id==id_selected_image.image_id].dropna()
plt.plot(data_zps.sim_mag, data_zps.mag_iso, '.', label='zackay')

data_sps = dt_sps[dt_sps.image_id==id_selected_image.image_id].dropna()
plt.plot(data_sps.sim_mag, data_sps.mag, '.', label='scorr')

data_ois = dt_ois[dt_ois.image_id==id_selected_image.image_id].dropna()
plt.plot(data_ois.sim_mag, data_ois.mag_iso, '.', label='bramich')

data_hot = dt_hot[dt_hot.image_id==id_selected_image.image_id].dropna()
plt.plot(data_hot.sim_mag, data_hot.mag_iso, '.', label='alard')

plt.xlim(14, 22)
plt.ylim(14, 22)
plt.xlabel('sim mag')
plt.ylabel('recovered mag_iso')
plt.legend(loc='best')

plt.savefig(os.path.join(plot_dir, 'low_goyet_sim_mag_vs_mag_iso.png'), dpi=400)


# =============================================================================
# Distribuciones de goyet vs pars
# =============================================================================

#~ def goyet_vs_pars_plot(dataset, dia='zackay'):
    #~ mag_range = [16, 20]
    #~ mag_bins = np.arange(16, 20, 0.5)
    #~ data = dataset[dataset.sim_mag<=20]
    #~ data = data[data.sim_mag>=16]
    #~ data = data[data.VALID_MAG==True]
    #~ cube = data[['r_scales', 'gx_mag', 'm1_diam',
    #~ 'm2_diam', 'ref_starzp', 'ref_starslope',
    #~ 'ref_fwhm', 'new_fwhm', 'eff_col', 'px_scale', 'ref_back_sbright',
    #~ 'new_back_sbright', 'exp_time', 'mag_offset', 'goyet']]

    #~ cols = ['r_scales', 'gx_mag', 'm1_diam', 'ref_starzp', 'ref_starslope',
            #~ 'ref_fwhm', 'new_fwhm', 'eff_col', 'px_scale', 'ref_back_sbright',
            #~ 'new_back_sbright', 'exp_time', 'mag_offset']

    #~ for a_par in cols:
        #~ subplot()


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

o = dt_ois[['image_id', 'id_simulation']].drop_duplicates()
s = dt_sps[['image_id', 'id_simulation']].drop_duplicates()
h = dt_hot[['image_id', 'id_simulation']].drop_duplicates()
z = dt_zps[['image_id', 'id_simulation']].drop_duplicates()

images_hot = pd.read_sql_query("""SELECT * FROM "ImagesHOT" """, engine)
images_zps = pd.read_sql_query("""SELECT * FROM "Images" """, engine)
images_ois = pd.read_sql_query("""SELECT * FROM "ImagesOIS" """, engine)
images_sps = pd.read_sql_query("""SELECT * FROM "SImages" """, engine)

hot_ids = []
for id_row, hotrow in images_hot.iterrows():
    z_id = images_zps.loc[images_zps.id == hotrow[0]].simulation_id.values
    s_id = images_sps.loc[images_sps.id == hotrow[0]].simulation_id.values
    o_id = images_ois.loc[images_ois.id == hotrow[0]].simulation_id.values
    if (z_id==s_id and s_id==o_id and z_id==o_id):
        hot_ids.append(z_id[0])
    else:
        hot_ids.append(np.nan)

images_hot['simulation_id_imputed'] = hot_ids
mask = images_hot.simulation_id.isna()
mask = mask & ~images_hot.simulation_id_imputed.isna()
images_hot.loc[mask,'simulation_id'] = images_hot.loc[mask,'simulation_id_imputed'].values

simulations = pd.read_sql_query(""" SELECT * FROM "Simulation" """, engine)
simulated = pd.read_sql_query(""" SELECT * FROM "Simulated" """, engine)
simulated = simulated[['simulation_id', 'image_id', 'simage_id', 'scorrimage_id',
       'image_id_ois', 'image_id_hot']].drop_duplicates()

## la tabla posta es la simulated

o = images_ois[['id', 'simulation_id']].drop_duplicates()
s = images_sps[['id', 'simulation_id']].drop_duplicates()
h = images_hot[['id', 'simulation_id']].drop_duplicates()
z = images_zps[['id', 'simulation_id']].drop_duplicates()

print(len(o), len(s), len(h), len(z))
print(np.max(o.id), np.max(s.id), np.max(h.id), np.max(z.id))
print(np.max(o.simulation_id), np.max(s.simulation_id),
      np.max(h.simulation_id), np.max(z.simulation_id))


merge = pd.merge(z, o, on='simulation_id', how='inner',
                 suffixes=('_z','_o'))
print(np.sum(merge.id_z!=merge.id_o))

merge = pd.merge(z, s, on='simulation_id', how='inner',
                 suffixes=('_z','_s'))
print(np.sum(merge.id_z!=merge.id_s))

merge = pd.merge(z, h, on='simulation_id', how='inner',
                 suffixes=('_z','_h'))
print(np.sum(merge.id_z!=merge.id_h))

merge = pd.merge(o, s, on='simulation_id', how='inner',
                 suffixes=('_o','_s'))
print(np.sum(merge.id_s!=merge.id_o))

merge = pd.merge(o, h, on='simulation_id', how='inner',
                 suffixes=('_o','_h'))
print(np.sum(merge.id_o!=merge.id_h))

merge = pd.merge(h, s, on='simulation_id', how='inner',
                 suffixes=('_h','_s'))
print(np.sum(merge.id_h!=merge.id_s))


merge = pd.merge(z, o, on='id', how='inner',
                 suffixes=('_z','_o'))
print(np.sum(merge.simulation_id_z!=merge.simulation_id_o))

merge = pd.merge(z, s, on='id', how='inner',
                 suffixes=('_z','_s'))
print(np.sum(merge.simulation_id_z!=merge.simulation_id_s))

merge = pd.merge(z, h, on='id', how='inner',
                 suffixes=('_z','_h'))
print(np.sum(merge.simulation_id_z!=merge.simulation_id_h))

merge = pd.merge(o, s, on='id', how='inner',
                 suffixes=('_o','_s'))
print(np.sum(merge.simulation_id_s!=merge.simulation_id_o))

merge = pd.merge(o, h, on='id', how='inner',
                 suffixes=('_o','_h'))
print(np.sum(merge.simulation_id_o!=merge.simulation_id_h))

merge = pd.merge(h, s, on='id', how='inner',
                 suffixes=('_h','_s'))
print(np.sum(merge.simulation_id_h!=merge.simulation_id_s))



merge = pd.merge(left=simulated, right=s, left_on='simage_id', right_on='id',
                 how='left', suffixes=('','_s'))
merge = pd.merge(left=merge, right=o, left_on='image_id_ois', right_on='id',
                 how='left', suffixes=('','_o'))
merge = pd.merge(left=merge, right=h, left_on='image_id_hot', right_on='id',
                 how='left', suffixes=('','_h'))
merge = pd.merge(left=merge, right=z, left_on='image_id', right_on='id',
                 how='left', suffixes=('','_z'))

print(np.sum(merge.simulation_id_h!=merge.simulation_id_s))
print(np.sum(merge.simulation_id!=merge.simulation_id_h))
print(merge[merge.simulation_id!=merge.simulation_id_h]['simulation_id_h'])

merge.loc[merge.simulation_id!=merge.simulation_id_h, 'simulation_id_h'] = \
merge.loc[merge.simulation_id!=merge.simulation_id_h, 'simulation_id']

#ahora merge tiene la tabla de ids correctamente alineadas!!

ids0 = simus.simulation_id.drop_duplicates().values

dt_zps = dt_zps.loc[dt_zps['']]



# =============================================================================
# funcion para ml
# =============================================================================
def group_ml(train_data, group_cols=['m1_diam', 'exp_time', 'new_fwhm'],
             target=['IS_REAL'], cols=['mag'], var_thresh=0.1, percentile=30.,
             method='Bramich'):
    rows = []
    for pars, data in train_data.groupby(group_cols):
        train, test = train_test_split(data[cols+target].dropna(), test_size=0.25,
                                       stratify=data[cols+target].dropna().IS_REAL)
        d_ois = train[cols]
        y_ois = train[target]

        scaler = preprocessing.StandardScaler().fit(d_ois)
        X_ois = scaler.transform(d_ois)
        X_test_ois = scaler.transform(test[cols])

        # =============================================================================
        # univariate
        # =============================================================================
        thresh = var_thresh
        sel = VarianceThreshold(threshold=thresh)
        X_ois = sel.fit_transform(X_ois)
        X_test_ois = sel.transform(X_test_ois)
        newcols_ois = d_ois.columns[sel.get_support()]
        print('Dropped columns = {}'.format(d_ois.columns[~sel.get_support()]))
        d_ois = pd.DataFrame(X_ois, columns=newcols_ois)

        percentile = percentile
        scores, selector, selected_cols = select(X_ois, y_ois, percentile)
        scoring_ois = pd.DataFrame(scores, index=newcols_ois, columns=['ois'])
        selection_ois = scoring_ois.loc[newcols_ois.values[selected_cols][0]]
        dat_ois = pd.DataFrame(X_ois, columns=newcols_ois)[selection_ois.index]

        # =============================================================================
        # KNN
        # =============================================================================
        model = neighbors.KNeighborsClassifier(n_neighbors=7, weights='uniform', n_jobs=-1)

        rslt0_knn_ois_uniform = experiment(model, X_ois, y_ois.values.ravel(), printing=True)
        model.fit(X_ois, y_ois.values.ravel())
        preds = model.predict(X_test_ois)
        rslt0_knn_ois_uniform['test_preds'] = preds
        print(metrics.classification_report(test.IS_REAL.values.ravel(), preds))
        acc_knn0 = metrics.accuracy_score(test.IS_REAL.values.ravel(), preds)
        rslt0_knn_ois_uniform['test_bacc'] = acc_knn0

        rslts_knn_ois_uniform = experiment(model, dat_ois.values, y_ois.values.ravel(), printing=True)
        model.fit(dat_ois.values, y_ois.values.ravel())
        preds = model.predict(selector.transform(X_test_ois))
        rslts_knn_ois_uniform['test_preds'] = preds
        print(metrics.classification_report(test.IS_REAL.values.ravel(), preds))
        acc_knn = metrics.accuracy_score(test.IS_REAL.values.ravel(), preds)
        rslt0_knn_ois_uniform['test_bacc'] = acc_knn

        # =============================================================================
        # randomforest
        # =============================================================================
        corr = d_ois.corr()
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        # remove corr columns
        correlated_features = set()
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > 0.8:
                    colname = corr.columns[i]
                    correlated_features.add(colname)
        decorr_ois = d_ois.drop(correlated_features, axis=1)
        corr = decorr_ois.corr()

        model = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)
        ois_importance = importance_perm_kfold(decorr_ois.values, y_ois.values.ravel(),
            model, cols=decorr_ois.columns, method=method)

        res_ois = pd.concat(ois_importance, axis=1)
        full_cols = list(decorr_ois.index).extend(['Random'])
        m = res_ois.mean(axis=1).reindex(full_cols)
        s = res_ois.std(axis=1).reindex(full_cols)

        thresh = m.loc['Random'] + 3*s.loc['Random']
        spikes = m - 3*s
        selected = spikes > thresh
        signif = (m - m.loc['Random'])/s
        selected = signif>2.5
        dat_ois = d_ois[selected[selected].index]

        n_fts = np.min([len(dat_ois.columns), 7])
        model = RandomForestClassifier(n_estimators=800, max_features=n_fts,
                                       min_samples_leaf=20, n_jobs=-1)
        rslts_ois_rforest = experiment(model, dat_ois.values, y_ois.values.ravel(), printing=True)
        model.fit(dat_ois.values, y_ois.values.ravel())
        d_test = pd.DataFrame(X_test_ois, columns=newcols_ois)[selected[selected].index]
        preds = model.predict(d_test.values)
        #rslts_ois_rforest['test_preds'] = preds
        print(metrics.classification_report(test.IS_REAL.values.ravel(), preds))
        acc_rforest = cf.metrics.accuracy_score(test.IS_REAL.values.ravel(), preds)
        #rslts_ois_rforest['test_bacc'] = acc_rforest

        rslt0_ois_rforest = experiment(model, d_ois.values, y_ois.values.ravel(), printing=True)
        model.fit(d_ois.values, y_ois.values.ravel())
        preds = model.predict(X_test_ois)
        # rslt0_ois_rforest['test_preds'] = preds
        print(metrics.classification_report(test.IS_REAL.values.ravel(), preds))
        acc_rforest0 = metrics.accuracy_score(test.IS_REAL.values.ravel(), preds)
        # rslt0_ois_rforest['test_bacc'] = acc_rforest0

# =============================================================================
# SVC
# =============================================================================
        #svc = SVC(kernel='linear',
        #          cache_size=2048,
        #          class_weight='balanced',
        #          probability=True)
        #svc = svm.LinearSVC(dual=False, tol=1e-5)
        rfecv = feature_selection.RFECV(estimator=svc, step=1, cv=StratifiedKFold(6),
                      scoring='accuracy', n_jobs=-1)

        rfecv.fit(np.ascontiguousarray(X_ois), y_ois.values.ravel())
        print("Optimal number of features : {}" .format(rfecv.n_features_))
        sel_cols_ois = newcols_ois[rfecv.support_]
        print(sel_cols_ois)
        dat_ois = d_ois[sel_cols_ois]

        model = svc
        rslts_ois_svc = cf.experiment(model, dat_ois.values, y_ois.values.ravel(), printing=True, probs=False)
        model.fit(dat_ois.values, y_ois.values.ravel())
        preds = model.predict(pd.DataFrame(X_test_ois, columns=newcols_ois)[sel_cols_ois].values)
        #rslts_ois_svc['test_preds'] = preds
        print(cf.metrics.classification_report(test.IS_REAL.values.ravel(), preds))
        acc_svc = cf.metrics.accuracy_score(test.IS_REAL.values.ravel(), preds)
        #rslts_ois_svc['test_bacc'] = acc_svc

        rslt0_ois_svc = cf.experiment(model, d_ois.values, y_ois.values.ravel(), printing=True)
        model.fit(d_ois.values, y_ois.values.ravel())
        preds = model.predict(X_test_ois)
        #rslt0_ois_svc['test_preds'] = preds
        print(cf.metrics.classification_report(test.IS_REAL.values.ravel(), preds))
        acc_svc0 = cf.metrics.accuracy_score(test.IS_REAL.values.ravel(), preds)
        #rslt0_ois_svc['test_bacc'] = acc_svc0


        vals = [acc_knn0, acc_knn, acc_rforest0, acc_rforest, acc_svc, acc_svc0]
        rows.append(list(pars)+vals)

    ml_cols = ['m1_diam', 'exp_time', 'new_fwhm', 'acc_knn0', 'acc_knn',
               'acc_rforest0', 'acc_rforest', 'acc_svc', 'acc_svc0']
    ml_results = pd.DataFrame(rows, columns=ml_cols)
    return ml_results


#~ if __name__ == '__main__':
    #~ import sys
    #~ print(sys.argv)
    #~ m1 = sys.argv[1]
    #~ path = sys.argv[2]
    #~ sys.exit(main(float(m1), path))

    
