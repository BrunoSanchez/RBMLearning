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


storefile = '/mnt/clemente/bos0109/table_store.h5'

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

    simulated = store['simulated']
    simulations = store['simulations']

    simulations = simulations[simulations.failed_to_subtract==False]
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

# =============================================================================
# plot de deltas de magnitud sobre magnitud (goyet)
# =============================================================================
    plt.figure(figsize=(9,3))
    plt.title('mag offsets over mag simulated')
    plt.subplot(141)
    dmag = dt_zps.goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag zps')

    plt.subplot(142)
    dmag = dt_ois.goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag ois')

    plt.subplot(143)
    dmag = dt_hot.goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag hot')

    plt.subplot(144)
    dmag = dt_sps.goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag sps')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'goyet_full.svg'), dpi=400)
    plt.clf()

# =============================================================================
# plot de deltas de magnitud sobre magnitud (goyet)
# =============================================================================
    plt.figure(figsize=(9,3))
    plt.title('mag offsets over mag simulated')
    plt.subplot(141)
    dmag = dt_zps.goyet_iso
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag zps')

    plt.subplot(142)
    dmag = dt_ois.goyet_iso
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag ois')

    plt.subplot(143)
    dmag = dt_hot.goyet_iso
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag hot')

    plt.subplot(144)
    dmag = dt_sps.goyet_iso
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag sps')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'goyet_iso_full.svg'), dpi=400)
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
# Mean goyet alto
# =============================================================================
# =============================================================================
# Seleccionamos los mean_goyet
# =============================================================================
    pars = ['mean_goyet', 'image_id', 'id_simulation', 'mag', 'sim_mag',
            'goyet', 'goyet_iso', 'mean_goyet_iso', 'IS_REAL']
    subset_zps = dt_zps[pars]
    subset_ois = dt_ois[pars]
    subset_sps = dt_sps[pars]
    subset_hot = dt_hot[pars]

    del(dt_zps)
    del(dt_sps)
    del(dt_ois)
    del(dt_hot)
    gc.collect()

# =============================================================================
# Veamos la distrubicion general de las medias de los goyet
# =============================================================================
    plt.figure(figsize=(9,3))
    plt.title('mean goyets for each technique')
    plt.subplot(141)
    dmag = subset_zps[['mean_goyet', 'image_id']].drop_duplicates().mean_goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('mean goyet zps')

    plt.subplot(142)
    dmag = subset_ois[['mean_goyet', 'image_id']].drop_duplicates().mean_goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('mean goyet ois')

    plt.subplot(143)
    dmag = subset_hot[['mean_goyet', 'image_id']].drop_duplicates().mean_goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('mean goyet hot')

    plt.subplot(144)
    dmag = subset_sps[['mean_goyet', 'image_id']].drop_duplicates().mean_goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('mean goyet sps')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mean_goyets.svg'), dpi=400)
    plt.clf()

# =============================================================================
# Veamos la distrubicion general de las medias de los goyet ISO
# =============================================================================
    plt.figure(figsize=(9,3))
    plt.title('mean goyets_iso for each technique')
    plt.subplot(141)
    dmag = subset_zps[['mean_goyet_iso', 'image_id']].drop_duplicates().mean_goyet_iso
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('mean goyet_iso zps')

    plt.subplot(142)
    dmag = subset_ois[['mean_goyet_iso', 'image_id']].drop_duplicates().mean_goyet_iso
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('mean goyet_iso ois')

    plt.subplot(143)
    dmag = subset_hot[['mean_goyet_iso', 'image_id']].drop_duplicates().mean_goyet_iso
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('mean goyet_iso hot')

    plt.subplot(144)
    dmag = subset_sps[['mean_goyet_iso', 'image_id']].drop_duplicates().mean_goyet_iso
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('mean goyet_iso sps')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mean_goyets_iso.svg'), dpi=400)
    plt.clf()


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
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag zps')

    plt.subplot(142)
    dmag = subset_ois_hi.goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag ois')

    plt.subplot(143)
    dmag = subset_hot_hi.goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag hot')

    plt.subplot(144)
    dmag = subset_sps_hi.goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag sps')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'delta_over_mags_hi_goyet.svg'), dpi=400)
    plt.clf()

# =============================================================================
# Como quedan los diagramas de error de magnitud vs magnitud simulada
# =============================================================================
    plt.figure(figsize=(8,4))
    bins = np.arange(6.5, 26.5, .5)
    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_hot_hi, bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='g--', label='Hotpants')

    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_sps_hi, bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='m:', label='Scorr')

    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_zps_hi, bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='b.-', label='Zackay')

    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_ois_hi, bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='ro-', label='Bramich')

    plt.tick_params(labelsize=16)
    plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    plt.xlabel('Sim Mag', fontsize=16)
    plt.title('Simulated Data', fontsize=14)
    plt.legend(loc='best', fontsize=14)

    plt.xlim(10, 22.5)
    plt.ylim(-3, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_hi_goyet.svg'),
                format='svg', dpi=480)

# =============================================================================
# Liberamos algo de memoria
# =============================================================================

    del(subset_hot_hi)
    del(subset_sps_hi)
    del(subset_zps_hi)
    del(subset_ois_hi)

    gc.collect()

# =============================================================================
# Mean goyet bajo
# =============================================================================
# =============================================================================
# vetamos por mean goyet
# =============================================================================
    subset_zps_lo = subset_zps[subset_zps.mean_goyet<0.01]
    subset_hot_lo = subset_hot[subset_hot.mean_goyet<0.01]
    subset_sps_lo = subset_sps[subset_sps.mean_goyet<0.01]
    subset_ois_lo = subset_ois[subset_ois.mean_goyet<0.01]

# =============================================================================
# Como quedan las distros de goyet individuales
# =============================================================================
    plt.figure(figsize=(9,3))
    plt.title('mag offsets over mag simulated')
    plt.subplot(141)
    dmag = subset_zps_lo.goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag zps')

    plt.subplot(142)
    dmag = subset_ois_lo.goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag ois')

    plt.subplot(143)
    dmag = subset_hot_lo.goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag hot')

    plt.subplot(144)
    dmag = subset_sps_lo.goyet
    dmag = dmag.dropna()
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag sps')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'delta_over_mags_lo_goyet.svg'), dpi=400)
    plt.clf()

# =============================================================================
# Como quedan los diagramas de error de magnitud vs magnitud simulada
# =============================================================================

    plt.figure(figsize=(8,4))
    bins = np.arange(6.5, 26.5, .5)
    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_sps_lo.dropna(), bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='m:', label='Scorr')

    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_zps_lo.dropna(), bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='b.-', label='Zackay')

    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_ois_lo.dropna(), bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='ro-', label='Bramich')

    mean_det, stdv_det, sqrtn, mean_sim = cf.binning_res(subset_hot_lo.dropna(), bins=bins)
    plt.errorbar(mean_sim, mean_det, yerr=stdv_det/sqrtn, fmt='g--', label='Hotpants')

    plt.tick_params(labelsize=16)
    plt.ylabel('Mag Aper - Sim Mag', fontsize=16)
    plt.xlabel('Sim Mag', fontsize=16)
    plt.title('Simulated Data', fontsize=14)
    plt.legend(loc='best', fontsize=14)

    plt.xlim(10, 22.5)
    plt.ylim(-3, 3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mag_diff_vs_simmag_lo_goyet.svg'),
                format='svg', dpi=480)

# =============================================================================
#  Queremos los image id con buen goyet y ver quienes son
# =============================================================================
    ids_mix = store['ids_mix']

    pars = ['image_id', 'mean_goyet', 'mean_goyet_iso', 'id_simulation']

    sel_zps = pd.merge(left=subset_zps[pars].drop_duplicates(),
                       right=ids_mix[['simulation_id', 'image_id']],
                       left_on='image_id', right_on='image_id',
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
    ##
    merged = pd.merge(left=sel_zps, right=sel_sps,
                      left_on='simulation_id', right_on='simulation_id',
                      how='inner', suffixes=('_zps', '_sps'))

    merged = pd.merge(left=merged, right=sel_ois,
                      left_on='simulation_id', right_on='simulation_id',
                      how='inner', suffixes=('', '_ois'))

    merged = pd.merge(left=merged, right=sel_hot,
                      left_on='simulation_id', right_on='simulation_id',
                      how='inner', suffixes=('', '_hot2'))

    del(sel_zps)
    del(sel_sps)
    del(sel_ois)
    del(sel_hot)
    gc.collect()
# =============================================================================
# Simplemente usamos los thresholds definidos antes
# =============================================================================

    merged['has_goyet_zps'] = merged['mean_goyet_zps'] < 0.01
    merged['has_goyet_sps'] = merged['mean_goyet_sps'] < 0.01
    merged['has_goyet_ois'] = merged['mean_goyet'] < 0.01
    merged['has_goyet_hot'] = merged['mean_goyet_hot2'] < 0.01

    merged['mix_goyet'] = merged.has_goyet_zps.astype(int) + \
                          merged.has_goyet_sps.astype(int) + \
                          merged.has_goyet_ois.astype(int) + \
                          merged.has_goyet_hot.astype(int)

# =============================================================================
# Distribucion de esta sumatoria
# =============================================================================
    plt.figure(figsize=(4,3))
    plt.title('images passing goyet threshold')

    plt.hist(merged.mix_goyet)
    plt.xlabel('combined goyet')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mix_goyets.svg'), dpi=400)

# =============================================================================
#  Cortamos por la suma... tiene que valer 3 o mas
# =============================================================================

    merged['selected'] = merged.mix_goyet>=3

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
    #selection = merged[merged.selected==True]

    und_z = pd.merge(left=merged[['image_id_zps', 'selected']],
                     right=store['und_z'],
                     left_on='image_id_zps', right_on='image_id',
                     how='right')[['selected', 'app_mag', 'simulated_id']]
    und_z = und_z[und_z.selected==True].drop_duplicates()

    und_s = pd.merge(left=merged[['image_id_sps', 'selected']],
                     right=store['und_s'],
                     left_on='image_id_sps', right_on='image_id',
                     how='right')[['selected', 'app_mag', 'simulated_id']]
    und_s = und_s[und_s.selected==True].drop_duplicates()

    und_h = pd.merge(left=merged[['image_id_hot', 'selected']],
                     right=store['und_h'],
                     left_on='image_id_hot', right_on='image_id',
                     how='right')[['selected', 'app_mag', 'simulated_id']]
    und_h = und_h[und_h.selected==True].drop_duplicates()

    und_o = pd.merge(left=merged[['image_id_ois', 'selected']],
                     right=store['und_b'],
                     left_on='image_id_ois', right_on='image_id',
                     how='right')[['selected', 'app_mag', 'simulated_id']]
    und_o = und_o[und_o.selected==True].drop_duplicates()

    import ipdb; ipdb.set_trace()
    size=200
    pars = ['simulation_id','app_mag']
    res = []
    for i_chunk, chunk in enumerate(np.array_split(simus[pars], size)):
        cross = pd.merge(left=merged[['simulation_id', 'selected']],
                         right=chunk,
                         on='simulation_id', how='right', sort=False, copy=False)
        res.append(cross[cross.selected==True])
        if i_chunk%2:
            interm_res = cf.optimize_df(pd.concat(res))
            store['intermediate_res_{}'.format(i_chunk)] = interm_res
            store.flush()
            del(interm_res)
            res = []

    import ipdb; ipdb.set_trace()

    res = []
    for i in range(size):
        if i%2:
            res.append(store['intermediate_res_{}'.format(i)]

    simus = pd.concat(res)

    #~ simus = pd.merge(left=merged[['simulation_id', 'selected']],
                      #~ right=simus[['simulation_id','app_mag']],
                      #~ on='simulation_id', how='right', sort=False, copy=False)
    #~ simus = simus[simus.selected==True]
    #simus.drop_duplicates(inplace=True)

    d_zps = pd.merge(left=merged[['image_id_zps', 'selected']],
                     right=subset_zps[['image_id', 'mag', 'IS_REAL']],
                     left_on='image_id_zps', right_on='image_id',
                     how='right')
    d_zps = d_zps[d_zps.selected==True].drop_duplicates()

    d_sps = pd.merge(left=merged[['image_id_sps', 'selected']],
                     right=subset_sps[['image_id', 'mag', 'IS_REAL']],
                     left_on='image_id_sps', right_on='image_id',
                     how='right')
    d_sps = d_sps[d_sps.selected==True].drop_duplicates()

    d_hot = pd.merge(left=merged[['image_id_hot', 'selected']],
                     right=subset_hot[['image_id', 'mag', 'IS_REAL']],
                     left_on='image_id_hot', right_on='image_id',
                     how='right')
    d_hot = d_hot[d_hot.selected==True].drop_duplicates()

    d_ois = pd.merge(left=merged[['image_id_ois', 'selected']],
                     right=subset_ois[['image_id', 'mag', 'IS_REAL']],
                     left_on='image_id_ois', right_on='image_id',
                     how='right')
    d_ois = d_ois[d_ois.selected==True].drop_duplicates()

# =============================================================================
# plot de funcion de luminosidad inyectada
# =============================================================================
    plt.figure(figsize=(6,3))
    plt.hist(simus2['app_mag'], cumulative=False, bins=25, log=True)
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
    plt.rcParams['text.usetex'] = True

    plt.subplot(131)
    x_bins, vals = cf.custom_histogram(simus.app_mag.values, bins=bins,
                                    cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'black', label='Injected')

    x_bins, vals = cf.custom_histogram(d_ois[d_ois.IS_REAL==True].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'ro-', label='Bramich')

    x_bins, vals = cf.custom_histogram(d_zps[d_zps.IS_REAL==True].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'b.-', label='Zackay')

    x_bins, vals = cf.custom_histogram(d_sps[d_sps.IS_REAL==True].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'm:', label='$S_{corr}$')

    x_bins, vals = cf.custom_histogram(d_hot[d_hot.IS_REAL==True].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'g--', label='A-Lupton')

    if cumulative:
        plt.ylabel(r'$N(>r)$', fontsize=16)
    else:
        plt.ylabel(r'$N(m)dm$', fontsize=16)
    plt.ylim(1, 1e8)
    plt.xlim(7., 23.5)
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
    x_bins, vals = cf.custom_histogram(d_ois[d_ois.IS_REAL==False].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'ro-', label='Bramich')
    x_bins, vals = cf.custom_histogram(d_zps[d_zps.IS_REAL==False].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'b.-', label='Zackay')
    x_bins, vals = cf.custom_histogram(d_sps[d_sps.IS_REAL==False].mag.values,
                                    bins=bins, cumulative=cumulative)
    plt.semilogy(x_bins, vals, 'm:', label='$S_{corr}$')
    x_bins, vals = cf.custom_histogram(d_hot[d_hot.IS_REAL==False].mag.values,
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


    store.close()
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
