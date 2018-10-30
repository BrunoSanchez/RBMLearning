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


def main(m1_diam=1.54, plots_path='./plots/.'):
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
    dt_ois = dt_zps

    dt_zps = store['dt_sps']
    dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
    dt_zps = cf.optimize_df(dt_zps)
    dt_zps['MAG_APER'] = -2.5*np.log10(dt_zps.cflux)
    dt_zps['VALID_MAG'] = dt_zps['MAG_APER']<30
    dt_zps['mag_offset'] = dt_zps['sim_mag'] - dt_zps['MAG_APER']
    dt_sps = dt_zps

    dt_zps = store['dt_hot']
    dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
    dt_zps = cf.optimize_df(dt_zps)
    dt_zps['VALID_MAG'] = dt_zps['MAG_APER']<30
    dt_zps['mag_offset'] = dt_zps['sim_mag'] - dt_zps['MAG_APER']
    dt_hot = dt_zps

    dt_zps = store['dt_zps']
    dt_zps = dt_zps[dt_zps.m1_diam==m1_diam]
    dt_zps = cf.optimize_df(dt_zps)
    dt_zps['VALID_MAG'] = dt_zps['MAG_APER']<30
    dt_zps['mag_offset'] = dt_zps['sim_mag'] - dt_zps['MAG_APER']


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

# =============================================================================
# plot de deltas de magnitud
# =============================================================================
    plt.figure(figsize=(9,3))
    plt.subplot(141)
    dmag = dt_zps[(dt_zps.VALID_MAG==True)*(dt_zps.mag_offset!=np.nan)].mag_offset
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag')

    plt.subplot(142)
    dmag = dt_ois[(dt_ois.VALID_MAG==True)*(dt_ois.mag_offset!=np.nan)].mag_offset
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag')

    plt.subplot(143)
    dmag = dt_ois[(dt_ois.VALID_MAG==True)*(dt_ois.mag_offset!=np.nan)].mag_offset
    plt.hist(dmag, log=True)
    plt.xlabel('delta mag')

    plt.subplot(144)
    dt_zps = store['dt_sps']
    dt_zps['MAG_APER'] = -2.5*np.log10(dt_zps.cflux)
    dt_zps = dt_zps[dt_zps['MAG_APER']<30]
    dt_zps.dropna(axis='rows', inplace=True)
    delta_mag = dt_zps['sim_mag'] - dt_zps['MAG_APER']
    plt.hist(delta_mag, log=True)
    plt.xlabel('delta mag')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'delta_mags_zps.svg'), dpi=400)
    plt.clf()

# =============================================================================
# plot de deltas de magnitud sobre magnitud (goyet)
# =============================================================================
    plt.figure(figsize=(9,3))
    plt.subplot(141)
    dt_zps = store['dt_zps']
    dt_zps = dt_zps[dt_zps['MAG_APER']<30]
    dt_zps.dropna(axis='rows', inplace=True)
    delta_mag = dt_zps['sim_mag'] - dt_zps['MAG_APER']
    delta_over_mag = delta_mag/dt_zps['sim_mag']
    plt.hist(delta_over_mag, log=True)
    plt.xlabel('delta mag')

    plt.subplot(142)
    dt_zps = store['dt_ois']
    dt_zps = dt_zps[dt_zps['MAG_APER']<30]
    dt_zps.dropna(axis='rows', inplace=True)
    delta_mag = dt_zps['sim_mag'] - dt_zps['MAG_APER']
    delta_over_mag = delta_mag/dt_zps['sim_mag']
    plt.hist(delta_over_mag, log=True)
    plt.xlabel('delta mag')

    plt.subplot(143)
    dt_zps = store['dt_hot']
    dt_zps = dt_zps[dt_zps['MAG_APER']<30]
    dt_zps.dropna(axis='rows', inplace=True)
    delta_mag = dt_zps['sim_mag'] - dt_zps['MAG_APER']
    delta_over_mag = delta_mag/dt_zps['sim_mag']
    plt.hist(delta_over_mag, log=True)
    plt.xlabel('delta mag')

    plt.subplot(144)
    dt_zps = store['dt_sps']
    dt_zps['MAG_APER'] = -2.5*np.log10(dt_zps.cflux)
    dt_zps = dt_zps[dt_zps['MAG_APER']<30]
    dt_zps.dropna(axis='rows', inplace=True)
    delta_mag = dt_zps['sim_mag'] - dt_zps['MAG_APER']
    delta_over_mag = delta_mag/dt_zps['sim_mag']
    plt.hist(delta_over_mag, log=True)
    plt.xlabel('delta mag')

    plt.savefig(os.path.join(plot_dir, 'delta_over_mags_zps.svg'), dpi=400)
    plt.clf()

# =============================================================================
# plot de
# =============================================================================

    return


if __name__ == '__main__':
    import sys
    print(sys.argv)
    m1 = sys.argv[1]
    path = sys.argv[2]
    sys.exit(main(float(m1), path))
