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
    import ipdb; ipdb.set_trace()
    simulations = simulations[simulations.m1_diam==m1_diam]

    simulated = pd.merge(left=simulations, right=simulated,
                             right_on='simulation_id', left_on='id', how='left')

    plt.figure(figsize=(6,3))
    plt.hist(simulated['app_mag'], cumulative=False, bins=25, log=True)
    plt.xlabel(r'$mag$', fontsize=16)
    plt.tick_params(labelsize=15)
    plt.ylabel(r'$N(m) dm$', fontsize=16)
    #plt.ylabel(r'$\int_{-\infty}^{mag}\phi(m\prime)dm\prime$', fontsize=16)
    plt.savefig(os.path.join(plot_dir, 'lum_fun_simulated.svg'), dpi=400)

    return


if __name__ == '__main__':
    import sys
    print(sys.argv)
    m1 = sys.argv[1]
    path = sys.argv[2]
    sys.exit(main(float(m1), path))
