#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  01_datasets.py
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

plot_dir = os.path.abspath('./plots/.')
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

#sns.set_context(font_scale=16)
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams['text.usetex'] = True

#engine = create_engine('sqlite:///../RBpipeline/newrbogus22-dev.db')
CONNECTION = 'postgresql://jarvis:Bessel0@172.18.122.4:5432/resimulation_docker'
#CONNECTION = 'postgresql://jarvis:Bessel0@toritos:5432/resimu_docker'
engine = create_engine(CONNECTION)

def main(argv):

    #estas son las fuentes simuladas
    try:
        simulated = store['simulated']
    except:
        simulated = pd.read_sql_query("""SELECT * FROM "Simulated" """, engine)
        simulated = cf.optimize_df(simulated)
        store['simulated'] = simulated
        store.flush()

    #estas son las simulaciones programadas
    try:
        simulations = store['simulations']
    except:
        simulations = pd.read_sql_query("""SELECT * FROM "Simulation" """, engine)
        simulations = cf.optimize_df(simulations)
        store['simulations'] = simulations
        store.flush()

    try:
        und_z = store['und_z']
    except:
        und_z = pd.read_sql_query(""" SELECT S.x, S.y, S.app_mag, S.image_id, U.simulated_id
                                 FROM "Simulated" S INNER JOIN "Undetected" U
                                 ON S.id=U.simulated_id """,
                              engine)
        und_z = cf.optimize_df(und_z)
        und_z.drop_duplicates(inplace=True)
        store['und_z'] = und_z
        store.flush()

    try:
        und_s = store['und_s']
    except:
        und_s = pd.read_sql_query("""SELECT S.x, S.y, S.app_mag, S.image_id, U.simulated_id
                                 FROM "Simulated" S INNER JOIN "SUndetected" U
                                 ON S.id=U.simulated_id""",
                              engine)
        und_s = cf.optimize_df(und_s)
        und_s.drop_duplicates(inplace=True)
        store['und_s'] = und_s
        store.flush()

    try:
        und_sc = store['und_sc']
    except:
        und_sc = pd.read_sql_query("""SELECT S.x, S.y, S.app_mag, S.image_id, U.simulated_id
                                 FROM "Simulated" S INNER JOIN "SCorrUndetected" U
                                 ON S.id=U.simulated_id""",
                              engine)
        und_sc = cf.optimize_df(und_sc)
        und_sc.drop_duplicates(inplace=True)
        store['und_sc'] = und_sc
        store.flush()

    try:
        und_b = store['und_b']
    except:
        und_b = pd.read_sql_query("""SELECT S.x, S.y, S.app_mag, S.image_id, U.simulated_id
                                 FROM "Simulated" S INNER JOIN "UndetectedOIS" U
                                 ON S.id=U.simulated_id""",
                              engine)
        und_b = cf.optimize_df(und_b)
        und_b.drop_duplicates(inplace=True)
        store['und_b'] = und_b
        store.flush()

    try:
        und_h = store['und_h']
    except:
        und_h = pd.read_sql_query("""SELECT S.x, S.y, S.app_mag, S.image_id, U.simulated_id
                                 FROM "Simulated" S INNER JOIN "UndetectedHOT" U
                                 ON S.id=U.simulated_id""",
                              engine)
        und_h = cf.optimize_df(und_h)
        und_h.drop_duplicates(inplace=True)
        store['und_h'] = und_h
        store.flush()

    try:
        dt_zps = store['dt_zps']
    except:
        dt_zps = pd.merge(pd.read_sql_table('Detected', engine),
                      pd.read_sql_query("""SELECT
                                            D.id,
                                            S.app_mag as sim_mag,
                                            S.r_scales as r_scales,
                                            S.gx_mag as gx_mag,
                                            S.id as sim_id,
                                            SI.m1_diam as m1_diam
                                        FROM "Detected" D
                                            LEFT JOIN "Images" I
                                                ON D.image_id=I.id
                                            LEFT JOIN "Reals" R
                                                ON D.id=R.detected_id
                                            LEFT JOIN "Simulated" S
                                                ON S.id=R.simulated_id
                                            LEFT JOIN "Simulation" SI
                                                ON SI.id=I.simulation_id
                                                """, engine),
                                          on='id', suffixes=('',''))
        dt_zps.IS_REAL = dt_zps.IS_REAL.astype('bool').astype(int)
        dt_zps.drop_duplicates(inplace=True)
        dt_zps = cf.optimize_df(dt_zps)
        store['dt_zps'] = dt_zps
        store.flush()

    try:
        dt_sps = store['dt_sps']
    except:
        dt_sps = pd.merge(pd.read_sql_table('SDetected', engine),
                      pd.read_sql_query("""SELECT
                                            D.id,
                                            S.app_mag as sim_mag,
                                            S.r_scales as r_scales,
                                            S.gx_mag as gx_mag,
                                            S.id as sim_id,
                                            SI.m1_diam as m1_diam
                                        FROM "SDetected" D
                                            LEFT JOIN "SImages" I
                                                ON D.image_id=I.id
                                            LEFT JOIN "SReals" R
                                                ON D.id=R.detected_id
                                            LEFT JOIN "Simulated" S
                                                ON S.id=R.simulated_id
                                            LEFT JOIN "Simulation" SI
                                                ON SI.id=I.simulation_id""", engine),
                                          on='id', suffixes=('',''))
        dt_sps.IS_REAL = dt_sps.IS_REAL.astype('bool').astype(int)
        dt_sps.drop_duplicates(inplace=True)
        dt_sps = cf.optimize_df(dt_sps)
        store['dt_sps'] = dt_sps
        store.flush()

    try:
        dt_ois = store['dt_ois']
    except:
        dt_ois = pd.merge(pd.read_sql_table('DetectedOIS', engine),
                      pd.read_sql_query("""SELECT
                                            D.id,
                                            S.app_mag as sim_mag,
                                            S.r_scales as r_scales,
                                            S.gx_mag as gx_mag,
                                            S.id as sim_id,
                                            SI.m1_diam
                                        FROM "DetectedOIS" D
                                            LEFT JOIN "ImagesOIS" I
                                                ON D.image_id=I.id
                                            LEFT JOIN "RealsOIS" R
                                                ON D.id=R.detected_id
                                            LEFT JOIN "Simulated" S
                                                ON S.id=R.simulated_id
                                            LEFT JOIN "Simulation" SI
                                                ON SI.id=I.simulation_id""", engine),
                                          on='id', suffixes=('',''))
        dt_ois.IS_REAL = dt_ois.IS_REAL.astype('bool').astype(int)
        dt_ois.drop_duplicates(inplace=True)
        dt_ois = cf.optimize_df(dt_ois)
        store['dt_ois'] = dt_ois
        store.flush()

    try:
        dt_hot = store['dt_hot']
    except:
        dt_hot = pd.merge(pd.read_sql_table('DetectedHOT', engine),
                      pd.read_sql_query("""SELECT
                                            D.id,
                                            S.app_mag as sim_mag,
                                            S.r_scales as r_scales,
                                            S.gx_mag as gx_mag,
                                            S.id as sim_id,
                                            SI.m1_diam
                                        FROM "DetectedHOT" D
                                            LEFT JOIN "ImagesHOT" I
                                                ON D.image_id=I.id
                                            LEFT JOIN "RealsHOT" R
                                                ON D.id=R.detected_id
                                            LEFT JOIN "Simulated" S
                                                ON S.id=R.simulated_id
                                            LEFT JOIN "Simulation" SI
                                                ON SI.id=I.simulation_id""", engine),
                                          on='id', suffixes=('',''))
        dt_hot.IS_REAL = dt_hot.IS_REAL.astype('bool').astype(int)
        dt_hot.drop_duplicates(inplace=True)
        dt_hot = cf.optimize_df(dt_hot)
        store['dt_hot'] = dt_hot
        store.flush()


    store.close()



if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
