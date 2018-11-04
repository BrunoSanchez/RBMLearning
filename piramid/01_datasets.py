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
import gc
from sqlalchemy import create_engine

import numpy as np
import seaborn as sns
import pandas as pd

from astropy.stats import sigma_clipped_stats

import custom_funs as cf

storefile = '/mnt/clemente/bos0109/table_store.h5'

store = pd.HDFStore(storefile)
store.open()

plot_dir = os.path.abspath('./plots/.')
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

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
    del(simulated)
    #estas son las simulaciones programadas
    try:
        simulations = store['simulations']
    except:
        simulations = pd.read_sql_query("""SELECT * FROM "Simulation" """, engine)
        simulations = cf.optimize_df(simulations)
        store['simulations'] = simulations
        store.flush()
    del(simulations)
    gc.collect()
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
    del(und_z)
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
    del(und_s)
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
    del(und_sc)
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
    del(und_b)
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
    del(und_h)
    gc.collect()
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
                                            SI.m1_diam as m1_diam,
                                            SI.m2_diam as m2_diam,
                                            SI.executed as executed,
                                            SI.id as id_simulation,
                                            SI.ref_starzp as ref_starzp,
                                            SI.ref_starslope as ref_starslope,
                                            SI.ref_fwhm as ref_fwhm,
                                            SI.new_fwhm as new_fwhm,
                                            SI.eff_col as eff_col,
                                            SI.px_scale as px_scale,
                                            SI.ref_back_sbright as ref_back_sbright,
                                            SI.new_back_sbright as new_back_sbright,
                                            SI.exp_time as exp_time
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
        dt_zps.executed = dt_zps.executed.astype('bool').astype(int)
        dt_zps.IS_REAL = dt_zps.IS_REAL.astype('bool').astype(int)
        dt_zps.drop_duplicates(inplace=True)

        dt_zps['VALID_MAG'] = dt_zps['MAG_APER']<30
        dt_zps['mag_offset'] = dt_zps['sim_mag'] - dt_zps['MAG_APER']
        cals = cf.cal_mags(dt_zps)
        dt_zps = pd.merge(dt_zps, cals, on='image_id', how='left')
        dt_zps['mag'] = dt_zps['MAG_APER'] * dt_zps['slope'] + dt_zps['mean_offset']

        dt_zps['VALID_MAG_iso'] = dt_zps['MAG_ISO']<30
        dt_zps['mag_offset_iso'] = dt_zps['sim_mag'] - dt_zps['MAG_ISO']
        cals = cf.cal_mags_iso(dt_zps)
        dt_zps = pd.merge(dt_zps, cals, on='image_id', how='left')
        dt_zps['mag_iso'] = dt_zps['MAG_ISO'] * dt_zps['slope_iso'] + \
                            dt_zps['mean_offset_iso']

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

        dt_zps = cf.optimize_df(dt_zps)
        store['dt_zps'] = dt_zps
        store.flush()
    del(dt_zps)
    gc.collect()
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
                                            SI.m1_diam as m1_diam,
                                            SI.m2_diam as m2_diam,
                                            SI.executed as executed,
                                            SI.id as id_simulation,
                                            SI.ref_starzp as ref_starzp,
                                            SI.ref_starslope as ref_starslope,
                                            SI.ref_fwhm as ref_fwhm,
                                            SI.new_fwhm as new_fwhm,
                                            SI.eff_col as eff_col,
                                            SI.px_scale as px_scale,
                                            SI.ref_back_sbright as ref_back_sbright,
                                            SI.new_back_sbright as new_back_sbright,
                                            SI.exp_time as exp_time
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
        dt_sps.executed = dt_sps.executed.astype('bool').astype(int)
        dt_sps.IS_REAL = dt_sps.IS_REAL.astype('bool').astype(int)
        dt_sps.drop_duplicates(inplace=True)

        dt_sps['MAG_APER'] = -2.5*np.log10(dt_sps['cflux'])
        dt_sps['MAG_ISO'] = -2.5*np.log10(dt_sps['cflux'])

        dt_sps['VALID_MAG'] = dt_sps['MAG_APER']<30
        dt_sps['mag_offset'] = dt_sps['sim_mag'] - dt_sps['MAG_APER']
        cals = cf.cal_mags(dt_sps)
        dt_sps = pd.merge(dt_sps, cals, on='image_id', how='left')
        dt_sps['mag'] = dt_sps['MAG_APER'] * dt_sps['slope'] + dt_sps['mean_offset']

        dt_sps['VALID_MAG_iso'] = dt_sps['MAG_ISO']<30
        dt_sps['mag_offset_iso'] = dt_sps['sim_mag'] - dt_sps['MAG_ISO']
        dt_sps['mag_iso'] = dt_sps['mag']

        dt_sps['goyet'] = np.abs(dt_sps['sim_mag'] - dt_sps['mag'])/dt_sps['sim_mag']
        dt_sps['goyet_iso'] = dt_sps['goyet']

        grouped = dt_sps.dropna().groupby(['image_id'], sort=False)
        dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet'])[0])
        dd.name = 'mean_goyet'
        dt_sps = pd.merge(dt_sps, dd.to_frame(), on='image_id', how='left')
        dt_sps['mean_goyet_iso'] = dt_sps['mean_goyet']

        dt_sps = cf.optimize_df(dt_sps)
        store['dt_sps'] = dt_sps
        store.flush()
    del(dt_sps)
    gc.collect()
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
                                            SI.m1_diam as m1_diam,
                                            SI.m2_diam as m2_diam,
                                            SI.executed as executed,
                                            SI.id as id_simulation,
                                            SI.ref_starzp as ref_starzp,
                                            SI.ref_starslope as ref_starslope,
                                            SI.ref_fwhm as ref_fwhm,
                                            SI.new_fwhm as new_fwhm,
                                            SI.eff_col as eff_col,
                                            SI.px_scale as px_scale,
                                            SI.ref_back_sbright as ref_back_sbright,
                                            SI.new_back_sbright as new_back_sbright,
                                            SI.exp_time as exp_time
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
        dt_ois.executed = dt_ois.executed.astype('bool').astype(int)
        dt_ois.IS_REAL = dt_ois.IS_REAL.astype('bool').astype(int)
        dt_ois.drop_duplicates(inplace=True)

        dt_ois['VALID_MAG'] = dt_ois['MAG_APER']<30
        dt_ois['mag_offset'] = dt_ois['sim_mag'] - dt_ois['MAG_APER']
        cals = cf.cal_mags(dt_ois)
        dt_ois = pd.merge(dt_ois, cals, on='image_id', how='left')
        dt_ois['mag'] = dt_ois['MAG_APER'] * dt_ois['slope'] + dt_ois['mean_offset']

        dt_ois['VALID_MAG_iso'] = dt_ois['MAG_ISO']<30
        dt_ois['mag_offset_iso'] = dt_ois['sim_mag'] - dt_ois['MAG_ISO']
        cals = cf.cal_mags_iso(dt_ois)
        dt_ois = pd.merge(dt_ois, cals, on='image_id', how='left')
        dt_ois['mag_iso'] = dt_ois['MAG_ISO'] * dt_ois['slope_iso'] + \
                            dt_ois['mean_offset_iso']

        dt_ois['goyet'] = np.abs(dt_ois['sim_mag'] - dt_ois['mag'])/dt_ois['sim_mag']
        grouped = dt_ois.dropna().groupby(['image_id'], sort=False)
        dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet'])[0])
        dd.name = 'mean_goyet'
        dt_ois = pd.merge(dt_ois, dd.to_frame(), on='image_id', how='left')

        dt_ois['goyet_iso'] = np.abs(dt_ois['sim_mag'] - dt_ois['mag_iso'])/dt_ois['sim_mag']
        grouped = dt_ois.dropna().groupby(['image_id'], sort=False)
        dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet_iso'])[0])
        dd.name = 'mean_goyet_iso'
        dt_ois = pd.merge(dt_ois, dd.to_frame(), on='image_id', how='left')

        dt_ois = cf.optimize_df(dt_ois)
        store['dt_ois'] = dt_ois
        store.flush()
    del(dt_ois)
    gc.collect()
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
                                            SI.m1_diam as m1_diam,
                                            SI.m2_diam as m2_diam,
                                            SI.executed as executed,
                                            SI.id as id_simulation,
                                            SI.ref_starzp as ref_starzp,
                                            SI.ref_starslope as ref_starslope,
                                            SI.ref_fwhm as ref_fwhm,
                                            SI.new_fwhm as new_fwhm,
                                            SI.eff_col as eff_col,
                                            SI.px_scale as px_scale,
                                            SI.ref_back_sbright as ref_back_sbright,
                                            SI.new_back_sbright as new_back_sbright,
                                            SI.exp_time as exp_time
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
        dt_hot.executed = dt_hot.executed.astype('bool').astype(int)
        dt_hot.IS_REAL = dt_hot.IS_REAL.astype('bool').astype(int)
        dt_hot.drop_duplicates(inplace=True)

        dt_hot['VALID_MAG'] = dt_hot['MAG_APER']<30
        dt_hot['mag_offset'] = dt_hot['sim_mag'] - dt_hot['MAG_APER']
        cals = cf.cal_mags(dt_hot)
        dt_hot = pd.merge(dt_hot, cals, on='image_id', how='left')
        dt_hot['mag'] = dt_hot['MAG_APER'] * dt_hot['slope'] + dt_hot['mean_offset']

        dt_hot['VALID_MAG_iso'] = dt_hot['MAG_ISO']<30
        dt_hot['mag_offset_iso'] = dt_hot['sim_mag'] - dt_hot['MAG_ISO']
        cals = cf.cal_mags_iso(dt_hot)
        dt_hot = pd.merge(dt_hot, cals, on='image_id', how='left')
        dt_hot['mag_iso'] = dt_hot['MAG_ISO'] * dt_hot['slope_iso'] + \
                            dt_hot['mean_offset_iso']

        dt_hot['goyet'] = np.abs(dt_hot['sim_mag'] - dt_hot['mag'])/dt_hot['sim_mag']
        grouped = dt_hot.dropna().groupby(['image_id'], sort=False)
        dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet'])[0])
        dd.name = 'mean_goyet'
        dt_hot = pd.merge(dt_hot, dd.to_frame(), on='image_id', how='left')

        dt_hot['goyet_iso'] = np.abs(dt_hot['sim_mag'] - dt_hot['mag_iso'])/dt_hot['sim_mag']
        grouped = dt_hot.dropna().groupby(['image_id'], sort=False)
        dd = grouped.apply(lambda df: sigma_clipped_stats(df['goyet_iso'])[0])
        dd.name = 'mean_goyet_iso'
        dt_hot = pd.merge(dt_hot, dd.to_frame(), on='image_id', how='left')

        dt_hot = cf.optimize_df(dt_hot)
        store['dt_hot'] = dt_hot
        store.flush()
    del(dt_hot)
    gc.collect()
    store.close()



if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
