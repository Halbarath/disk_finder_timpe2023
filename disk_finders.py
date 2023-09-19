#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-impact disk finder.

This module contains disk finding algorithms that can differentiate the central
planet from the disk and ejecta.

"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pynbody
from scipy.stats import binned_statistic
import subprocess
from sys import exit

import constants as C
from rotation import angular_momentum, angular_velocity, obliquity, rotation_period
from utils import cart2kep, orbital_velocity, weighted_percentiles
from figures import centering_check, FTVD_plot


def center_on_largest_remnant(coll, pids, rho_perc=50.0):
    """Centers snapshot on core of the largest remnant.

    Calculates the center of mass for all particles with PIDs in the provided
    list and with a density above the 'rho_cut' threshold. The
    threshold is determined by the percentile 'rho_perc', which should isolate
    only the dense core of the main fragment. This should work in the case of
    two large post-impact fragments (e.g., a hit-and-run).

    Args:
        snap (pynbody): Snapshot to be centered.
        pids (list): List of PIDs which will be used to calculate the CoM.
        rho_perc (float): Calculate center of mass using only particles with
                          densities above this percentile.

    Returns:
        pynbody: Centered snapshot.

    """

    LR = coll.load_copy()[pids]

    rho_cut = np.percentile(LR['rho'].in_units('g cm**-3'), rho_perc)

    LR = LR[LR['rho'].in_units('g cm**-3') > rho_cut]


    # [WARNING] Using the mean/average means that a single dense particle at a
    # large distance can significantly affect the COM position. Using the median
    # is a robust alternative.
    xcm = float(np.average(np.array(LR['pos'])[:,0],
                weights=LR['mass'].in_units('5.9722e24 kg')))
    ycm = float(np.average(np.array(LR['pos'])[:,1],
                weights=LR['mass'].in_units('5.9722e24 kg')))
    zcm = float(np.average(np.array(LR['pos'])[:,2],
                weights=LR['mass'].in_units('5.9722e24 kg')))

    vxcm = float(np.average(np.array(LR['vel'])[:,0],
                 weights=LR['mass'].in_units('5.9722e24 kg')))
    vycm = float(np.average(np.array(LR['vel'])[:,1],
                 weights=LR['mass'].in_units('5.9722e24 kg')))
    vzcm = float(np.average(np.array(LR['vel'])[:,2],
                 weights=LR['mass'].in_units('5.9722e24 kg')))

    del LR

    # Fix positions
    coll['pos'][:,0] -= xcm
    coll['pos'][:,1] -= ycm
    coll['pos'][:,2] -= zcm
    # Fix velocities
    coll['vel'][:,0] -= vxcm
    coll['vel'][:,1] -= vycm
    coll['vel'][:,2] -= vzcm


    # Second pass in case a few dense particles far out are throwing the mean
    xcm = float(np.average(np.array(coll[(coll['r'] < 5) & (coll['rho'].in_units('g cm**-3') > rho_cut)]['pos'])[:,0],
                weights=coll[(coll['r'] < 5) & (coll['rho'].in_units('g cm**-3') > rho_cut)]['mass'].in_units('5.9722e24 kg')))
    ycm = float(np.average(np.array(coll[(coll['r'] < 5) & (coll['rho'].in_units('g cm**-3') > rho_cut)]['pos'])[:,1],
                weights=coll[(coll['r'] < 5) & (coll['rho'].in_units('g cm**-3') > rho_cut)]['mass'].in_units('5.9722e24 kg')))
    zcm = float(np.average(np.array(coll[(coll['r'] < 5) & (coll['rho'].in_units('g cm**-3') > rho_cut)]['pos'])[:,2],
                weights=coll[(coll['r'] < 5) & (coll['rho'].in_units('g cm**-3') > rho_cut)]['mass'].in_units('5.9722e24 kg')))

    vxcm = float(np.average(np.array(coll[(coll['r'] < 5) & (coll['rho'].in_units('g cm**-3') > rho_cut)]['vel'])[:,0],
                 weights=coll[(coll['r'] < 5) & (coll['rho'].in_units('g cm**-3') > rho_cut)]['mass'].in_units('5.9722e24 kg')))
    vycm = float(np.average(np.array(coll[(coll['r'] < 5) & (coll['rho'].in_units('g cm**-3') > rho_cut)]['vel'])[:,1],
                 weights=coll[(coll['r'] < 5) & (coll['rho'].in_units('g cm**-3') > rho_cut)]['mass'].in_units('5.9722e24 kg')))
    vzcm = float(np.average(np.array(coll[(coll['r'] < 5) & (coll['rho'].in_units('g cm**-3') > rho_cut)]['vel'])[:,2],
                 weights=coll[(coll['r'] < 5) & (coll['rho'].in_units('g cm**-3') > rho_cut)]['mass'].in_units('5.9722e24 kg')))

    # Fix positions
    coll['pos'][:,0] -= xcm
    coll['pos'][:,1] -= ycm
    coll['pos'][:,2] -= zcm
    # Fix velocities
    coll['vel'][:,0] -= vxcm
    coll['vel'][:,1] -= vycm
    coll['vel'][:,2] -= vzcm

    # Return modified snapshot
    return coll


def FTVD_sliding_window(stdfile, pid_list, R_est_init, rho_perc,
                        dev_perc=0.05, wsize=0.1, rmin=0.1):
    """Radially bins the deviation from solid body rotation to find a radius.

    Args:
        stdfile (str): Path to post-impact Tipsy file.
        pids_list (dict): Lists of particle IDs belonging to remnants.
        R_est_init (float): Initial naive guess for planet radius (Earth radii).
        rho_perc (float): Percentile for density cut in centering subroutine.
        dev_perc (float): Maximum deviation from expected rotation rate.
        wsize (float): Sliding window size in Earth radii.
        rmin (float): Minimum radius when checking deviation (Earth radii).

    Returns:
        mids (1d array):  Midpoints of bins.
        means (1d array): Mean value of deltav in each bin.
        R_est (float):    Estimated radius of planet.
        incomplete_merger_flag (bool): Flag to indicate potential inc. merger.
    """
    # Output figure names
    fig_name = f"{str(os.path.basename(stdfile).split('.')[0])}"

    # Load snapshot and restrict to area around central planet
    s = pynbody.load(stdfile)

    pids_LR = pid_list['LR']

    # Center collision on largest remnant (to avoid centering problems with SLR)
    s = center_on_largest_remnant(s, pids_LR, rho_perc=rho_perc)

    # Number of radial bins depends on number of particles
    nbins = int(1e-2 * len(s))

    # Maximum radius in which to search for planet radius
    xmax = 5 * R_est_init

    # Restrict to only particles within this radius to boost performance
    s = s[s['rxy'] < xmax]


    # Generate bins from 0 to xmax (Earth radii).
    bins = np.linspace(rmin, xmax, num=nbins)

    binsize = bins[1] - bins[0]

    # Get midpoints of bins
    mids = []
    for i in range(1, len(bins)):
        mids.append((bins[i]+bins[i-1])/2)

    # Assign particles to bins
    s['rbin'] = np.array(pd.cut(x=s['rxy'], bins=bins, labels=mids,
                                include_lowest=True))

    rvals = []
    dvals = []

    # Initialize radius estimate
    R_est = 0

    empty_cnt = 0
    exceed_cnt = 0

    incomplete_merger_flag = False

    print('\tTraversing bins')

    # Traverse radial bins outward
    for r in sorted(mids):

        # Start of sliding window
        wmin = np.max([0.5*rmin, r-wsize])

        # Midpoint of sliding window
        wmid = (wmin + r) / 2

        # Get slice corresponding to sliding window
        swin = s[(s['rxy'] > wmin) & (s['rxy'] <= r)]

        # How many particles in the sliding window?
        nwin = len(swin)

        if nwin == 0:
            print('[ERROR] Sliding window is empty!')

        # Median rotation rate in sliding window (Hz)
        # This is the expected rotation rate in the current bin
        omega_exp = np.median(swin['vt'].in_units('km s**-1') / swin['rxy'].in_units('km'))

        # Delete window slice for performance
        del swin


        # Get slice corresponding to current bin
        sbin = s[s['rbin'] == r]

        # How many paticles in current bin?
        nbin = len(sbin)

        # If bin is empty, assume that current value of r is radius.
        if len(sbin) == 0:

            empty_cnt += 1

            if empty_cnt > 2:

                if r < R_est_init:
                    # Set flag
                    print("\t[WARNING] Merger may be incomplete!")
                    incomplete_merger_flag = True

                print('\tFound radius [EMPTY]')
                return rvals, dvals, r-(2*binsize), incomplete_merger_flag

            continue

        else:
            empty_cnt = 0


        # Expected transverse velocity for each particle in bin assuming solid-body rotation
        sbin['vt_exp'] = sbin['rxy'].in_units('km') * omega_exp

        # Fractional deviation from expected rotation rate for each particle in bin
        sbin['vt_dev'] = (sbin['vt'].in_units('km s**-1') - sbin['vt_exp']) / sbin['vt_exp']

        # Median fractional transverse velocity deviation in current bin
        ftvd = np.median(sbin['vt_dev'])

        rvals.append(r)
        dvals.append(ftvd)


        if r < rmin:

            R_est = r

        elif abs(ftvd) > dev_perc:

            exceed_cnt += 1

            if exceed_cnt > 2:

                if r < R_est_init:
                    # Set flag
                    print("\t[WARNING] Merger may be incomplete!")
                    incomplete_merger_flag = True

                print('\tFound radius [EXCEED]')

                return rvals, dvals, r, incomplete_merger_flag

        else:
            exceed_cnt = 0

        del sbin

    print("\t[WARNING] FTVD unable to estimate planet radius!")

    return rvals, dvals, np.nan, incomplete_merger_flag


def FTVD(stdfile, pid_list, SLR_bound, SLR_peri, rho_perc=25, dev_perc=0.05):
    """Disk finding algorithm based on deviation from solid body rotation.

    This function is an implementation of the fractional tranverse velocity
    deviation (FTVD) disk finder. To understand how it works in
    detail, see the 'FTVresults_finder.ipynb' Jupyter notebook at
    'moon-impact-survey/notebooks/'.

    Args:
        stdfile (str): Post-impact Tipsy file to run disk finder on.
        pid_list (dict): List of PIDs belonging to each remnant.
        SLR_bound (bool): Is SLR gravitationally bound to LR?
        SLR_peri (float): If bound, peridistance of SLR orbit in Earth radii.
        rho_perc (float): Density percentile that determines which particles are
                          used to determine the solid body rotation rate.
        dev_perc (float): Threshold for acceptable deviation from solid body
                          rotation.

    Returns:
        results (dict): Planet, disk, and ejecta properties.
        pids (dict): Particle ID lists for the planet, disk, and ejecta.

    """

    results = {}

    pids_LR = pid_list['LR']
    pids_SLR = pid_list['SLR']
    # The PIDS of any additional remnants are also available in pids

    # Load collision
    coll = pynbody.load(stdfile)

    # Determine density cutoff for initial guess
    rho_crit = np.percentile(coll['rho'].in_units('g cm**-3'), rho_perc)

    # Mass of initial conservative estimate
    M_est_init = np.sum(coll[coll['rho'].in_units('g cm**-3') > rho_crit]['mass'].in_units('5.9722e24 kg'))

    # Calculate an initial guess for the planet radius using the critcal density
    R_est_init = (3*M_est_init*C.MEARTH_G/(4*np.pi*rho_crit))**(1/3)/C.REARTH_CM

    del coll

    rmin = 0.2
    #rmin = 0.5 * R_est_init

    # Determine planet radius using sliding window
    bins, means, R_est, incomplete_merger_flag = FTVD_sliding_window(stdfile,
            pid_list, R_est_init, rho_perc, dev_perc=0.05, wsize=0.15, rmin=rmin)


    results['incomplete_merger_flag'] = incomplete_merger_flag

    print('\tSliding window algorithm completed')


    # Reload collision in case it was modified above
    coll = pynbody.load(stdfile)

    # Center collision on largest remnant
    coll = center_on_largest_remnant(coll, pids_LR, rho_perc=rho_perc)

    # Isolate central planet
    planet = coll[coll['r'].in_units('6.3710084e3 km') <= R_est]

    M_p = float(np.sum(planet['mass'].in_units('5.9722e24 kg')))

    # Rotation rate of planet
    omega_p = weighted_percentiles(planet['vt'].in_units('km s**-1') / planet['rxy'].in_units('km'), 50,
                                   sample_weight=planet['mass'].in_units('5.9722e24 kg'))

    del coll, planet


    print('\tDetermining Kepler intercept')
    # Find radius (R_max) at which vt_sb > vt_kepler
    for r in np.linspace(0.01, 10, num=1000):

        R_max = r

        y_sb = C.REARTH_KM * r * omega_p

        y_kep = orbital_velocity(M_p*C.MEARTH_KG, r*C.REARTH_M)

        if y_sb >= y_kep:
            break

    if R_est >= R_max:
        R_est = R_max


    # Plot results of FTVD disk finder
    # Should add pids_SLR to mark particles belonging to SLR...
    print('\tCreating FTVD plots')

    FTVD_plot(stdfile, omega_p, M_p, R_est, R_est_init, rmin, dev_perc,
              bins, means, pids_LR, xlim=5)


    # How close are we to the Hot Spin Stability Limit (HSSL)?
    HSSL = R_est / R_max

    # Compute orbits and separate disk from ejecta
    pids = {"planet":[], "disk":[], "ejecta":[]}


    coll = pynbody.load(stdfile)

    coll = center_on_largest_remnant(coll, pids_LR, rho_perc=rho_perc)

    print('\tStarting periapsis checks')

    with open(f"{stdfile}.disk", 'w') as out:

        print(f"{len(coll)} {len(coll)} 0", file=out)

        for idx in range(0, len(coll)):

            # Treat SLR and other remnants as super-particles
            if idx in pids_SLR:

                if SLR_bound:
                    if SLR_peri <= R_est:
                        pids['planet'].append(idx)
                        print("0", file=out)
                    else:
                        pids['disk'].append(idx)
                        print("1", file=out)

                else:
                    pids['ejecta'].append(idx)
                    print("2", file=out)

                continue


            r_i   = coll['r'][idx]
            rxy_i = coll['r'][idx]

            if r_i < R_est:
                pids['planet'].append(idx)
                print("0", file=out)

            else:

                m_i = coll['mass'].in_units('kg')[idx]

                pos = np.array(coll['pos'].in_units('m')[idx])

                vel = np.array(coll['vel'].in_units('m s**-1')[idx])

                # Compute orbits
                a, e, i, Omega, omega = cart2kep(pos, vel, m_i, M_p*C.MEARTH_KG)

                # Calculate periastron distance
                r_peri = a * (1.0 - e) / C.REARTH_M


                if e >= 1.0:
                    pids['ejecta'].append(idx)
                    print("2", file=out)
                elif r_peri <= R_est:
                    pids['planet'].append(idx)
                    print("0", file=out)
                else:
                    pids['disk'].append(idx)
                    print("1", file=out)

    print('\tPeriapsis checks complete')

    # Planet
    if len(pids['planet']) > 0:
        planet = coll[pids['planet']]
        results['N_p'] = len(planet)
        results['M_p'] = float(np.sum(planet['mass'].in_units('5.9722e24 kg'))) # Earth masses
        results['Fe_p'] = float(planet[planet['metals'] == 55]['mass'].in_units('5.9722e24 kg').sum() / planet['mass'].in_units('5.9722e24 kg').sum())
        results['R_p'] = R_est  # Earth radii
        results['J_p'] = angular_momentum(planet)
        #results['omega'] = np.median(planet['vt'].in_units('km s**-1') / planet['rxy'].in_units('km'))
        results['omega'] = angular_velocity(planet)
        results['P_rot'] = rotation_period(results['omega'])
        results['theta_p'] = obliquity(planet)
        results['omega_sb'] = np.nan #omega_sb
        results['P_rot_sb'] = np.nan #P_rot_sb
        results['HSSL'] = HSSL
        del planet
    else:
        results['N_p'] = 0
        results['M_p'] = 0
        results['Fe_p'] = np.nan
        results['R_p'] = 0
        results['J_p'] = 0
        results['omega'] = np.nan
        results['P_rot'] = np.nan
        results['omega_sb'] = np.nan
        results['P_rot_sb'] = np.nan
        results['HSSL'] = np.nan


    if len(pids['disk']) > 0:
        disk = coll[pids['disk']]
        results['N_disk'] = len(disk)
        results['M_disk'] = float(np.sum(disk['mass'].in_units('5.9722e24 kg')))  # Earth masses
        results['Fe_disk'] = float(disk[disk['metals'] == 55]['mass'].in_units('5.9722e24 kg').sum() / disk['mass'].in_units('5.9722e24 kg').sum())
        results['J_disk'] = angular_momentum(disk)  # J-s
        del disk
    else:
        results['N_disk'] = 0
        results['M_disk'] = 0
        results['Fe_disk'] = np.nan
        results['J_disk'] = 0


    if len(pids['ejecta']) > 0:
        ejecta = coll[pids['ejecta']]
        results['N_ej'] = len(ejecta)
        results['M_ej'] = float(np.sum(ejecta['mass'].in_units('5.9722e24 kg')))  # Earth masses
        results['Fe_ej'] = float(ejecta[ejecta['metals'] == 55]['mass'].in_units('5.9722e24 kg').sum() / ejecta['mass'].in_units('5.9722e24 kg').sum())
        results['J_ej'] = angular_momentum(ejecta)  # J-s
        del ejecta
    else:
        results['N_ej'] = 0
        results['M_ej'] = 0
        results['Fe_ej'] = np.nan
        results['J_ej'] = 0


    # Bound material
    if len(pids['planet']) > 0:
        bound_pids = pids['planet'] + pids['disk']
        bound = coll[bound_pids]
        results['N_bound'] = len(bound)
        results['M_bound'] = float(np.sum(bound['mass'].in_units('5.9722e24 kg')))
        results['Fe_bound'] = float(bound[bound['metals'] == 55]['mass'].in_units('5.9722e24 kg').sum() / bound['mass'].in_units('5.9722e24 kg').sum())
        results['J_bound'] = angular_momentum(bound)  # J-s
        del bound
    else:
        results['N_bound'] = 0
        results['M_bound'] = 0
        results['J_bound'] = 0
        results['Fe_bound'] = np.nan


    del coll

    # Disk finder algo parameters
    results['rho_perc'] = rho_perc
    results['rho_crit'] = rho_crit
    results['dev_perc'] = dev_perc
    #results['dev_crit'] = dev_crit
    results['R_est_init'] = R_est_init

    return results, pids
