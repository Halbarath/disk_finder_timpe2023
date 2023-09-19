#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains functions related to rotational properties.

This module contains the various schema used to calculate properties related to
the rotation of various bodies in the system.

Todo:
    * None

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pynbody

import constants as C
import disk_finders
from utils import orbital_velocity


def centering_check(s, pid_list, pdf_name):
    """Snapshot of the post-impact state."""

    LR = s[pid_list['LR']]

    if len(pid_list['SLR']) > 0:
        SLR = s[pid_list['SLR']]
    else:
        SLR = None


    # Get name of collision
    name = os.path.basename(os.getcwd())

    # Matplotlib plot parameters
    plt.rc('axes', linewidth=2)

    a4_inches = (8.268,11.693)

    fs_axlab = 15

    fontsize = 12
    fontweight = 'normal'


    # Instantiate figure
    fig, axes = plt.subplots(1, 1, figsize=(9,9))

    ax = axes

    ax.set_aspect('equal')

    blim = 5

    ax.set_xlim(-blim, blim)
    ax.set_ylim(-blim, blim)

    # LR
    x_LR = LR['x'].in_units('6.3710084e3 km')  # Earth radii
    y_LR = LR['y'].in_units('6.3710084e3 km')  # Earth radii
    c_LR = 'red' #ejecta['rho'].in_units('g cm**-3')

    ax.scatter(x_LR, y_LR, marker='.', s=5, c=c_LR, alpha=0.5,
               zorder=3, rasterized=True)

    # SLR
    if SLR:
        x_SLR = SLR['x'].in_units('6.3710084e3 km')  # Earth radii
        y_SLR = SLR['y'].in_units('6.3710084e3 km')  # Earth radii
        c_SLR = 'blue' #ejecta['rho'].in_units('g cm**-3')

        ax.scatter(x_SLR, y_SLR, marker='.', s=5, c=c_SLR, alpha=0.5,
                   zorder=5, rasterized=True)
    else:
        print('\tNo SLR present')

    ax.axvline(x=0, color='black', zorder=9)
    ax.axhline(y=0, color='black', zorder=9)

    # Finishing touches
    fig.suptitle(name, fontsize=12)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight(fontweight)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight(fontweight)


    plt.tight_layout()

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(f"{pdf_name}_center.pdf", format='pdf', dpi=72)

    del fig, axes


def post_impact_snapshot(stdfile, pid_frag, pid_ftvd, R_est,
                         rho_perc=25, blim=10):
    """Snapshot of the post-impact state."""

    # Get name of collision
    name = os.path.basename(os.getcwd())

    fig_name = str(os.path.basename(stdfile).split('.')[0])

    # Matplotlib plot parameters
    plt.rc('axes', linewidth=2)

    a4_inches = (8.268,11.693)

    fs_axlab = 15

    fontsize = 12
    fontweight = 'normal'

    c_roche = 'red'
    c_hill = '#333333'
    c_rest = 'cyan'

    # Earth-Sun Hill sphere radius
    R_Hill = 235.2  # Earth radii

    # Earth-Moon Roche limit
    R_rigid = 1.49  # Earth radii
    R_fluid = 2.88  # Earth radii

    # Lunar semi-major axis
    R_Moon = 60.4  # Earth radii


    if blim == 3:
        ticks = [-2, -1, 0, 1, 2]
    elif blim == 5:
        ticks = [-4, -2, 0, 2, 4]
    elif blim == 10:
        ticks = [-7.5, -5, -2.5, 0, 2.5, 5, 7.5]
    elif blim == 15:
        ticks = [-10, -5, 0, 5, 10]
    elif blim == 25:
        ticks = [-20, -10, 0, 10, 20]
    elif blim == 50:
        ticks = [-40, -20, 0, 20, 40]
    else:
        ticks = [-200,-100,0,100,200]


    # Load snapshot
    s = pynbody.load(stdfile)

    s = disk_finders.center_on_largest_remnant(s, pid_frag['LR'],
                                               rho_perc=rho_perc)

    if len(pid_ftvd['planet']) > 0:
        planet = s[pid_ftvd['planet']]
    else:
        planet = None

    if len(pid_ftvd['disk']) > 0:
        disk = s[pid_ftvd['disk']]
    else:
        disk = None

    if len(pid_ftvd['ejecta']) > 0:
        ejecta = s[pid_ftvd['ejecta']]
    else:
        ejecta = None


    # Instantiate figure
    fig, axes = plt.subplots(2, 2, figsize=(8.9,9), sharex=True, sharey=True)

    # TOP LEFT PANEL
    ax = axes[0][0]

    ax.set_aspect('equal')

    ax.set_xlim(-blim, blim)
    ax.set_ylim(-blim, blim)

    ax.set_ylabel(r"$\hat{y}$ $(R_{\oplus})$", fontsize=fs_axlab)

    ax.set_yticks(ticks)

    ax.get_xaxis().set_visible(False)

    x = s['x'].in_units('6.3710084e3 km')  # Earth radii
    y = s['y'].in_units('6.3710084e3 km')  # Earth radii
    c = s['rho'].in_units('g cm**-3')

    ax.scatter(x, y, marker='.', s=5, c=c, alpha=0.1,
               vmin=0, vmax=20,
               zorder=1, rasterized=True)


    # Fluid Roche limit
    Roche = plt.Circle((0.0, 0.0), R_fluid,
                       facecolor='none', edgecolor=c_roche,
                       linewidth=1, linestyle='--', zorder=9)

    _ = ax.add_artist(Roche)

    if blim > 150:
        # Hill sphere
        Hill = plt.Circle((0.0, 0.0), R_Hill,
                          facecolor='none', edgecolor=c_hill,
                          linewidth=1, linestyle='--', zorder=9)

        _ = ax.add_artist(Hill)


    # BOTTOM LEFT PANEL
    ax = axes[1][0]

    ax.set_aspect('equal')

    ax.set_xlim(-blim, blim)
    ax.set_ylim(-blim, blim)

    ax.set_xlabel(r"$\hat{x}$ $(R_{\oplus})$", fontsize=fs_axlab)
    ax.set_ylabel(r"$\hat{z}$ $(R_{\oplus})$", fontsize=fs_axlab)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    x = s['x'].in_units('6.3710084e3 km')  # Earth radii
    y = s['z'].in_units('6.3710084e3 km')  # Earth radii
    c = s['rho'].in_units('g cm**-3')

    ax.scatter(x, y, marker='.', s=5, c=c, alpha=0.1,
               vmin=0, vmax=20,
               zorder=1, rasterized=True)

    # Fluid Roche limit
    Roche = plt.Circle((0.0, 0.0), R_fluid,
                       facecolor='none', edgecolor=c_roche,
                       linewidth=1, linestyle='--', zorder=9)

    _ = ax.add_artist(Roche)

    if blim > 150:
        # Hill sphere
        Hill = plt.Circle((0.0, 0.0), R_Hill,
                          facecolor='none', edgecolor=c_hill,
                          linewidth=1, linestyle='--', zorder=9)

        circ = ax.add_artist(Hill)


    # TOP RIGHT PANEL
    ax = axes[0][1]

    ax.set_aspect('equal')

    ax.set_xlim(-blim, blim)
    ax.set_ylim(-blim, blim)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Planet
    if planet:
        x_p = planet['x'].in_units('6.3710084e3 km')  # Earth radii
        y_p = planet['y'].in_units('6.3710084e3 km')  # Earth radii
        c_p = 'black' #planet['rho'].in_units('g cm**-3')

        ax.scatter(x_p, y_p, marker='.', s=5, c=c_p, alpha=0.1,
                   zorder=1, rasterized=True)

    # Disk
    if disk:
        x_d = disk['x'].in_units('6.3710084e3 km')  # Earth radii
        y_d = disk['y'].in_units('6.3710084e3 km')  # Earth radii
        c_d = 'blue' #disk['rho'].in_units('g cm**-3')

        ax.scatter(x_d, y_d, marker='.', s=5, c=c_d, alpha=0.25,
                   zorder=2, rasterized=True)

    # Ejecta
    if ejecta:
        x_ej = ejecta['x'].in_units('6.3710084e3 km')  # Earth radii
        y_ej = ejecta['y'].in_units('6.3710084e3 km')  # Earth radii
        c_ej = 'red' #ejecta['rho'].in_units('g cm**-3')

        ax.scatter(x_ej, y_ej, marker='.', s=5, c=c_ej, alpha=1,
                   zorder=3, rasterized=True)


    # Radius estimated by FTVD disk finder
    radius = plt.Circle((0.0, 0.0), R_est,
                        facecolor='none', edgecolor=c_rest,
                        linewidth=1, linestyle='--', zorder=9)

    _ = ax.add_artist(radius)


    # BOTTOM RIGHT PANEL
    ax = axes[1][1]

    ax.set_aspect('equal')

    ax.set_xlim(-blim, blim)
    ax.set_ylim(-blim, blim)

    ax.set_xlabel(r"$\hat{x}$ $(R_{\oplus})$", fontsize=fs_axlab)
    #ax.set_ylabel(r"$\hat{z}$ $(R_{\oplus})$", fontsize=fs_axlab)

    ax.get_yaxis().set_visible(False)

    ax.set_xticks(ticks)

    # Planet
    if planet:
        x_p = planet['x'].in_units('6.3710084e3 km')  # Earth radii
        y_p = planet['z'].in_units('6.3710084e3 km')  # Earth radii
        c_p = 'black' #planet['rho'].in_units('g cm**-3')

        ax.scatter(x_p, y_p, marker='.', s=5, c=c_p, alpha=0.1,
                   zorder=1, rasterized=True)

    # Disk
    if disk:
        x_d = disk['x'].in_units('6.3710084e3 km')  # Earth radii
        y_d = disk['z'].in_units('6.3710084e3 km')  # Earth radii
        c_d = 'blue' #disk['rho'].in_units('g cm**-3')

        ax.scatter(x_d, y_d, marker='.', s=5, c=c_d, alpha=0.1,
                   zorder=2, rasterized=True)

    # Ejecta
    if ejecta:
        x_ej = ejecta['x'].in_units('6.3710084e3 km')  # Earth radii
        y_ej = ejecta['z'].in_units('6.3710084e3 km')  # Earth radii
        c_ej = 'red' #ejecta['rho'].in_units('g cm**-3')

        ax.scatter(x_ej, y_ej, marker='.', s=5, c=c_ej, alpha=1,
                   #vmin=0, vmax=20,
                   zorder=3, rasterized=True)

    # Radius estimated by FTVD disk finder
    radius = plt.Circle((0.0, 0.0), R_est,
                        facecolor='none', edgecolor=c_rest,
                        linewidth=1, linestyle='--', zorder=9)

    _ = ax.add_artist(radius)


    # Finishing touches
    fig.suptitle(name, fontsize=12)

    for ax in axes.flatten():

        #ax.grid(True)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontweight(fontweight)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontweight(fontweight)


    plt.tight_layout()

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(f"{fig_name}_post_{blim}.pdf", format='pdf', dpi=72)

    del fig, axes


def FTVD_plot(stdfile, omega, M_p, R_est, R_est_init, R_min, sigmav, bins, means,
              pids_LR, rho_perc=25, xlim=10):
    """Helpful plots for FTVD disk finder.

    This function is will generate several plots that are helpful in debugging
    the FTVD disk finder.

    Args:
        stdfile (str): Post-impact Tipsy file.
        omega (float): Solid-body rotation rate in Hz.
        M_p (float): Planet mass as determined by disk finder in Earth masses.
        R_est (float): Estimated radius in Earth radii.
        R_est_init (float): Initial guess for radius in Earth radii.
        sigmav (float): Maximum allowable fractional deviation from solid-body
                        rotation.
        bins (list): Median transverse velocity in bin.
        means (list): Midpoints of bins in Earth radii.
        pids_LR (list): List of PIDs belonging to largest remnant.
        rho_perc (float): Density percentile that determines which particles are
                          used to find the center of mass.
        xlim (float): Maximum x-axis range for plot.

    Returns:
        None

    """
    # Output figure base name
    fig_name = str(os.path.basename(stdfile).split('.')[0])

    # Matplotlib plot parameters
    plt.rc('axes', linewidth=2)

    a4_inches = (8.268,11.693)

    fs_axlab = 15

    fontsize = 12


    # Load snapshot
    s = pynbody.load(stdfile)

    s = disk_finders.center_on_largest_remnant(s, pids_LR, rho_perc=rho_perc)


    # Instantiate figure
    fig, axes = plt.subplots(2, 1, figsize=a4_inches, sharex=True)


    # TOP PANEL
    ax = axes[0]

    ax.set_xlim(0, xlim)
    ax.set_ylim(-0.25, 10)

    ax.set_ylabel(r'$v_{t}$ $(km/s)$', fontsize=fs_axlab)

    ax.get_xaxis().set_visible(False)

    x = s['rxy'].in_units('6.3710084e3 km')  # Earth radii
    y = s['vt'].in_units('km s**-1')

    ax.axvline(x=R_min, linestyle='-', color='red', alpha=0.5,
               label='$R_{min}$')

    ax.axvline(x=R_est_init, linestyle=':', color='red', alpha=0.5,
               label='$R_{est\_init}$')

    ax.axvline(x=R_est, linestyle='--', color='red', alpha=0.5,
               label='$R_{est}$')

    ax.scatter(x, y, marker='.', s=5, c='black', alpha=0.25,
               zorder=1, rasterized=True)

    # Plot solid body rotation curve
    sbx = np.linspace(0, xlim, num=100)

    sby = [C.REARTH_KM * r * omega for r in sbx]

    ax.plot(sbx, sby, linewidth=2, linestyle='-',
            color='cyan', alpha=0.7, zorder=5,
            label='Solid body')


    # Plot Keplerian velocity curve
    kepx = np.linspace(0.1, xlim, num=1000)

    kepy = [orbital_velocity(float(M_p)*C.MEARTH_KG, r*C.REARTH_M) for r in kepx]

    ax.plot(kepx, kepy, linewidth=2, linestyle='--',
            color='black', alpha=0.7, zorder=5,
            label='Keplerian')


    ax.text(2.0, 1.0, f"$\Omega$ = {omega:.2e} Hz", fontsize=fs_axlab)

    # Legend
    ax.legend(prop={'size': fontsize})


    # BOTTOM PANEL
    ax = axes[1]

    ax.set_xlim(0, xlim)
    ax.set_ylim(-0.1, 0.1)

    ax.set_xlabel(r'$r_{xy}$ $(R_{\oplus})$', fontsize=fs_axlab)
    ax.set_ylabel(r'$(v_{t} - v_{t,sb}) / v_{t,sb}$', fontsize=fs_axlab)

    ax.set_yticks([-0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075])
    ax.set_yticklabels(['-7.5', '-5.0', '-2.5', '0.0', '+2.5', '+5.0', '+7.5'])

    ax.axvline(x=R_min, linestyle='-', color='red', alpha=0.5, label='$R_{min}$')
    ax.axvline(x=R_est_init, linestyle=':', color='red', alpha=0.5, label=r'$R_{est\_init}$')
    ax.axvline(x=R_est, linestyle='--', color='red', alpha=0.5, label=r'$R_{est}$')


    ax.fill_between(np.linspace(0, xlim, num=10), -sigmav, sigmav,
                    color='cyan', alpha=0.25, label='$\sigma_{v_{\perp}}$')

    ax.axhline(y=0, linestyle='-', color='cyan', alpha=0.9)


    ax.plot(bins, means, color='black', label='Median')


    ax.legend(prop={'size': fontsize})


    for ax in axes:
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontweight('bold')


    plt.tight_layout()

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(f"{fig_name}_ftvd_{xlim}.pdf", format='pdf', dpi=72)
    #plt.savefig(f"ftvd_radial_profiles_x{xlim}.png", format='png', dpi=350)

    del fig, axes
