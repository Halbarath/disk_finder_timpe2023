#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains the various schema used to classify collision outcomes.

This module contains the various schema used to classify collision outcomes. It
currently includes the schema used in Leinhardt & Stewart (2012) and a novel
scheme wherein the classes are exhaustive and mutually exclusive.

Todo:
    * Implement novel schema

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import numpy as np
from constants import G_SI


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    if np.linalg.norm(vector) == 0:
        return np.array([0.0, 0.0, 0.0])
    else:
        return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors v1 and v2."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angular_momentum(s):
    """Calculate the angular momentum in a snapshot with pynbody."""

    J = angular_momentum_vector(s)

    return np.linalg.norm(J)


def angular_momentum_vector(s):
    """Calculate the angular momentum vector of the snapshot with pynbody."""

    m = s['mass'].in_units('kg').reshape(len(s), 1)

    v = s['vel'].in_units('m s**-1')

    p = m * v

    r = s['pos'].in_units('m')

    return np.cross(r, p).sum(axis=0)


def cart2kep(r, v, mass, central_mass):
    """Cartesian to Keplerian coordinate conversion.

    Converts cartesian position and velocity to semi-major axis, eccentricity,
    and inclination. This saves time over calculating all Keplerian elements.
    This function is used by the disk finders to check which particles are bound
    to the post-impact system.

    Args:
        r (list): (x,y,z) Cartesian Positions
        v (list): (vx,vy,vz) Cartesian Velocities
        mass (float): Particle Mass
        central_mass (float): Mass of central object

    Returns:
        float: Semi-major axis
        float: Eccentricity
        float: Inclination

    """

    # Gravitational Parameter
    mu = G_SI * ( central_mass + mass )

    # Cartesian Unit Vectors
    iHat = np.array([1., 0., 0.])
    jHat = np.array([0., 1., 0.])
    kHat = np.array([0., 0., 1.])

    # Eccentricity
    h = np.cross(r, v)
    evec = 1. / mu * np.cross(v, h) - r / np.linalg.norm(r)
    ecc = np.linalg.norm(evec)

    # Semi Major Axis
    a = np.dot(h,h) / ( mu * ( 1. - ecc**2. ))

    # Inclination
    inc = np.arccos(np.dot(kHat, h) / np.linalg.norm(h))

    # Longitude of the Ascending Node
    n = np.cross(kHat, h)
    if inc == 0.0:
        Omega = 0.0
    else:
        Omega = np.arccos(np.dot(iHat, n) / np.linalg.norm(n))
        if np.dot(n, jHat) < 0:
            Omega = 2. * np.pi - Omega

    # Argument of Perigee
    # For Zero Inclination, Fall Back to 2D Case
    # http://en.wikipedia.org/wiki/Argument_of_periapsis
    if inc == 0.0:
        omega = np.arctan2(evec[1]/ecc,evec[0]/ecc)
    else:
        omega = np.arccos(np.dot(n, evec) / (np.linalg.norm(n) * ecc))
        if np.dot(evec, kHat) < 0:
            omega = 2. * np.pi - omega

    return a, ecc, inc, Omega, omega


def orbital_velocity(M, r):
    """Calculate the Keplerian orbital velocity at r.

    Args:
        M (float): Central mass in kg.
        r (float): Orbital distance in m.

    Returns:
        float: Orbital velocity in km/s.
    """
    
    return 1e-3 * np.sqrt(G_SI * M / r)


def rotation_period(omega):
    """Convert rotation rate (Hz) to rotation period (hours)."""
    if omega > 1e-12:
        return 2 * np.pi / (3600 * float(omega))
    else:
        return 0


def weighted_percentiles(values, percentiles, sample_weight=None):
    """ Very close to numpy.percentile, but supports weights.
        NOTE: percentiles should be in [0, 100]!

    Args:
        values: numpy.array with data
        percentiles: array-like with many percentiles needed
        sample_weight: array-like of the same length as `values`

    Returns:
        numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(percentiles) / 100.0
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'percentiles should be in [0, 100]'
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)
