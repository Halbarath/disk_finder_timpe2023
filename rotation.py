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

import numpy as np
import pynbody

from utils import angle_between, weighted_percentiles


def angular_momentum(s):
    """Calculate the angular momentum in a snapshot with pynbody."""

    J = angular_momentum_vector(s)

    return float(np.linalg.norm(J))


def angular_momentum_vector(s):
    """Calculate the angular momentum vector of the snapshot with pynbody."""

    m = s['mass'].in_units('kg').reshape(len(s), 1)

    v = s['vel'].in_units('m s**-1')

    p = m * v

    r = s['pos'].in_units('m')

    return np.cross(r, p).sum(axis=0)


def angular_velocity(s):
    """Calculate the rotation rate of the snapshot."""

    r = s['pos'].in_units('km')

    r2 = (r**2).sum(axis=1).reshape(len(s),1)

    v = s['vel'].in_units('km s**-1')

    s['omega'] = np.cross(r, v) / (r2 + 1e-9)

    w = np.linalg.norm(s['omega'], axis=1)

    return weighted_percentiles(w,50,sample_weight=s['mass'].in_units('5.9722e24 kg'))


def obliquity(s):
    """Calculates the obliquity of an object.

    The obliquity is calculated by determining the angular momentum vector of
    the object and then calculating the angle between this vector and z-hat.

    Args:
        s (pynbody): Pynbody subsnap of isolated body.

    Returns:
        float: Obliquity in degrees.

    """
    unit_z = np.array([0,0,1])

    J_vec = angular_momentum_vector(s)

    theta = angle_between(J_vec, unit_z)

    return np.rad2deg(theta)


def rotation_period(omega):
    """Convert rotation rate to rotation period.

    Args:
        omega (float): Rotation rate in Hz.

    Returns:
        float: Rotation period in hours.

    """

    if omega > 1e-12:
        return 2 * np.pi / (3600 * float(omega))
    else:
        return 0
