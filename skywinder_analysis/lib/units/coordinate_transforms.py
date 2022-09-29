import numpy as np


def cart_to_altaz(x, y, h=(83 - 38), centerline=100, verbose=False):
    # Cartesian coordinates in km on the pmc plane to alt/az
    l = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan(l / h)
    phi = np.arcsin(x / l)
    alt = 90 - np.rad2deg(theta)
    az = centerline + np.rad2deg(phi)
    if verbose:
        print('Cartesian x, y:', x, y)
        print('Alt az:', alt, az)
    return alt, az


def altaz_to_cart(alt, az, phi_offset=100, h=(83 - 38)):
    # Alt az to cartesian coordinates in km on the pmc plane.
    # Phi offset is the anti-sun angle if x=0 corresponding to anti sun direction is desired.
    theta = 90 - alt  # Zero degrees is straight up
    phi = az - phi_offset  # 100 degrees is roughly center of these particular solutions.
    l = h * np.tan(np.deg2rad(theta))
    y = l * np.cos(np.deg2rad(phi))
    x = l * np.sin(np.deg2rad(phi))
    return x, y
