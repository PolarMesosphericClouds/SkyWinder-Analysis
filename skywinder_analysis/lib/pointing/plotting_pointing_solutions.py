import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs


def plot_mollweide(ra_dec_coords, degrees=False):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="mollweide", axisbg='LightCyan')
    ax.grid(True)
    for coords in ra_dec_coords:
        fov = np.array(coords)
        ra = fov[:, 0]
        dec = fov[:, 1]
        if degrees:
            ra = np.deg2rad(ra)
            dec = np.deg2rad(dec)
        ax.plot(ra, dec, '.')


def plot_az_alt_cartesian(all_altaz_coords):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_ylabel('Alt (deg)')
    ax.set_xlabel('Az (deg)')
    for coords in all_altaz_coords[:]:
        alts = []
        azs = []
        for coord in coords:
            alts.append(coord.alt.value)
            azs.append(coord.az.value)
        ax.plot(azs, alts, '.')
    ax.plot()
    ax.set_ylim(15, 70)


def plot_orthographic_projection(all_altaz_coords, central_longitude, central_latitude, az_offset):
    plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude, central_latitude))

    ax.set_global()
    ax.gridlines()

    colors = 'bgrcmyk'

    ax.plot(range(-180, 180, 1), [65] * len(range(-180, 180, 1)), color='black', linewidth=1, marker='.',
            transform=ccrs.Geodetic())
    # 25 degrees off zenith

    for i, corners in enumerate(all_altaz_coords):
        lats = []
        lons = []
        for corner in corners:
            lats.append(corner.alt.value)
            lons.append(corner.az.value - az_offset)

        ax.scatter(lons, lats,
                   color=colors[i], marker='o', transform=ccrs.Geodetic())


def plot_gnomonic_projection(all_altaz_coords, az_offset):
    plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=ccrs.Gnomonic(central_latitude=90))

    ax.set_global()
    ax.gridlines()

    colors = 'bgrcmyk'

    ax.plot(range(-180, 180, 1), [65] * len(range(-180, 180, 1)), color='black', linewidth=1, marker='.',
            transform=ccrs.Geodetic())
    # 25 degrees off zenith

    for i, coords in enumerate(all_altaz_coords):
        lats = []
        lons = []
        for coord in coords:
            lats.append(coord.alt.value)
            lons.append(coord.az.value - az_offset)

        ax.scatter(lons, lats,
                   color=colors[i], marker='o', transform=ccrs.Geodetic())
