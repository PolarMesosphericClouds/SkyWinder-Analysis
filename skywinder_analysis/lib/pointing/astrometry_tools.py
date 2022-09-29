import numpy as np
import sys
import csv

sys.path.append('/home/bjorn/astrometry.net-0.73/')
# This needs to be a custom setting. Refactor in future.

import tempfile
import os
import time
import subprocess
from skywinder_analysis.lib.pointing.fits import streaming_text_table

SOLVE_FIELD = 'solve-field'
WCSINFO = 'wcsinfo'
WCS_XY2RD = 'wcs-xy2rd'


def get_wcs_info(wcs_file):
    command = ' '.join([WCSINFO, wcs_file])
    wcs_info_string = subprocess.check_output(command, shell=True)
    result = {}
    for line in wcs_info_string.splitlines():
        k, v = line.split()
        try:
            v = float(v)
        except ValueError:
            pass
        result[k] = v
    return result


def get_wcs_xy2rd(wcs_file, xcoord, ycoord):
    command = ' '.join([WCS_XY2RD, '-x', str(xcoord), '-y', str(ycoord), '-w', wcs_file])
    wcs_xy2rd_string = subprocess.check_output(command, shell=True)
    return wcs_xy2rd_string


def get_mesh(xcoords, ycoords, work_dir=None, time_limit=10, verbose=False):
    # Get pointing for an image from list of xcoords and ycoords of stars in image.
    # Returns ra/dec for a mesh of points within the solved image.
    if work_dir is None:
        work_dir = tempfile.mkdtemp(suffix='pyadn_work')
    if verbose:
        print("working in %s" % work_dir)
        print("writing xy text file")
    text_xy = os.path.join(work_dir, 'xy.txt')
    with open(text_xy, 'w') as fh:
        for (x, y) in zip(xcoords, ycoords):
            fh.write('%f,%f\n' % (x, y))
    if verbose:
        print("creating fits file")
    stt = streaming_text_table(text_xy, split=",",
                               skiplines=0, headerline="x,y",
                               coltypes=[np.float64, np.float64], floatvalmap={'': np.nan},
                               intvalmap=dict(null=-1))
    fits_xy = os.path.join(work_dir, 'xy.fits')
    stt.write_to(fits_xy)

    command = [SOLVE_FIELD,
               fits_xy,
               '-l', str(time_limit),  # give up after this many seconds
               '-w', '3232',
               '-e', '4864',
               '--no-plot']
    command = ' '.join(command)

    tic = time.time()
    try:
        solve_result = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        solve_result = e.output
    elapsed = time.time() - tic
    if verbose:
        print("solved in %.3f" % elapsed)
        print("Solve command: %s" % (' '.join(command)))
        print("Solve result:\n" + solve_result)
    with open(os.path.join(work_dir, 'solve.log'), 'wb') as fh:
        fh.write(solve_result)

    corners = []

    for x in [0, 3232 / 4, 3232 / 2, 3 * 3232 / 4, 3232]:
        for y in [0, 4864 / 4, 4864 / 2, 3 * 4864 / 4, 4864]:
            # Used for full PMC-Turbo images (3232, 4864)
            print(x, y)
            corner = get_wcs_xy2rd(os.path.join(work_dir, 'xy.wcs'), x, y)
            corners.append(corner.decode())

    corner_coords = []

    for corner in corners:
        corner = str(corner)
        text = corner.split('(')[-1]
        values = text.strip(')\n').split(', ')
        print(values)
        ra = float(values[0])
        dec = float(values[1])
        corner_coords.append((ra, dec))

    return corner_coords


def get_corner_and_blob_coords(xcoords, ycoords, blobs_csv_filename, work_dir=None, time_limit=10, verbose=False):
    # Get pointing for an image from list of xcoords and ycoords of stars in image.
    # Returns ra/dec for four corners of the image within the solved image,
    # as well as the ra/dec for each of the coordinates given in the blobs_csv_filename.

    # This can be simplified as it already takes a list of blob coords. They aren't necessarily the coords that
    # one wants, however.
    if work_dir is None:
        work_dir = tempfile.mkdtemp(suffix='pyadn_work')
    if verbose:
        print("working in %s" % work_dir)
        print("writing xy text file")
    text_xy = os.path.join(work_dir, 'xy.txt')
    with open(text_xy, 'w') as fh:
        for (x, y) in zip(xcoords, ycoords):
            fh.write('%f,%f\n' % (x, y))
    if verbose:
        print("creating fits file")
    stt = streaming_text_table(text_xy, split=",",
                               skiplines=0, headerline="x,y",
                               coltypes=[np.float64, np.float64], floatvalmap={'': np.nan},
                               intvalmap=dict(null=-1))
    fits_xy = os.path.join(work_dir, 'xy.fits')
    stt.write_to(fits_xy)

    command = [SOLVE_FIELD,
               fits_xy,
               '-l', str(time_limit),  # give up after this many seconds
               '-w', '3232',
               '-e', '4864',
               '--no-plot']
    command = ' '.join(command)

    tic = time.time()
    try:
        solve_result = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        solve_result = e.output
    elapsed = time.time() - tic
    if verbose:
        print("solved in %.3f" % elapsed)
        print("Solve command: %s" % (' '.join(command)))
        print("Solve result:\n" + solve_result)
    with open(os.path.join(work_dir, 'solve.log'), 'w') as fh:
        fh.write(solve_result)

    corners = []

    for x in [0, 3232]:
        for y in [0, 4864]:
            print(x, y)
            corner = get_wcs_xy2rd(os.path.join(work_dir, 'xy.wcs'), x, y)
            corners.append(corner)

    corner_coords = []

    for corner in corners:
        text = corner.split('(')[-1]
        values = text.strip(')\n').split(', ')
        print(values)
        ra = float(values[0])
        dec = float(values[1])
        corner_coords.append((ra, dec))

    blobs_ = []
    blobs_ = []
    with open(blobs_csv_filename, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            blobs_.append((int(row[0]), int(row[1])))

    blob_locations = []
    for blob_ in blobs_:
        x, y = blob_
        blob_location = get_wcs_xy2rd(os.path.join(work_dir, 'xy.wcs'), x, y)
        blob_locations.append(blob_location)

    blob_coords = []
    for blob_location in blob_locations:
        text = blob_location.split('(')[-1]
        values = text.strip(')\n').split(', ')
        print(values)
        ra = float(values[0])
        dec = float(values[1])
        blob_coords.append((ra, dec))

    return corner_coords, blob_coords
