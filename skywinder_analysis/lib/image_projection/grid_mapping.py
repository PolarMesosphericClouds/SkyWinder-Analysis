import pandas as pd
import os
import numpy as np
import scipy.interpolate as interpolate


def create_interpolation_fns(cam_num, pointing_directory):
    fn = os.path.join(pointing_directory, 'c{0}_coordinates.csv'.format(cam_num))
    x_original = np.arange(0, 3232, 34)
    y_original = np.arange(0, 4864, 34)
    HW = np.loadtxt(fn, delimiter=',')


def run_grid_map(fn, cam_num, bin_factor, pointing_directory):
    df = pd.read_csv(
        os.path.join(pointing_directory, ('c%d_2018-07-12_2330_solution.csv' % cam_num)))


    # f_h = interpolate.interp2d(df.az, df.alt, df.h,
    #                            kind='cubic', bounds_error=False, fill_value=np.nan)
    # f_w = interpolate.interp2d(df.az, df.alt, df.w,
    #                            kind='cubic', bounds_error=False, fill_value=np.nan)

    # Get pixel coordinates from x, y on sky
    h = f_h(x, y)
    w = f_w(x, y)

