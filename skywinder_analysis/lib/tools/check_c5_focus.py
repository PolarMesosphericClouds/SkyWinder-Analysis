import glob
import os
import pandas as pd
import numpy as np
from scipy import interpolate as inter

import pmc_analysis


def check_focus(times):
    df1 = pd.read_csv('/data/mounted_filesystems/nas2/c5/var/pmclogs/housekeeping/camera/2018-07-07_100517.csv', comment='#')
    df2 = pd.read_csv('/data/mounted_filesystems/nas2/c5/var/pmclogs/housekeeping/camera/2018-07-13_031422.csv', comment='#')

    full_focus = np.concatenate((np.asarray(df1.EFLensFocusCurrent, dtype=np.float64),
                        np.asarray(df2.EFLensFocusCurrent, dtype=np.float64)))
    epoch = np.concatenate((np.asarray(df1.epoch, dtype=np.float64),
                        np.asarray(df2.epoch, dtype=np.float64)))
    f_focus = inter.interp1d(epoch, full_focus)

    focus = f_focus(times)
    focus_flag = np.ones(len(times))
    focus_flag[focus < 4000] = 0
    focus_flag[focus > 6000] = 0

    return focus_flag
