import os
import numpy as np
import pandas as pd

pointing_dir = '/data/mounted_filesystems/nas2/resources/pointing_solutions/'

def get_camera_pointing_array(camera_number):
    if camera_number != 99:
        fn = 'c{0}_2018-07-12_2330_solution.csv'.format(camera_number)
    else:
        fn = 'c99_2019-12-18_0205_solution.csv'.format(camera_number)
    df = pd.read_csv(os.path.join(pointing_dir, fn))
    az = np.radians(np.asarray(df.az))
    if camera_number != 99:
        az += np.radians(81.6525)
    az = np.reshape(az, (5,5)) 
    alt = np.reshape(np.radians(np.asarray(df.alt)), (5,5))
    return az, alt
