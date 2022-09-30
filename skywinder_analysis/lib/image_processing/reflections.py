import os

import numpy as np
import scipy.stats
from scipy.interpolate import interp1d 

from skywinder_analysis.lib.tools import blosc_file, periodic_image_finder
from skywinder_analysis.lib.image_processing import flat_field, dark_images, sky_brightness_model

def make_composite_reflections_image(camera_number, start, end, interval):
    fns = periodic_image_finder.find_periodic_images_full_fn(camera_number, start, end, interval)
    imd = np.zeros((3232, 4864))
    for fn in fns:
        img = flat_field.get_foreground_subtracted_image(camera_number, fn)
        imd += 20000.0*img/float(len(fns))
    imd -= 2*np.min(imd)
    return imd 

