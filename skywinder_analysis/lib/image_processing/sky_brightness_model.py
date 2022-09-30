import numpy as np

from skywinder_analysis.lib.tools import  relative_camera_pointing
from skywinder_analysis.lib.tools import GPS_pointing_and_sun as GPS
from scipy.interpolate import  RectBivariateSpline as RBS

def simulate_sky_in_flight(camera_number, img_time):
    sun_az, sun_alt = GPS.get_sun_az_alt(img_time)
    camera_az_offset = GPS.get_pointing_rotator(img_time) - sun_az + np.pi
    
    sky_brightness = np.zeros((5, 5))
    az, alt = relative_camera_pointing.get_camera_pointing_array(camera_number)
    az += camera_az_offset

    for i in range(5):
        for j in range(5):
            sky_brightness[i, j] = 1./np.sin(alt[i, j])*(1 + 
                  (np.cos(az[i, j])*np.cos(sun_alt)*np.cos(alt[i, j]) + np.sin(alt[i, j])*np.sin(sun_alt))**2)
    
    x_old = np.arange(0, 3233, 808)
    y_old = np.arange(0, 4865, 1216)

    x_new = np.arange(0,3232,1)
    y_new = np.arange(0,4864,1)
    
    f_sky = RBS(x_old, y_old, sky_brightness)
    
    full_sky_brightness = f_sky(x_new, y_new)
    return full_sky_brightness
