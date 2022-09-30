import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import scipy.stats as ss
from scipy.interpolate import RectBivariateSpline as RBS
from skywinder_analysis.lib.image_processing import flat_field, binning
from skywinder_analysis.lib.tools import GPS_pointing_and_sun as GPS
from skywinder_analysis.lib.tools import periodic_image_finder
from astropy.coordinates import EarthLocation
from astropy import units as u

def get_image_translation_Earth_frame(timestamp0, timestamp1):
    #Gets the translation of the gondola from one timestamp to the next in the coordinate system defined
    #at timestamp0 (with +y axis pointing towards North and +x axis pointing East)
    lat0, long0, alt0 = GPS.get_lat_long_alt_SIPS(timestamp0)
    lat1, long1, alt1 = GPS.get_lat_long_alt_SIPS(timestamp1)
    mylocation0 = EarthLocation(lat=lat0*u.deg,  lon=long0*u.deg, height=alt0/3.28*u.meter)
    mylocation1 = EarthLocation(lat=lat1*u.deg,  lon=long1*u.deg, height=alt1/3.28*u.meter)
    x1 = np.array([mylocation1.x.value - mylocation0.x.value,
               mylocation1.y.value - mylocation0.y.value, 
               mylocation1.z.value - mylocation0.z.value])/1000.0
    # x1 is the translation vector between the gondola position at timestamp0 and timestamp1 in the 
    #geodetic frame (+x points to lat = 0, long = 0, +y points to lat = 0, long = 90 degrees,
    #+z points to lat = 90 degrees)
    
    theta = np.radians(long0)
    phi = np.radians(lat0)
    NE_x = np.array([-np.sin(theta),
                     np.cos(theta),
                    0])
    NE_y = np.array([-np.cos(theta)*np.sin(phi),
                    -np.sin(theta)*np.sin(phi),
                    np.cos(phi)])
    NE_z = np.array([np.cos(theta)*np.cos(phi),
                    np.sin(theta)*np.cos(phi),
                    np.sin(phi)])
    #The NE_i vectors define the local NESW coordinate system relative to the geodetic frame.
    
    x1_NE = np.array([np.dot(x1, NE_x), np.dot(x1, NE_y), np.dot(x1, NE_z)])
    #x1_G is the translation vector in the local NESW frame.
    
    return x1_NE

def get_image_translation_Earth_frame_array(timestamp0, timestamp_array):
    #Gets the translation of the gondola from one timestamp to the next in the coordinate system defined
    #at timestamp0 (with +y axis pointing towards North and +x axis pointing East)
    lat0, long0, alt0 = GPS.get_lat_long_alt_SIPS(timestamp0)
    lat_array, long_array, alt_array = GPS.get_lat_long_alt_SIPS_array(timestamp_array)
    mylocation0 = EarthLocation(lat=lat0*u.deg,  lon=long0*u.deg, height=alt0/3.28*u.meter)
    mylocation_array = EarthLocation(lat=lat_array*u.deg,  lon=long_array*u.deg, height=alt_array/3.28*u.meter)
    x1 = np.array([mylocation_array.x.value - mylocation0.x.value,
                   mylocation_array.y.value - mylocation0.y.value, 
                   mylocation_array.z.value - mylocation0.z.value])/1000.0
    # x1 is the translation vector between the gondola position at timestamp0 and timestamp1 in the 
    #geodetic frame (+x points to lat = 0, long = 0, +y points to lat = 0, long = 90 degrees,
    #+z points to lat = 90 degrees)
    
    theta = np.radians(long0)
    phi = np.radians(lat0)  
    NE_x = np.array([-np.sin(theta),
                     np.cos(theta),
                    0])
    NE_y = np.array([-np.cos(theta)*np.sin(phi),
                    -np.sin(theta)*np.sin(phi),
                    np.cos(phi)])
    NE_z = np.array([np.cos(theta)*np.cos(phi),
                    np.sin(theta)*np.cos(phi),
                    np.sin(phi)])
    #The NE_i vectors define the local NESW coordinate system relative to the geodetic frame.
    
    NE = np.array([NE_x, NE_y, NE_z])
    
    x1_NE = np.matmul(NE, x1)
    #x1_NE is the translation vector in the local NESW frame.
    
    return x1_NE

def get_x_y_arrays(camera_numbers=[4,5,6,7], bin_factor = 8, alignment='north', translate = False, current_time=1531350000, reference_time=1531350000, clipped=False):
    #Gets X- and Y- arrays for scatter plot. alignment='north' returns arrays in coordinate system with +y to the north, +x to east.
    #alignment='antisun' returns arrays in coordinate system with +y in antisun direction.

    #In order to compensate for gondola translation, current time and reference time must be specified. The origin of the coordinate system will be 
    #directly overhead at the reference time - other images will be translated relative to that point. This only works for alignment='north'.

    x_old = np.arange(0, 3232, 34)
    y_old = np.arange(0, 4864, 34)

    if clipped:
        x_new = np.arange(0, 3232, bin_factor)
        y_new = np.arange(0, 4416, bin_factor)
    else:
        x_new = np.arange(0, 3232, bin_factor)
        y_new = np.arange(0, 4864, bin_factor)

    X_array = np.zeros(0)
    Y_array = np.zeros(0)
    
    for i in range(len(camera_numbers)):
        camera_number = camera_numbers[i]
        fn = '/data/mounted_filesystems/nas2/resources/pixel_coordinates_downsampled/c{0}_coordinates.csv'.format(camera_number)
        XY = np.loadtxt(fn, delimiter=',')
        X = XY.T[0].reshape(96,144)
        Y = XY.T[1].reshape(96,144)
        f_x = RBS(x_old, y_old, X)
        X_inter = f_x(x_new, y_new)
        f_y = RBS(x_old, y_old, Y)
        Y_inter = f_y(x_new, y_new)
        
        X_array = np.hstack((X_array, X_inter.flatten()))
        Y_array = np.hstack((Y_array, Y_inter.flatten()))
    
    if translate and current_time == 1531350000 and reference_time == 1531350000:
        print('Warning: in order to account for gondola translation, current and reference time must be specified')
    if translate and alignment=='antisun':
        print('Warning: translate option not valid for antisun alignment')

    if alignment == 'north':
        az_offset = GPS.get_pointing_rotator(current_time)

        if translate:
            X_offset = get_image_translation_Earth_frame(reference_time, current_time)
        else:
            X_offset=np.array([0, 0, 0])

        scaling = 1.0 + X_offset[2]/45.0
        
        X_array_trans = (X_array*np.cos(az_offset)*scaling
                        + Y_array*np.sin(az_offset)*scaling
                        + X_offset[0])
        Y_array_trans = (-X_array*np.sin(az_offset)*scaling
                        + Y_array*np.cos(az_offset)*scaling
                        + X_offset[1])

    elif alignment == 'antisun':        
        X_array_trans = X_array
        Y_array_trans = Y_array
    else:
        print('alignment must be either \'north\' or \'antisun\'')
        return

    return X_array_trans, Y_array_trans

def convert_pixel_arrays_to_sky_coordinates(X_px, Y_px, timestamps, camera_number=5, bin_factor = 8, reference_time=1531350000):
    #Converts arrays of pixel coordinates to coordinates on the cloud layer. Returns arrays in coordinate system with +y to the north, +x to east.
    #In order to compensate for gondola translation, current time and reference time must be specified. The origin of the coordinate system will be 
    #directly overhead at the reference time - other images will be translated relative to that point. 

    x_old = np.arange(0, 3232, 34)
    y_old = np.arange(0, 4864, 34)
    fn = '/data/mounted_filesystems/nas2/resources/pixel_coordinates_downsampled/c{0}_coordinates.csv'.format(camera_number)
    XY = np.loadtxt(fn, delimiter=',')
    X = XY.T[0].reshape(96,144)
    Y = XY.T[1].reshape(96,144)
    f_x = RBS(x_old, y_old, X)
    f_y = RBS(x_old, y_old, Y)
    X_array = np.zeros(0)
    Y_array = np.zeros(0)
    for x, y, timestamp in zip(X_px, Y_px, timestamps):
        X0 = f_x(x*bin_factor + bin_factor/2., y*bin_factor + bin_factor/2.).flatten()
        Y0 = f_y(x*bin_factor + bin_factor/2., y*bin_factor + bin_factor/2.).flatten()
    
        az_offset = GPS.get_pointing_rotator(timestamp)

        X_offset = get_image_translation_Earth_frame(reference_time, timestamp)
        
        scaling = 1.0 + X_offset[2]/45.0

        X0_trans = (X0*np.cos(az_offset)*scaling
                        + Y0*np.sin(az_offset)*scaling
                        + X_offset[0])
        Y0_trans = (-X0*np.sin(az_offset)*scaling
                        + Y0*np.cos(az_offset)*scaling
                        + X_offset[1])
        X_array = np.hstack((X_array, X0_trans))
        Y_array = np.hstack((Y_array, Y0_trans))    


    return X_array, Y_array

def convert_pixel_arrays_to_sky_coordinates_array(X_px, Y_px, timestamps, camera_number=5, bin_factor = 8, 
                                                 reference_time=1531350000):
    #Converts arrays of pixel coordinates to coordinates on the cloud layer. Returns arrays in coordinate system with +y to the north, +x to east.
    #In order to compensate for gondola translation, current time and reference time must be specified. The origin of the coordinate system will be 
    #directly overhead at the reference time - other images will be translated relative to that point. 

    x_old = np.arange(0, 3232, 34)
    y_old = np.arange(0, 4864, 34)
    fn = '/data/mounted_filesystems/nas2/resources/pixel_coordinates_downsampled/c{0}_coordinates.csv'.format(camera_number)
    XY = np.loadtxt(fn, delimiter=',')
    X = XY.T[0].reshape(96,144)
    Y = XY.T[1].reshape(96,144)
    f_x = RBS(x_old, y_old, X)
    f_y = RBS(x_old, y_old, Y)
    X_array = np.zeros(0)
    Y_array = np.zeros(0)
    #for x, y, timestamp in zip(X_px, Y_px, timestamps):
        
    X0 = f_x.ev(X_px*bin_factor + bin_factor/2., Y_px*bin_factor + bin_factor/2.).flatten()
    Y0 = f_y.ev(X_px*bin_factor + bin_factor/2., Y_px*bin_factor + bin_factor/2.).flatten()
    
    az_offset = GPS.get_pointing_rotator_array(timestamps)
    X_offset = get_image_translation_Earth_frame_array(reference_time, timestamps)

    scaling = 1.0 + X_offset[2,:]/45.0

    X0_trans = (X0*np.cos(az_offset)*scaling
                    + Y0*np.sin(az_offset)*scaling
                    + X_offset[0,:])
    Y0_trans = (-X0*np.sin(az_offset)*scaling
                    + Y0*np.cos(az_offset)*scaling
                    + X_offset[1,:])

    return X0_trans, Y0_trans

def get_brightness_array(timestamp, camera_numbers=[4,5,6,7], bin_factor = 8, reflection_window = 600, clipped=False):
    #Returns brightness array for given time, cameras, etc. Output will have same format as X- and 
    #Y-arrays assuming same parameters are chosen.

    brightness = np.zeros(0)
    for i in range(len(camera_numbers)):
        camera_number = camera_numbers[i]
        
        img_fn = periodic_image_finder.find_periodic_images_full_fn(camera_number, timestamp, timestamp, 1)
        if reflection_window >= 600:
            if clipped:
                img = flat_field.get_final_cleaned_image(img_fn[0], reflection_window = reflection_window)[:,:4416]
            else:
                img = flat_field.get_final_cleaned_image(img_fn[0], reflection_window = reflection_window)
        else:
            if clipped:
                img = flat_field.get_final_cleaned_image_aurora(img_fn[0], reflection_window = reflection_window)[:,:4416]
            else:
                img = flat_field.get_final_cleaned_image_aurora(img_fn[0], reflection_window = reflection_window)
            
        flat_img = binning.bucket(img, (bin_factor, bin_factor)).flatten()

        brightness = np.hstack((brightness, flat_img))
    
    return brightness

def get_x_y_brightness(timestamp, camera_numbers=[4,5,6,7], bin_factor = 8, alignment='north', translate = False, reference_time=1531350000, reflection_window = 600, clipped=False):
    #Returns X, Y and brightness arrays.
    if translate and reference_time == 1531350000:
        print('Warning: in order to account for gondola translation, current and reference time must be specified')

    brightness = get_brightness_array(timestamp, camera_numbers, bin_factor, reflection_window, clipped)
    X, Y = get_x_y_arrays(camera_numbers, bin_factor, alignment, translate, timestamp, reference_time, clipped)
    return X, Y, brightness

def plot_projection(x, y, b, vmin=None, vmax=None, fn='fig.png', savefig=True, limits = [(0, 10), (0, 10)], bin_factor = 4,
                    outpath='/home/christopher/PMC-Analysis/skywinder_analysis/resources/temp/', colormap=cm.inferno):
    
    extent = max(np.abs(limits[0][1] - limits[0][0]),
                 np.abs(limits[1][1] - limits[1][0]))
    n_pixels = extent*64./bin_factor
    scaling = 250.*(50/n_pixels)**2
    size = (x**2 + y**2 + 45**2)/60**2

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    
    if not vmin:
        vmin = ss.scoreatpercentile(b, 0.1)
    if not vmax:
        vmax = ss.scoreatpercentile(b, 99.9)
        
    scat = ax.scatter(x, y, s=size*scaling, c=b, cmap=colormap, vmin=vmin, vmax=vmax,
                  alpha=.9)
    ax.set_aspect('equal')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    if savefig:
        fig.savefig(os.path.join(outpath, fn), dpi=100)
    fig.clf()
    plt.close(fig)
        
def get_vmin_vmax(start, end, camera_numbers=[6,7], reflection_window = 600, percentiles = [0.1, 99.9]):
    #Returns vmin and vmax values for image sequence
    fns = []
    for cam in camera_numbers:
        fn_list = periodic_image_finder.find_periodic_images_full_fn(cam, start, end, 600)
        for fn in fn_list:
            fns.append(fn)
            
    vmin = []
    vmax = []

    for i in range(len(fns)):
        img = flat_field.get_final_cleaned_image(fns[i], reflection_window)
        vmin.append(ss.scoreatpercentile(img, percentiles[0]))
        vmax.append(ss.scoreatpercentile(img, percentiles[1]))

    return min(vmin), max(vmax)





