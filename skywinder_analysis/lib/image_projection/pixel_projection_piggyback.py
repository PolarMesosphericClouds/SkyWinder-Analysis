import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import scipy.stats as ss
from scipy.interpolate import RectBivariateSpline as RBS
from skywinder_analysis.lib.image_processing import flat_field_piggyback, binning
from skywinder_analysis.lib.image_projection import pixel_projection as pp
from skywinder_analysis.lib.tools import GPS_pointing_and_sun as GPS
from skywinder_analysis.lib.tools import periodic_image_finder
from astropy.coordinates import EarthLocation
from astropy import units as u


def get_x_y_arrays(bin_factor = 8, alignment='north', translate = False, current_time=1577000000, reference_time=1577000000, clipped=False, masked=True):
    #Gets X- and Y- arrays for scatter plot. alignment='north' returns arrays in coordinate system with +y to the north, +x to east.
    #alignment='antisun' returns arrays in coordinate system with +y in antisun direction.

    #In order to compensate for gondola translation, current time and reference time must be specified. The origin of the coordinate system will be 
    #directly overhead at the reference time - other images will be translated relative to that point. This only works for alignment='north'.

    x_old = np.arange(0, 3232, 34)
    y_old = np.arange(0, 4864, 34)

    if clipped:
        x_new = np.arange(0, 3232, bin_factor)
        y_new = np.arange(0, clipped, bin_factor)
    else:
        x_new = np.arange(0, 3232, bin_factor)
        y_new = np.arange(0, 4864, bin_factor)

      
    fn = '/data/mounted_filesystems/nas2/resources/pixel_coordinates_downsampled/c99_coordinates.csv'
    XY = np.loadtxt(fn, delimiter=',')
    X = XY.T[0].reshape(96,144)
    Y = XY.T[1].reshape(96,144)
    f_x = RBS(x_old, y_old, X)
    X_inter = f_x(x_new, y_new)
    f_y = RBS(x_old, y_old, Y)
    Y_inter = f_y(x_new, y_new)
    if masked:
        MASK, indices = flat_field_piggyback.get_mask(bin_factor)    
        X_array = X_inter[indices].flatten()
        Y_array = Y_inter[indices].flatten()
    else:
        X_array = X_inter.flatten()
        Y_array = Y_inter.flatten()
    
    if alignment == 'north':
        az_offset = GPS.get_pointing_rotator(current_time)

        if translate:
            X_offset = pp.get_image_translation_Earth_frame(reference_time, current_time)
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

def convert_pixel_arrays_to_sky_coordinates(X_px, Y_px, timestamps, bin_factor = 8, reference_time=1577000000):
    #Converts arrays of pixel coordinates to coordinates on the cloud layer. Returns arrays in coordinate system with +y to the north, +x to east.
    #In order to compensate for gondola translation, current time and reference time must be specified. The origin of the coordinate system will be 
    #directly overhead at the reference time - other images will be translated relative to that point. 

    x_old = np.arange(0, 3232, 34)
    y_old = np.arange(0, 4864, 34)
    fn = '/data/mounted_filesystems/nas2/resources/pixel_coordinates_downsampled/c99_coordinates.csv'
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
        

        X_offset = pp.get_image_translation_Earth_frame(reference_time, timestamp)
        
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

def convert_pixel_arrays_to_sky_coordinates_array(X_px, Y_px, timestamps, bin_factor = 8, reference_time=1577000000):
    #Converts arrays of pixel coordinates to coordinates on the cloud layer. Returns arrays in coordinate system with +y to the north, +x to east.
    #In order to compensate for gondola translation, current time and reference time must be specified. The origin of the coordinate system will be 
    #directly overhead at the reference time - other images will be translated relative to that point. 

    x_old = np.arange(0, 3232, 34)
    y_old = np.arange(0, 4864, 34)
    fn = '/data/mounted_filesystems/nas2/resources/pixel_coordinates_downsampled/c99_coordinates.csv'
    
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
    X_offset = pp.get_image_translation_Earth_frame_array(reference_time, timestamps)

    scaling = 1.0 + X_offset[2,:]/45.0

    X0_trans = (X0*np.cos(az_offset)*scaling
                    + Y0*np.sin(az_offset)*scaling
                    + X_offset[0,:])
    Y0_trans = (-X0*np.sin(az_offset)*scaling
                    + Y0*np.cos(az_offset)*scaling
                    + X_offset[1,:])

    return X0_trans, Y0_trans

def get_brightness_array(timestamp, bin_factor = 8, reflection_window = 600, clipped=False, masked=True):
    #Returns brightness array for given time, cameras, etc. Output will have same format as X- and 
    #Y-arrays assuming same parameters are chosen.
 
    img_fn = periodic_image_finder.find_periodic_images_full_fn(99, timestamp, timestamp, 1)
    
    if clipped:
        img = flat_field_piggyback.get_final_cleaned_image(img_fn[0], reflection_window = reflection_window, masked=masked)[:,:clipped]
    else:
        img = flat_field_piggyback.get_final_cleaned_image(img_fn[0], reflection_window = reflection_window, masked=masked)
    
    if masked:
        MASK, indices = flat_field_piggyback.get_mask(bin_factor)
        flat_img = binning.bucket(img, (bin_factor, bin_factor))[indices].flatten()
    else:
        flat_img = binning.bucket(img, (bin_factor, bin_factor)).flatten()
    
    return flat_img

def get_x_y_brightness(timestamp, bin_factor = 8, alignment='north', translate = False, reference_time=1577000000, reflection_window = 600, clipped=False, masked=True):
    #Returns X, Y and brightness arrays.
    brightness = get_brightness_array(timestamp, bin_factor, reflection_window, clipped, masked)
    X, Y = get_x_y_arrays(bin_factor, alignment, translate, timestamp, reference_time, clipped, masked)
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
        
def get_vmin_vmax(start, end, reflection_window = 600, percentiles = [0.1, 99.9], masked=True):
    #Returns vmin and vmax values for image sequence
    fns = []
    fn_list = periodic_image_finder.find_periodic_images_full_fn(99, start, end, 600)
    for fn in fn_list:
        fns.append(fn)
            
    vmin = []
    vmax = []

    for i in range(len(fns)):
        img = flat_field_piggyback.get_final_cleaned_image(fns[i], reflection_window, masked)
        vmin.append(ss.scoreatpercentile(img, percentiles[0]))
        vmax.append(ss.scoreatpercentile(img, percentiles[1]))

    return min(vmin), max(vmax)





