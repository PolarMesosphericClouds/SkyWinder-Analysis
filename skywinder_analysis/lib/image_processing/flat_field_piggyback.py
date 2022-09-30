from __future__ import division

import numpy as np
from scipy.interpolate import interp1d
from scipy import optimize as opt
import os
import datetime as dt

from skywinder_analysis.lib.tools import blosc_file, check_c5_focus
from skywinder_analysis.lib.image_processing import sky_brightness_model, dark_images, binning


def get_final_cleaned_image(fn, reflection_window=600, masked=True):
    #Reflection window should be given in seconds. The actual window used is the 
    #value given rounded down to the nearest 10 minutes (or rounded up to 10 minutes if less than 10 minutes)
    #Returns final cleaned image in units of counts/second, with some arbitrary offset.
    timestamp = int(fn.split('=')[-1][:10])
    
    calibrated_image = get_calibrated_image(fn)
    reflection_image = get_composite_flat(timestamp, reflection_window)
    if masked:
        MASK, indices = get_mask()
        return 10*(calibrated_image - reflection_image)*MASK
    else:
        return 10*(calibrated_image - reflection_image)

def get_mask(bin_factor=1):
    MASK = np.ones((3232,4864))
    for i in range(500):
        MASK[i, :4*(500 - i)] = np.nan
        MASK[-i-1, -4*(500 - i):] = np.nan
    mask = np.where(np.isnan(MASK) == False)
    binned_MASK = binning.bucket(MASK, (bin_factor,bin_factor))
    binned_mask = np.where(np.isnan(binned_MASK) == False)
    return binned_MASK, binned_mask

def get_calibrated_image(fn):
    camera_number = 99
    timestamp = int(fn.split('=')[-1][:10])
    
    img, chunk = blosc_file.load_blosc_image(fn)
    exposure = float(chunk[0][6])
    
    flat = get_flat_field_image_pre_flight(camera_number)
    dark = dark_images.get_dark_image(camera_number, exposure)
    
    calibrated_image = ((img.astype('float') - dark.astype('float'))/flat.astype('float'))*np.mean(flat)*100000/exposure
    return calibrated_image

def get_composite_flat(timestamp, reflection_window = 600):
    N = np.max([reflection_window // 600, 1]) 
    #direc = '/home/christopher/SkyWinder-Analysis/skywinder_analysis/output/foreground_subtracted_composites_piggyback/'
    direc = '/data/mounted_filesystems/nas2/resources/piggyback_moving_averages/'
    fn_list = os.listdir(direc)
    time_list = []
    time0 = 1576454400
    for fn in fn_list:
        date = fn.split('--')[0][-10:]
        month = int(date[:2])
        day = int(date[3:5])
        time = fn.split('--')[1]
        hour = int(time[:2])
        minute = int(time[3:5])
        if month == 12:
            time = time0 + 86400*(day - 16) + 3600*hour + 60*(minute + 5)
        else:
            time = time0 + 86400*(day + 15) + 3600*hour + 60*(minute + 5)
        time_list.append(time)

    times = np.asarray(time_list)
    fns = np.asarray(fn_list)
    sorted_times = times[times.argsort()]
    sorted_fns = fns[times.argsort()] 
    
    index = np.where(sorted_times <= timestamp)[0][-1]
    if N % 2 == 0:
        if timestamp - sorted_times[index] > 300:
            index += 1
            
            
    imd = np.zeros((3232, 4864, 2))
    img = np.zeros((3232, 4864))
    
    for i in range(N - 1):
        temp, _ = blosc_file.load_blosc_image(os.path.join(direc, sorted_fns[index - N // 2 + 1 + i]))
        img += temp.astype('float')/float(N)
    
    imd[:,:,0], _ = blosc_file.load_blosc_image(os.path.join(direc, sorted_fns[index - N // 2]))
        
    x = np.array([sorted_times[index], sorted_times[index + 1]])
    if N % 2 == 0:
        x -= 300
        imd[:,:,1], _ = blosc_file.load_blosc_image(os.path.join(direc, sorted_fns[index + N // 2]))
    else:
        imd[:,:,1], _ = blosc_file.load_blosc_image(os.path.join(direc, sorted_fns[index + N // 2 + 1]))
    
    img_interp = interp1d(x, imd)
    
    img += img_interp(timestamp)/float(N)
    
    img0 = 2*img.astype('float')

    return img0

def get_flat_field_image_pre_flight(camera_number):
    if camera_number == 99:
        camera_number = 7
    flat, _ = blosc_file.load_blosc_image('/data/mounted_filesystems/nas2/resources/flat_fields_pre-flight/camera{0}_flat_field'.format(camera_number))
    return flat
