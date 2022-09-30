from __future__ import division

import numpy as np
from scipy.interpolate import interp1d
import os
import datetime as dt

from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import sky_brightness_model, dark_images


def get_final_cleaned_image(fn, reflection_window=600):
    #Reflection window should be given in seconds. The actual window used is the value given rounded down to the nearest 10 minutes (or rounded up to 10 minutes if less than 10 minutes)
    #Returns final cleaned image in units of counts/second, with some arbitrary offset.
    camera_number = int(fn.split('/')[4][1])
    timestamp = int(fn.split('=')[-1][:10])
    
    calibrated_image = get_foreground_subtracted_image(camera_number, fn)
    reflection_image = get_composite_flat(camera_number, timestamp, reflection_window)

    return 1000000.0*(calibrated_image - reflection_image)


def get_final_cleaned_image_aurora(fn, reflection_window=60):
    #Reflection window should be given in seconds. The actual window used is the value given rounded down to the nearest 10 minutes (or rounded up to 10 minutes if less than 10 minutes)
    #Returns final cleaned image in units of counts/second, with some arbitrary offset.
    camera_number = int(fn.split('/')[4][1])
    timestamp = int(fn.split('=')[-1][:10])
    
    calibrated_image = get_foreground_subtracted_image(camera_number, fn)
    reflection_image = get_composite_flat_aurora(camera_number, timestamp, reflection_window)

    return 1000000.0*(calibrated_image - reflection_image)

def get_calibrated_image(camera_number, fn):
    timestamp = int(fn.split('=')[-1][:10])
    
    img, chunk = blosc_file.load_blosc_image(fn)
    exposure = float(chunk[0][6])
    
    flat = get_flat_field_image_pre_flight(camera_number)
    dark = dark_images.get_dark_image(camera_number, exposure)
    
    calibrated_image = ((img.astype('float') - dark.astype('float'))/flat.astype('float'))*np.mean(flat)
    return calibrated_image

def get_foreground_subtracted_image(camera_number, fn):
    timestamp = int(fn.split('=')[-1][:10])
    
    img, chunk = blosc_file.load_blosc_image(fn)
    exposure = float(chunk[0][6])
    
    flat = get_flat_field_image_pre_flight(camera_number)
    sky = sky_brightness_model.simulate_sky_in_flight(camera_number, timestamp)
    dark = dark_images.get_dark_image(camera_number, exposure)
    
    flat_image = ((img.astype('float') - dark.astype('float'))/flat.astype('float'))*np.mean(flat)
    foreground_subtracted_image = ((flat_image.astype('float') - sky.astype('float')*np.mean(flat_image)/np.mean(sky))/exposure)
    return foreground_subtracted_image


def get_composite_flat(camera_number, timestamp, reflection_window = 600):
    N = np.max([reflection_window // 600, 1]) 
    direc = '/data/mounted_filesystems/nas2/resources/foreground_subtracted_composites_new/c{0}/'.format(camera_number)
    fns = sorted(os.listdir(direc))
    time_list = []
    for fn in fns:
        ts = dt.datetime.strptime(fn[30:47], '%m_%d_%Y--%H:%M').timestamp() - 14100
        time_list.append(ts)

    times = np.asarray(time_list)
    index = np.where(times <= timestamp)[0][-1]
    if N % 2 == 0:
        if timestamp - times[index] > 300:
            index += 1
            
            
    imd = np.zeros((3232, 4864, 2))
    img = np.zeros((3232, 4864))
    
    for i in range(N - 1):
        temp, _ = blosc_file.load_blosc_image(os.path.join(direc, fns[index - N // 2 + 1 + i]))
        img += temp.astype('float')/float(N)
    
    imd[:,:,0], _ = blosc_file.load_blosc_image(os.path.join(direc, fns[index - N // 2]))
        
    x = np.array([times[index], times[index + 1]])
    if N % 2 == 0:
        x -= 300
        imd[:,:,1], _ = blosc_file.load_blosc_image(os.path.join(direc, fns[index + N // 2]))
    else:
        imd[:,:,1], _ = blosc_file.load_blosc_image(os.path.join(direc, fns[index + N // 2 + 1]))
    
    img_interp = interp1d(x, imd)
    
    img += img_interp(timestamp)/float(N)
    
    img0 = img.astype('float')/20000.0
    
    img1 = img0 - 2*np.min(img0)

    return img1

def get_composite_flat_aurora(camera_number, timestamp, reflection_window = 60):
    N = np.max([reflection_window // 60, 1]) 
    direc = '/data/mounted_filesystems/nas2/resources/foreground_subtracted_composites_1min/c{0}/'.format(camera_number)
    fns = sorted(os.listdir(direc))

    times = []
    for fn in fns:
        day = day = int(fn[33:35])
        time = 1530403200 + 86400*(day-1) + int(fn[-5:-3])*3600 + int(fn[-2:])*60 - 30
        times.append(time)
    times = np.asarray(times)

    index = np.where(times <= timestamp)[0][-1]
    if N % 2 == 0:
        if timestamp - times[index] > 30:
            index += 1
            
            
    imd = np.zeros((3232, 4864, 2))
    img = np.zeros((3232, 4864))
    
    for i in range(N - 1):
        temp, _ = blosc_file.load_blosc_image(os.path.join(direc, fns[index - N // 2 + 1 + i]))
        img += temp.astype('float')/float(N)
    
    imd[:,:,0], _ = blosc_file.load_blosc_image(os.path.join(direc, fns[index - N // 2]))
        
    x = np.array([times[index], times[index + 1]])
    if N % 2 == 0:
        x -= 30
        imd[:,:,1], _ = blosc_file.load_blosc_image(os.path.join(direc, fns[index + N // 2]))
    else:
        imd[:,:,1], _ = blosc_file.load_blosc_image(os.path.join(direc, fns[index + N // 2 + 1]))
    
    img_interp = interp1d(x, imd)
    
    img += img_interp(timestamp)/float(N)
    
    img0 = img.astype('float')/20000.0
    
    img1 = img0 - 2*np.min(img0)

    return img1

def get_flat_field_image_pre_flight(camera_number):
    flat, _ = blosc_file.load_blosc_image('/data/mounted_filesystems/nas2/resources/flat_fields_pre-flight/camera{0}_flat_field'.format(camera_number))
    return flat

def get_uncorrected_flat_field_image_pre_flight(camera_number):
    flat, _ = blosc_file.load_blosc_image('/home/christopher/SkyWinder-Analysis/skywinder_analysis/resources/flat_fields/camera{0}_uncorrected_flat_field'.format(camera_number))
    return flat

def generate_flat_field_image_from_filenames(fns):
    images = []
    for fn in fns:
        try:
            image, _ = blosc_file.load_blosc_image(fn)
            images.append(image)
        except Exception as e:
            print(e)
            print(fn)
    return generate_flat_field_image_from_raw(images)


def generate_flat_field_image_from_raw(images):
    image_array = np.array(images)
    return np.average(image_array, axis=0)


def generate_flat_field_running_average(fns):
    ff_img = np.zeros((3232, 4864))
    images_added = 0
    for fn in fns:
        try:
            image, _ = blosc_file.load_blosc_image(fn)
            ff_img = (images_added * ff_img) / (images_added + 1) + image / (images_added + 1)
            images_added += 1
        except Exception as e:
            print(e)
            print(fn)
    return ff_img


def apply_flat_field(image, flat_field_image):
    return image * np.mean(flat_field_image) / flat_field_image


def apply_flat_field_no_relevel(image, flat_field_image):
    return image / flat_field_image


def apply_flat_field_subtract(image, flat_field_image):
    return (image + np.mean(flat_field_image)) - flat_field_image
