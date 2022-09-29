import pandas as pd
import numpy as np
from scipy import interpolate as inter

from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time

def get_lat_long_alt_SIP(timestamp, df= pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_solar_data.csv')):
    #If calling this function repeatedly, reading in the csv separately and passing it will be more efficient. 
    #Returns latitude and longitude in degrees, altitude in kft

    if timestamp > 1532000000:
        df = pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_solar_data_piggyback_unwrapped.csv')
    
    i0 = np.where(df.epoch <= timestamp)[0][-1]
    i1 = np.where(df.epoch > timestamp)[0][0]
    
    alt = df.altitude[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.altitude[i1] - df.altitude[i0])
    lat = df.latitude[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.latitude[i1] - df.latitude[i0])
    long = df.longitude[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.longitude[i1] - df.longitude[i0])
    
    return lat, long, alt

def get_lat_long_alt_SIP_array(timestamps,
                               df= pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_solar_data.csv')):
    #If calling this function repeatedly, reading in the csv separately and passing it will be more efficient. 
    #Returns latitude and longitude in degrees, altitude in kft
    
    if timestamps[0] > 1532000000:
        df = pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_solar_data_piggyback_unwrapped.csv')
    f_lat = inter.interp1d(df.epoch, df.latitude)
    f_long = inter.interp1d(df.epoch, df.longitude)
    f_alt = inter.interp1d(df.epoch, df.altitude)
    
    
    alt = f_alt(timestamps)
    lat = f_lat(timestamps)
    long = f_long(timestamps)
    
    return lat, long, alt

def get_pointing_stars(timestamp, df = pd.read_csv('/data/mounted_filesystems/nas2/resources/star_pointing_data.csv')):
    #If calling this function repeatedly, reading in the csv separately and passing it will be more efficient. 
    #Returns azimuth pointing of center of the field of view in radians relative to N in CW direction
    #(i.e. pi/2 corresponds to E).
    if timestamp >= 1531100000:    
        i0 = np.where(df.epoch <= timestamp)[0][-1]
        i1 = np.where(df.epoch > timestamp)[0][0]
    
        if df.az[i0] > 330 and df.az[i1] < 30:
            az = df.az[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.az[i1] + 360 - df.az[i0])
            az = az % 360
        else:
            az = df.az[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.az[i1] - df.az[i0])
        return np.radians(az)
    else:
        az = get_pointing_rotator(timestamp)
        return az
    
def get_pointing_stars_array(timestamps, df = pd.read_csv('/data/mounted_filesystems/nas2/resources/star_pointing_data.csv')):
    if np.min(timestamps) >= 1531100000:
        f_az = inter.interp1d(df.epoch, df.az)
        az = f_az(timestamps)
        return np.radians(az)
    else:
        az = get_pointing_rotator_array(timestamps)
        return az

def get_pointing_SIP(timestamp, df = pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_yaw_data_cleaned.csv')):
    #If calling this function repeatedly, reading in the csv separately and passing it will be more efficient. 
    #Returns azimuth pointing of center of the field of view in radians relative to N in CW direction
    #(i.e. pi/2 corresponds to E).
    if timestamp > 1532000000:
        print('No SIP data available: use get_pointing_rotator')
        
    i0 = np.where(df.epoch <= timestamp)[0][-1]
    i1 = np.where(df.epoch > timestamp)[0][0]
    
    if df.yaw[i0] > 330 and df.yaw[i1] < 30:
        yaw = df.yaw[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.yaw[i1] + 360 - df.yaw[i0])
        yaw = yaw % 360
    else:
        yaw = df.yaw[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.yaw[i1] - df.yaw[i0])
    
    return np.radians(yaw)

def get_pointing_SIP_array(timestamps, df = pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_yaw_data_cleaned.csv')):
    if timestamps[0] > 1532000000:
        print('No SIP data available: use get_pointing_rotator_array')
    f_yaw = inter.interp1d(df.epoch, df.yaw)
    yaw = f_yaw(timestamps)
        
    return np.radians(yaw)

def get_pointing_rotator_array(timestamps,
                              df = pd.read_csv('/data/mounted_filesystems/nas2/resources/rotator_pointing_data_unwrapped.csv',
                                                     header=0, names=['epoch', 'az'])):
    #If calling this function repeatedly, reading in the csv separately and passing it will be more efficient. 
    #Returns azimuth pointing of center of the field of view in radians relative to N in CW direction
    #(i.e. pi/2 corresponds to E).
    
    if timestamps[0] > 1532000000:
        df = pd.read_csv('/data/mounted_filesystems/nas2/resources/rotator_pointing_data_piggyback_unwrapped.csv')
        f_az = inter.interp1d(df.epoch, df.az)
        az = np.radians(f_az(timestamps))
    
    else:
        f_az = inter.interp1d(df.epoch, df.az)
        az_offset = f_az(timestamps)
    
        sun_az, sun_alt = get_sun_az_alt_array(timestamps)
        az = sun_az + np.radians(az_offset) - np.pi
    
    return az

def get_pointing_rotator(timestamp, df = pd.read_csv('/data/mounted_filesystems/nas2/resources/rotator_pointing_data.csv',
                                                     header=0, names=['epoch', 'az'])):
    #If calling this function repeatedly, reading in the csv separately and passing it will be more efficient. 
    #Returns azimuth pointing of center of the field of view in radians relative to N in CW direction
    #(i.e. pi/2 corresponds to E).
    if timestamp > 1532000000:
        df = pd.read_csv('/data/mounted_filesystems/nas2/resources/rotator_pointing_data_piggyback_unwrapped.csv')
    
    i0 = np.where(df.epoch <= timestamp)[0][-1]
    i1 = np.where(df.epoch > timestamp)[0][0]
    
    if df.az[i0] > 150 and df.az[i1] < -150:
        az_offset = df.az[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1]
                                                            - df.epoch[i0])*(df.az[i1] + 360 - df.az[i0])
        az_offset = az_offset % 360
    else:
        az_offset = df.az[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] -
                                                            df.epoch[i0])*(df.az[i1] - df.az[i0])
    if timestamp > 1532000000:
        az = np.radians(az_offset)
    else:
        sun_az, sun_alt = get_sun_az_alt(timestamp)
        az = sun_az + np.radians(az_offset) - np.pi
    
    return az

def get_sun_az_alt(timestamp, df= pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_solar_data.csv')):  
    #If calling this function repeatedly, reading in the csv separately and passing it will be more efficient. 
    #Returns the azimuth and altitude of the sun in radians at the location of the gondola.

    if timestamp > 1532000000:
        df = pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_solar_data_piggyback_unwrapped.csv')
    
    i0 = np.where(df.epoch < timestamp)[0][-1]
    i1 = np.where(df.epoch > timestamp)[0][0]
    if (df.sun_az[i0] > 330 and df.sun_az[i1] < 30):
        az = df.sun_az[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.sun_az[i1] + 360 - df.sun_az[i0])
    else:
        az = df.sun_az[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.sun_az[i1] - df.sun_az[i0])
    az = az % 360
    
    alt = df.sun_alt[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.sun_alt[i1] - df.sun_alt[i0])
    
    return np.radians(az), np.radians(alt)

def get_sun_az_alt_array(timestamps):
    if timestamps[0] < 1532000000:
        df = pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_solar_data_unwrapped.csv')
    else:
        df = pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_solar_data_piggyback_unwrapped.csv')
    f_sun_az = inter.interp1d(df.epoch, df.sun_az)
    f_sun_alt = inter.interp1d(df.epoch, df.sun_alt)
    
    return np.radians(f_sun_az(timestamps)), np.radians(f_sun_alt(timestamps))

def get_moon_az_alt(timestamp, df= pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_lunar_data_piggyback_unwrapped.csv')):  
    #If calling this function repeatedly, reading in the csv separately and passing it will be more efficient. 
    #Returns the azimuth and altitude of the sun in radians at the location of the gondola.

    if timestamp > 1532000000:
        df = pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_lunar_data_piggyback_unwrapped.csv')
    else:
        print('Still need to generate csv file of lunar data for 2018 flight.')
        return
    i0 = np.where(df.epoch < timestamp)[0][-1]
    i1 = np.where(df.epoch > timestamp)[0][0]
    if (df.moon_az[i0] > 330 and df.moon_az[i1] < 30):
        az = df.moon_az[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.moon_az[i1] + 360 - df.moon_az[i0])
    else:
        az = df.moon_az[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.moon_az[i1] - df.moon_az[i0])
    az = az % 360
    
    alt = df.moon_alt[i0] + (timestamp - df.epoch[i0])/(df.epoch[i1] - df.epoch[i0])*(df.moon_alt[i1] - df.moon_alt[i0])
    
    return np.radians(az), np.radians(alt)

def get_moon_az_alt_array(timestamps):
    if timestamps[0] < 1532000000:
        print('Still need to generate csv file of lunar data for 2018 flight.')
        return
        #df = pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_solar_data_unwrapped.csv')
    else:
        df = pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_lunar_data_piggyback_unwrapped.csv')
    f_moon_az = inter.interp1d(df.epoch, df.moon_az)
    f_moon_alt = inter.interp1d(df.epoch, df.moon_alt)
    
    return np.radians(f_moon_az(timestamps)), np.radians(f_moon_alt(timestamps))