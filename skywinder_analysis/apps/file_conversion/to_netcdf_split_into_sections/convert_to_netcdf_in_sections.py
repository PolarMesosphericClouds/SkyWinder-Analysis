import netCDF4
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app

from skywinder_analysis.lib.tools import blosc_file as blosc
from skywinder_analysis.lib.image_processing import flat_field as ff
from skywinder_analysis.lib.image_processing import flat_field_piggyback as ffpb
from skywinder_analysis.lib.image_processing import binning
from skywinder_analysis.lib.tools import periodic_image_finder as pif
from skywinder_analysis.lib.tools import GPS_pointing_and_sun as GPS

import numpy as np
import pandas as pd
from datetime import datetime
import os


def make_datetime_from_timestamp(timestamp):
    year_month_day = datetime.utcfromtimestamp(timestamp).strftime('%Y%m%d')
    hour_minute_second = datetime.utcfromtimestamp(timestamp).strftime('%H%M%S')
    return year_month_day, hour_minute_second


class NetCDFConversionApp(skywinder_analysis_app.App):
    def create_netcdfs(self):
        self.create_output()
        subset_starts = np.arange(self.settings.start, self.settings.stop, self.settings.section_size) 
        for start in subset_starts:
            self.create_netcdf(start, start+self.settings.section_size)

    def create_netcdf(self, start, stop):
        self.logger.info('Creating container classes')
        bin_size = self.settings.bin_size
        Times = np.arange(start, stop, self.settings.interval)
        N = len(Times)
        n = len(self.settings.camera_numbers)
        Img_data = np.zeros((N, n, int(3232/bin_size), int(4864/bin_size)))
        Lats = np.zeros(N)
        Lons = np.zeros(N)
        Alts = np.zeros(N)
        Azs = np.zeros(N)
        Exposures = np.zeros((N,n))
        latlonalt_df = pd.read_csv('/data/mounted_filesystems/nas2/resources/SIP_gps_plus_solar_data.csv')


        for j in range(n):
            fns = pif.find_periodic_images_full_fn(self.settings.camera_numbers[j], start, stop, self.settings.interval)
            self.logger.info('%d times, %d fns, camera %d' % (N, len(fns), self.settings.camera_numbers[j]))
            self.logger.info('From %s to %s' % (fns[0], fns[-1]))
            for i in range(N):
                fn = fns[i]
                ts = int(fn.split('=')[-1][:10])
                if self.settings.piggyback:
                    if bin_size > 1:
                        img_data = binning.bucket(ffpb.get_final_cleaned_image(fn, self.settings.new_flat_field_window), (bin_size, bin_size))
                    else:
                        img_data = ffpb.get_final_cleaned_image(fn, 600)
                else:
                    if bin_size > 1:
                        img_data = binning.bucket(ff.get_final_cleaned_image(fn, self.settings.new_flat_field_window), (bin_size, bin_size))
                    else:
                        img_data = ff.get_final_cleaned_image(fn, 600)
                _, chunk = blosc.load_blosc_image(fn)
                lat_value, lon_value, alt_value = GPS.get_lat_long_alt_SIP(ts, df=latlonalt_df)
                az_value = GPS.get_pointing_rotator(ts)
                Img_data[i,j,:,:] = img_data
                if j == 0:
                    Lats[i] = lat_value
                    Lons[i] = lon_value
                    Alts[i] = alt_value*0.3048 #Convert to meters from feet  #/3280.
                    Azs[i] = az_value
                Exposures[i,j] = chunk[0][6]
                print(ts)

        ymd, hms_start = make_datetime_from_timestamp(start)
        _, hms_stop = make_datetime_from_timestamp(stop)
        netcdf_fn = 'pmc-turbo_l1_corrected-binned-images_%st%s_v01.nc' % (ymd, hms_start)
        outpath = os.path.join(self.out_path, netcdf_fn)
        dataset = netCDF4.Dataset(outpath, "w", format="NETCDF3_CLASSIC")

        dataset.createDimension('x_pixel', int(3232/bin_size))
        dataset.createDimension('y_pixel', int(4864/bin_size))
        dataset.createDimension('Times', None)
        dataset.createDimension('Camera_numbers', n)

        Imgs = dataset.createVariable('Imgs', 'i2', ('Times', 'Camera_numbers', 'x_pixel', 'y_pixel'))
        Time = dataset.createVariable('Time', 'f8', ('Times'))
        Az = dataset.createVariable('Azimuth', 'f4', ('Times'))
        Exposure = dataset.createVariable('Exposure', 'f4', ('Times', 'Camera_numbers'))
        Camera_number = dataset.createVariable('Camera_numbers', 'i2', ('Camera_numbers'))
        X_pixel = dataset.createVariable('x_pixel_index', 'i2', ('x_pixel'))
        Y_pixel = dataset.createVariable('y_pixel_index', 'i2', ('y_pixel'))

        Lat = dataset.createVariable('Latitude', 'f4', ('Times'))
        Lon = dataset.createVariable('Longitude', 'f4', ('Times'))
        Alt = dataset.createVariable('Altitude', 'f4', ('Times'))

        Imgs.UNITS = 'Uncalibrated brightness units'
        Imgs.CATDESC = 'PMC Turbo optical images. Taken with Kodak 16070 CCD and Hoya #25A red filter.'
        Imgs.DISPLAY_TYPE = 'image'
        Imgs.FIELDNAM = 'Images'
        Imgs.VAR_TYPE = 'data'
        Imgs.DEPEND_0 = 'Time'
        Imgs.DEPEND_1 = 'Camera_numbers'
        Imgs.DEPEND_2 = 'x_pixel_index'
        Imgs.DEPEND_3 = 'y_pixel_index'
        Imgs.FILLVAL = np.array([-32768], dtype=np.int16)[0]
        Imgs.VALIDMIN = np.array([-32767], dtype=np.int16)[0]
        Imgs.VALIDMAX = np.array([32767], dtype=np.int16)[0]
        Imgs.FORMAT ='I6'
        Imgs.LABL_PTR_1 = 'Height'
        Imgs.LABL_PTR_2 = 'Width'
            

        Time.UNITS = 'Seconds since 1970-01-01T00:00:00Z'
        Time.CATDESC = 'Unix time - seconds since 1970-01-01 00:00:00'
        Time.DISPLAY_TYPE = 'no_plot'
        Time.FIELDNAM = 'Timestamp'
        Time.VAR_TYPE = 'support_data'
        Time.FILLVAL = np.array([-1.0E31], dtype=np.float32)[0]
        Time.VALIDMIN = np.array([0.], dtype=np.float32)[0]
        Time.VALIDMAX = np.array([1.0E31], dtype=np.float32)[0]
        Time.FORMAT ='F12.2'
        Time.LABLAXIS = 'Time'
        

        Az.UNITS = 'Radians'
        Az.CATDESC = 'Radians CW from north (as viewed from above)'
        Az.DISPLAY_TYPE = 'time_series'
        Az.FIELDNAM = 'Azimuth'
        Az.VAR_TYPE = 'data'
        Az.DEPEND_0 = 'Time'
        Az.FILLVAL = np.array([-1.0E31], dtype=np.float32)[0]
        Az.VALIDMIN = np.array([0.], dtype=np.float32)[0]
        Az.VALIDMAX = np.array([6.283185], dtype=np.float32)[0]
        Az.FORMAT ='F5.2'
        Az.LABLAXIS = 'Azimuth'

        Exposure.UNITS = 'microseconds'
        Exposure.CATDESC = 'Exposure time of image'
        Exposure.DISPLAY_TYPE = 'time_series'
        Exposure.FIELDNAM = 'Exposure Time'
        Exposure.VAR_TYPE = 'data'
        Exposure.DEPEND_0 = 'Time'
        Exposure.DEPEND_1 = 'Camera_numbers'
        Exposure.FILLVAL = np.array([-1.0E31], dtype=np.float32)[0]
        Exposure.VALIDMIN = np.array([0.], dtype=np.float32)[0]
        Exposure.VALIDMAX = np.array([2.e7], dtype=np.float32)[0]
        Exposure.FORMAT = 'F7.0'
        Exposure.LABLAXIS = 'Exposure Time'

        Lat.UNITS = 'degrees'
        Lat.CATDESC = 'Latitude of gondola'
        Lat.DISPLAY_TYPE = 'time_series'
        Lat.FIELDNAM = 'Latitude'
        Lat.VAR_TYPE = 'data'
        Lat.DEPEND_0 = 'Time'
        Lat.FILLVAL = np.array([-1.0E31], dtype=np.float32)[0]
        Lat.VALIDMIN = np.array([-90.], dtype=np.float32)[0]
        Lat.VALIDMAX = np.array([90], dtype=np.float32)[0]
        Lat.FORMAT ='F3.2'
        Lat.LABLAXIS = 'Latitude'

        Lon.UNITS = 'degrees'
        Lon.CATDESC = 'Longitude of gondola'
        Lon.DISPLAY_TYPE = 'time_series'
        Lon.FIELDNAM = 'Longitude'
        Lon.VAR_TYPE = 'data'
        Lon.DEPEND_0 = 'Time'
        Lon.FILLVAL = np.array([-1.0E31], dtype=np.float32)[0]
        Lon.VALIDMIN = np.array([-180.], dtype=np.float32)[0]
        Lon.VALIDMAX = np.array([180], dtype=np.float32)[0]
        Lon.FORMAT ='F3.2'
        Lon.LABLAXIS = 'Longitude'

        Alt.UNITS = 'meters'
        Alt.CATDESC = 'Altitude of gondola'
        Alt.DISPLAY_TYPE = 'time_series'
        Alt.FIELDNAM = 'Altitude'
        Alt.VAR_TYPE = 'data'
        Alt.DEPEND_0 = 'Time'
        Alt.FILLVAL = np.array([-1.0E31], dtype=np.float32)[0]
        Alt.VALIDMIN = np.array([0.], dtype=np.float32)[0]
        Alt.VALIDMAX = np.array([50000], dtype=np.float32)[0]
        Alt.FORMAT ='F5.2'
        Alt.LABLAXIS = 'Longitude'

        Camera_number.UNITS = ' '
        Camera_number.CATDESC = ('Cameras numbered 1 to 3 right-to-left (narrow FOV)' +
                               ' and 4 to 7 left-to-right (wide FOV). See Fritts 2019' +
                               'overview paper for more details')
        Camera_number.DISPLAY_TYPE = 'no_plot'
        Camera_number.FIELDNAM = 'Camera numbers'
        Camera_number.VAR_TYPE = 'support_data'
        Camera_number.FILLVAL = np.array([-32768], dtype=np.int16)[0]
        Camera_number.VALIDMIN = np.array([1], dtype=np.int16)[0]
        Camera_number.VALIDMAX = np.array([99], dtype=np.int16)[0]
        Camera_number.FORMAT ='I2'
        Camera_number.LABLAXIS = 'Camera Number'


        X_pixel.UNITS = ' '
        X_pixel.CATDESC = ('pixel indices x axis')
        X_pixel.DISPLAY_TYPE = 'no_plot'
        X_pixel.FIELDNAM = 'x pixel'
        X_pixel.VAR_TYPE = 'support_data'
        X_pixel.FILLVAL = np.array([-32768], dtype=np.int16)[0]
        X_pixel.VALIDMIN = np.array([1], dtype=np.int16)[0]
        X_pixel.VALIDMAX = np.array([int(3232/bin_size)], dtype=np.int16)[0]
        X_pixel.FORMAT ='I2'
        X_pixel.LABLAXIS = 'X Pixel'


        Y_pixel.UNITS = ' '
        Y_pixel.CATDESC = ('pixel indices y axis')
        Y_pixel.DISPLAY_TYPE = 'no_plot'
        Y_pixel.FIELDNAM = 'y pixel'
        Y_pixel.VAR_TYPE = 'support_data'
        Y_pixel.FILLVAL = np.array([-32768], dtype=np.int16)[0]
        Y_pixel.VALIDMIN = np.array([1], dtype=np.int16)[0]
        Y_pixel.VALIDMAX = np.array([int(4864/bin_size)], dtype=np.int16)[0]
        Y_pixel.FORMAT ='I2'
        Y_pixel.LABLAXIS = 'Y Pixel'

        dataset.Source_name = netCDF4.chartostring(netCDF4.stringtochar(np.array('PMC Turbo>Polar Mesospheric Cloud Turbulence', dtype='S')))
        dataset.Data_type = netCDF4.chartostring(netCDF4.stringtochar(np.array('L1>Corrected Binned Images', dtype='S')))
        dataset.Descriptor = netCDF4.chartostring(netCDF4.stringtochar(np.array('PMC Cameras', dtype='S')))
        dataset.Data_version = netCDF4.chartostring(netCDF4.stringtochar(np.array('1', dtype='S')))
        dataset.Logical_file_id = netCDF4.chartostring(netCDF4.stringtochar(np.array('pmc_turbo_l1_images', dtype='S')))
        dataset.PI_name = netCDF4.chartostring(netCDF4.stringtochar(np.array('David Fritts', dtype='S')))
        dataset.PI_affiliation = netCDF4.chartostring(netCDF4.stringtochar(np.array('GATS inc', dtype='S')))
        dataset.TEXT = netCDF4.chartostring(netCDF4.stringtochar(np.array('', dtype='S')))
        dataset.Instrument_type = netCDF4.chartostring(netCDF4.stringtochar(np.array('Imaging and Remote Sensing (ITM/Earth)', dtype='S')))
        dataset.Discipline = netCDF4.chartostring(netCDF4.stringtochar(np.array('Heliophysics>Mesosphere and Thermosphere Dynamics', dtype='S')))
        dataset.Logical_source = netCDF4.chartostring(netCDF4.stringtochar(np.array('pmc-turbo_l1_corrected_binned_images', dtype='S')))
        dataset.Logical_source_description = netCDF4.chartostring(netCDF4.stringtochar(np.array('PMC Turbo corrected and binned images', dtype='S')))
        dataset.Acknowledgement = netCDF4.chartostring(netCDF4.stringtochar(np.array('', dtype='S')))
        dataset.LINK_TEXT = netCDF4.chartostring(netCDF4.stringtochar(np.array('PMC Turbo experiment described in the', dtype='S')))
        dataset.LINK_TITLE = netCDF4.chartostring(netCDF4.stringtochar(np.array('PMC Turbo overview paper', dtype='S')))
        dataset.HTTP_LINK = netCDF4.chartostring(netCDF4.stringtochar(np.array('https://doi.org/10.1029/2019JD030298', dtype='S')))
        dataset.Mission_group = netCDF4.chartostring(netCDF4.stringtochar(np.array('Long Distance Balloon missions', dtype='S')))

        dataset.summary = netCDF4.chartostring(netCDF4.stringtochar(np.array('PMC-Turbo images, flatfielded, with 10-minute moving window subtracted. Downsampled with 4x4 pixel binning', dtype='S')))

        X_pixel[:] = np.arange(1, int(3232/bin_size)+1, 1)
        Y_pixel[:] = np.arange(1, int(4864/bin_size)+1, 1)
        Imgs[:,:,:,:] = Img_data.astype(np.int16)
        Time[:] = Times
        Az[:] = Azs
        Exposure[:,:] = Exposures
        Camera_number[:] = self.settings.camera_numbers
        Lat[:] = Lats
        Lon[:] = Lons
        Alt[:] = Alts
        dataset.close()
        return


if __name__ == "__main__":
    import warnings; warnings.simplefilter('ignore')
    app =  NetCDFConversionApp()
    app.create_netcdfs()
