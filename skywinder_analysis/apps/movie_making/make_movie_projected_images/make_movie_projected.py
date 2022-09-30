import os
import sys
import time

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.interpolate import RectBivariateSpline as RBS
from skywinder_analysis.lib.image_projection import pixel_projection_new as pp
from skywinder_analysis.lib.image_projection import pixel_projection_piggyback
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app
from skywinder_analysis.lib.tools import GPS_pointing_and_sun as GPS


class MovieMakerApp(skywinder_analysis_app.App):
    def ani_frame(self):
        self.create_output()
        self.logger.info('Starting animation of frames')
        
        timestamps = np.arange(self.settings.start_time, self.settings.end_time, self.settings.interval)
        t0 = timestamps[-1]
        sun_az, sun_alt = GPS.get_sun_az_alt(t0)
        central_az = sun_az + GPS.get_pointing_rotator(t0)

        X_array = np.zeros(0)       
        Y_array = np.zeros(0)
        brightness = np.zeros(0)

        h = np.arange(0, 3232, 34)
        w = np.arange(0, 4864, 34)

        H = np.arange(0, 3232, self.settings.bin_factor)
        W = np.arange(0, 4864, self.settings.bin_factor)


        if self.settings.piggyback:
            X_array, Y_array = pixel_projection_piggyback.get_x_y_arrays(bin_factor=self.settings.bin_factor,
                                                                         alignment='north', translate=False, current_time=1531350000, reference_time=1531350000,
                                                                         masked=True, clipped=False)
        else:
            X_array, Y_array = pp.get_x_y_arrays(camera_numbers=self.settings.camera_numbers, bin_factor=self.settings.bin_factor,
                                                 alignment='north', translate=False, current_time=1531350000, reference_time=1531350000, clipped=False)

        frames = len(timestamps)
        self.logger.info('%d filenames passed' % frames)

        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        vmin = scipy.stats.scoreatpercentile(brightness, 0.1)
        vmax = scipy.stats.scoreatpercentile(brightness, 99.9)
        scat = ax.scatter(X_array, Y_array, s=2, c=brightness, cmap='jet', alpha=0.8, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_xlim(self.settings.limits[0])
        ax.set_ylim(self.settings.limits[1])
        ax.grid(True)
        start_at = time.time()
        ax = plt.gca()

        def update_img(n):
            sun_az_temp, sun_alt_temp = GPS.get_sun_az_alt(timestamps[n])
            az_offset = sun_az_temp + GPS.get_pointing_rotator(timestamps[n]) - np.pi
            X_offset = pp.get_image_translation_Earth_frame(t0, timestamps[n])
            scaling = 1.0 + X_offset[2]/45.0
            X_array_temp = (X_array*np.cos(az_offset)*scaling
                        + Y_array*np.sin(az_offset)*scaling
                        + X_offset[0])
            Y_array_temp = (-X_array*np.sin(az_offset)*scaling
                        + Y_array*np.cos(az_offset)*scaling
                        + X_offset[1])
            XY_array_temp = np.vstack((X_array_temp, Y_array_temp)).T 
            scat.set_offsets(XY_array_temp)

            if self.settings.piggyback:
                brightness = pixel_projection_piggyback.get_brightness_array(timestamps[n],
                                                                             bin_factor = self.settings.bin_factor, reflection_window = self.settings.reflection_window, clipped=self.settings.clipped)
            else:
                brightness = pp.get_brightness_array(timestamps[n], camera_numbers=self.settings.camera_numbers,
                                                     bin_factor = self.settings.bin_factor, reflection_window = self.settings.reflection_window, clipped=self.settings.clipped)
            scat.set_array(brightness)

            vmin = scipy.stats.scoreatpercentile(brightness, self.settings.min_percentile)
            vmax = scipy.stats.scoreatpercentile(brightness, self.settings.max_percentile)
            scat.set_clim(vmin, vmax)
            scat.set_cmap(cm.jet)

            return scat



        ani = animation.FuncAnimation(fig, update_img, frames, interval=50)

        ani_fn = os.path.join(self.out_path, self.settings.output_name)
        ani.save(ani_fn, writer=self.settings.writer, dpi=self.settings.dpi)
        return ani


if __name__ == "__main__":
    app = MovieMakerApp()
    app.ani_frame()
    app.end()
