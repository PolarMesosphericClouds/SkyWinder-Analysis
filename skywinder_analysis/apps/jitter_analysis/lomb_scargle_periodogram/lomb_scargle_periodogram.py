from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app
import glob
import os
import datetime
import scipy.stats
import scipy.signal
import scipy.interpolate
import pandas as pd
from skywinder_analysis.lib.image_processing import flat_field, binning
from skywinder_analysis.lib.pointing import blob_tools, blobs
from skywinder_turbo.camera.image_processing import hot_pixels
import matplotlib.pyplot as plt
import numpy as np


class img_container_series():
    def __init__(self, cam_num=0):
        self.cam_num = cam_num
        self.ff = False
        self.imgs = []
        self.fh = False
        self.fw = False
        self.blobs = []
        self.timestamps_ms = []


class blob_trajectory():
    def __init__(self, x, y, timestamp, f_az, f_alt, update_index):
        self.x = []
        self.y = []
        self.alt = []
        self.az = []
        self.delta_x = []
        self.delta_y = []
        self.delta_alt = []
        self.delta_az = []
        self.last_update = 0
        self.length = 0
        self.x.append(x)
        self.y.append(y)
        alt = f_alt(x,y)[0]
        az = f_az(x,y)[0]
        self.alt.append(alt)
        self.az.append(az)
        self.timestamps = []
        self.timestamps.append(timestamp)
        self.delta_x.append(0)
        self.delta_y.append(0)
        self.delta_alt.append(0)
        self.delta_az.append(0)
        self.last_update = update_index
        self.length += 1
        
    def update(self, x, y, timestamp, f_az, f_alt, update_index):
        self.delta_x.append(x - self.x[-1])
        self.delta_y.append(y - self.y[-1])
        alt = f_alt(x,y)[0]
        az = f_az(x,y)[0]
        self.delta_alt.append(alt - self.alt[-1])
        self.delta_az.append(az - self.az[-1])
        self.x.append(x)
        self.y.append(y)
        self.alt.append(alt)
        self.az.append(az)
        self.timestamps.append(timestamp)
        self.last_x = x
        self.last_y = y
        self.last_update = update_index
        self.length += 1


def group_blob_trajectories(image_container, f_az, f_alt, min_distance=10):
    blob_trajectories = []
    for n in range(len(image_container.blobs)):
        if n == 0:
            # Initialize
            df = image_container.blobs[n]
            ts = image_container.timestamps_ms[n]
            for i in range(len(df)):
                blob_trajectories.append(blob_trajectory(df.iloc[i].x, df.iloc[i].y, ts, f_az, f_alt, n))
        else:
            df = image_container.blobs[n]
            ts = image_container.timestamps_ms[n]
            for b_t in blob_trajectories:
                distance = np.sqrt(np.square(b_t.x[-1] - df.x) + np.square(b_t.y[-1] - df.y))
                if distance.min() < min_distance:
                    closest = df.iloc[distance.argsort()[0]]
                    b_t.update(closest.x, closest.y, ts, f_az, f_alt, n)
    return blob_trajectories    


class LSPeriodogramApp(skywinder_analysis_app.App):
    def get_periodograms(self):
        self.create_output()
        self.logger.info('Creating container classes')
        for fns in self.settings.all_filename_lists:
            ic = img_container_series(cam_num=self.settings.camera_number)
            for fn in fns:
                img = flat_field.get_final_cleaned_image(fn, self.settings.new_flat_field_window)
                ic.imgs.append(img)
                # Get blobs
                myblobs = self.get_blobs_from_img(img, 4, show_plot=False,blob_threshold=8,
                                             kernel_size=8, kernel_sigma=2,fitting_region_size=16,
                                             fit_blobs=True, cell_size=32, max_blobs_per_cell=2, rmax=4,
                                             low_sigma_filter=None)
                # Switch blobs to pandas for convenience
                ic.blobs.append(pd.DataFrame([vars(b) for b in myblobs]))
                ic.timestamps_ms.append(int(fn[-19:-6])) # get timestamp in ms from filename
            df = pd.read_csv(self.settings.pointing_solutions_dir)
            f_alt = scipy.interpolate.interp2d(df.h, df.w, df.alt, kind='cubic', bounds_error=False, fill_value=np.nan)
            f_az = scipy.interpolate.interp2d(df.h, df.w, df.az, kind='cubic', bounds_error=False, fill_value=np.nan)
            blob_trajectories = group_blob_trajectories(ic, f_az, f_alt)
            followed_stars = [bt for bt in blob_trajectories if bt.length >= self.settings.min_blob_trajectory_length]
            self.logger.info('%d stars followed.' % len(followed_stars))
            if len(followed_stars) == 0:
                self.logger.info('No stars sufficiently followed.')
            else:
                f = np.linspace(0.01, 2,1000)
                alt_periodograms = []
                az_periodograms = []
                fig, axs = plt.subplots(2, 1, figsize=(9,9))
                _axs = axs.flatten()
                for followed_star in followed_stars:
                    y = np.array(followed_star.alt)
                    t = (np.array(followed_star.timestamps)-followed_star.timestamps[0])/1000.
                    linear_fit = np.polyfit(t, y, 1)
                    fit = linear_fit[0]*t + linear_fit[1]
                    pgram = scipy.signal.lombscargle(t, y-fit, f, normalize=True)
                    alt_periodograms.append(pgram)

                    y = np.array(followed_star.az)
                    linear_fit = np.polyfit(t, y, 1)
                    fit = linear_fit[0]*t + linear_fit[1]
                    pgram = scipy.signal.lombscargle(t, y-fit, f, normalize=True)
                    az_periodograms.append(pgram)
                _axs[0].errorbar(f/(2.0*np.pi),
                            np.mean(np.stack(alt_periodograms), axis=0),
                            np.std(np.stack(alt_periodograms), axis=0),
                            fmt='-', color='blue', ecolor='black', capsize=0,
                           )
                _axs[0].set_title('Alt')
                _axs[1].errorbar(f/(2.0*np.pi),
                            np.mean(np.stack(az_periodograms), axis=0),
                            np.std(np.stack(az_periodograms), axis=0),
                            fmt='-', color='blue', ecolor='black', capsize=0,
                           )
                _axs[1].set_title('Az')
                for ax in axs:
                    ax.grid(True)
                    ax.set_xlim(0, .25)
                    ax.set_ylim(0)
                    ax.set_xlabel('frequency (Hz)')
                    ax.set_ylabel('Lomb-Scargle Power')
                fn_ts = fns[0].split('/')[-1].split('=')[-1][:10]
                ts_utc = datetime.datetime.utcfromtimestamp(int(fn_ts))
                ts_string = ts_utc.strftime('%Y-%m-%d_%H:%M')
                title = ts_string + ('+%d seconds' % self.settings.duration) + (' %d stars followed'%len(followed_stars))
                fig.suptitle(title)
                fig_fn = os.path.join(self.out_path, '%s.png'%ts_string)
                plt.savefig(fig_fn)
        return alt_periodograms, az_periodograms

    def get_blobs_from_img(self, img, cam_number, show_plot=True,
                           blob_threshold=6, kernel_size=8, kernel_sigma=2,
                           fitting_region_size=32, fit_blobs=True,cell_size=128,
                           max_blobs_per_cell=1, rmax=4,
                           low_sigma_filter=0.5, high_sigma_filter=10):
        hpm = hot_pixels.HotPixelMasker(np.load(self.settings.hot_pixel_dirs[cam_number]), self.settings.image_shape)
        img = hpm.process(img)
        b = blobs.BlobFinder(img, blob_threshold, kernel_size, kernel_sigma,
                             fitting_region_size, fit_blobs,cell_size,
                             max_blobs_per_cell, rmax)
        my_blobs_list = b.blobs
        if not low_sigma_filter:
            good_blobs = my_blobs_list
        else:
            good_blobs = [blob for blob in my_blobs_list if blob.sigma > low_sigma_filter]
            good_blobs = [blob for blob in good_blobs if blob.sigma < high_sigma_filter]
        if show_plot:
            show_blobs(img, good_blobs)
        self.logger.info('Found %d good blobs.' % len(good_blobs))
        return good_blobs


if __name__ == "__main__":
    import warnings; warnings.simplefilter('ignore')
    app = LSPeriodogramApp()
    app.get_periodograms()
