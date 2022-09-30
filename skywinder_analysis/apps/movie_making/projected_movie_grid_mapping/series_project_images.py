from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import binning
from skywinder_analysis.lib.image_processing import flat_field, flat_field_piggyback
import matplotlib.animation as animation
import scipy.stats
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from skywinder_analysis.lib.image_processing import stitching
import os
import sys
import datetime
import time
import pandas as pd
from scipy import interpolate
from skywinder_analysis.lib.units import coordinate_transforms


class img_container_series():
    def __init__(self, cam_num=0):
        self.cam_num = cam_num
        self.ff = False
        self.imgs = []
        self.fh = False
        self.fw = False


class GridMapMovieApp(skywinder_analysis_app.App):
    def project_images(self):
        self.create_output()
        self.logger.info('Creating container classes')
        frames = len(self.settings.all_filename_lists[self.settings.cam_nums[0]])
        ics = {}
        global_index = 0
        for cam_num in self.settings.cam_nums:
            ic = img_container_series(cam_num=cam_num)

            for fn in self.settings.all_filename_lists[cam_num]:
                if self.settings.use_new_flat_fields:
                    if self.settings.piggyback:
                        img = flat_field_piggyback.get_final_cleaned_image(fn, self.settings.new_flat_field_window)
                        img = binning.bucket(img, (self.settings.bin_factor, self.settings.bin_factor))
                        ic.imgs.append(img)
                    else:
                        img = flat_field.get_final_cleaned_image(fn, self.settings.new_flat_field_window)
                        img = binning.bucket(img, (self.settings.bin_factor, self.settings.bin_factor))
                        ic.imgs.append(img)
                elif self.settings.flat_field_filenames:
                    self.logger.info('Generating flat field from %d files' % len(self.settings.flat_field_filenames[cam_num]))
                    flat_field_image = flat_field.generate_flat_field_image_from_filenames(self.settings.flat_field_filenames[cam_num])
                    img, _ = blosc_file.load_blosc_image(fn)
                    img = flat_field.apply_flat_field(img, flat_field_image)
                    img = binning.bucket(img, (self.settings.bin_factor, self.settings.bin_factor))
                    ic.imgs.append(img)
                    # Generate static flat_field_image from filenames
                else:
                    self.logger.info('Using no flat field')
                    img, _ = blosc_file.load_blosc_image(fn)
                    img = binning.bucket(img, (self.settings.bin_factor, self.settings.bin_factor))
                    ic.imgs.append(img)
            ics[cam_num] = ic

        if not self.settings.provided_grid_map:
            self.logger.info('Building interpolation functions')
            for cam_num in self.settings.cam_nums:
                if self.settings.piggyback:
                    df = pd.read_csv(
                                     os.path.join(self.settings.pointing_directory,
                                                  'c99_1579217700_solution.csv'))
                else:
                    df = pd.read_csv(
                                     os.path.join(self.settings.pointing_directory,
                                     ('c%d_2018-07-12_2330_solution.csv' % cam_num)))
                f_h = interpolate.interp2d(df.az, df.alt, df.h,
                                           kind='cubic', bounds_error=False, fill_value=np.nan)
                f_w = interpolate.interp2d(df.az, df.alt, df.w,
                                           kind='cubic', bounds_error=False, fill_value=np.nan)
                ics[cam_num].fh = f_h
                ics[cam_num].fw = f_w

            def altaz_to_pixel(alt, az, cam_num, verbose=False):
                h = ics[cam_num].fh(az, alt)
                w = ics[cam_num].fw(az, alt)
                try:
                    h = int(np.rint(h))
                    w = int(np.rint(w))
                except ValueError as e:
                    return False
                if verbose:
                    print('H, W:', h, w)
                if h > 3231:
                    return False
                if w > 4863:
                    return False
                if h < 0:
                    return False
                if w < 0:
                    return False
                else:
                    return h, w

            def get_value(x, y, bin_factor=1, verbose=False):
                alt, az = coordinate_transforms.cart_to_altaz(x, y, centerline=self.settings.centerline, verbose=verbose)
                for cam_num in self.settings.cam_nums:
                    coords = altaz_to_pixel(alt, az, cam_num, verbose=verbose)
                    if coords == False:
                        continue
                    else:
                        h, w = coords
                        return h // bin_factor, w // bin_factor, cam_num
                return -1, -1, -1

            hs = []
            ws = []
            cams = []
            for i in self.settings.xvalues:
                for j in self.settings.yvalues:
                    h, w, cam = get_value(i, j, bin_factor=self.settings.bin_factor)
                    hs.append(h)
                    ws.append(w)
                    cams.append(cam)

            map_df = pd.DataFrame(data={'h':hs, 'w':ws, 'cam':cams})
            map_df.to_csv(os.path.join(self.out_path, 'map_df.csv'))
        else:
            map_df = pd.read_csv(self.settings.provided_grid_map)

        xmin = min(self.settings.xvalues)
        xmax = max(self.settings.xvalues)
        ymin = min(self.settings.yvalues)
        ymax = max(self.settings.yvalues)

        vmins = []
        #vmaxs = []
        #for cam_num in self.settings.cam_nums:
        #    vmin = scipy.stats.scoreatpercentile(
        #        ics[cam_num].imgs[0][self.settings.level_region[0]:self.settings.level_region[1],
        #        self.settings.level_region[2]:self.settings.level_region[3]]
        #        , .01)
        #    vmax = scipy.stats.scoreatpercentile(
        #        ics[cam_num].imgs[0][self.settings.level_region[0]:self.settings.level_region[1],
        #        self.settings.level_region[2]:self.settings.level_region[3]]
        #        , 99.9)
        #    vmins.append(vmin)
        #    vmaxs.append(vmax)

        #vmin = min(vmins)
        #vmax = max(vmaxs)


        imd = np.zeros((len(self.settings.xvalues), len(self.settings.yvalues)))
        imd = np.rot90(imd, k=1)

        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        #im = ax.imshow(imd, vmin=vmin, vmax=vmax, cmap=self.settings.color_scheme,
        #               origin='lower', extent=[xmin, xmax, ymax, ymin])

        im = ax.imshow(imd, cmap=self.settings.color_scheme,
                       origin='lower', extent=[xmin, xmax, ymax, ymin])

        if self.settings.wireframes:
            def altaz_to_cart(alt, az, h=(83-38)):
                theta = 90-alt # Zero degrees is straight up
                phi = az - 100 # 100 degrees is roughly center of these particular solutions.
                l = h*np.tan(np.deg2rad(theta))
                y = l * np.cos(np.deg2rad(phi))
                x = l * np.sin(np.deg2rad(phi))
                return x, y
            # Narrow field wireframes
            for cam_num in ['1','2','3']:
                df = pd.read_csv(self.settings.narrowfield_wireframe_solution_directories[narrow_cam_num])
                for sub_df in [df.loc[df.h == 0],
                       df.loc[df.h == 3232],
                       df.loc[df.w == 4864],
                       df.loc[df.w == 0]]:
                    x_km = []
                    y_km = []
                    for az, alt in zip(sub_df.az, sub_df.alt):
                        x, y = altaz_to_cart(alt, az)
                        x_km.append(x)
                        y_km.append(y)
                    ax.plot(x_km, y_km, '-', color='k')

        #ax.set_aspect('equal')
        #ax.set_axisbelow(True)
        if self.settings.grid:
            #ax.set_xlabel('x (km)')
            #ax.set_ylabel('y (km)')
            ax.xaxis.set_major_locator(plt.MultipleLocator(self.settings.grid_spacing))
            ax.yaxis.set_major_locator(plt.MultipleLocator(self.settings.grid_spacing))
            ax.grid(True)
        else:
            ax.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        vsize = imd.shape[0] / self.settings.dpi
        hsize = imd.shape[1] / self.settings.dpi
        fig.set_size_inches([hsize, vsize])


        xfig, yfig = fig.get_dpi() * fig.get_size_inches()
        fontsize = yfig / 24
        if self.settings.timestamp_in_image:
            time_txt = ax.text(0.05, .95, 'time:', ha='left', va='top', color=self.settings.text_color,
                               fontsize=fontsize, transform=ax.transAxes)
        else:
            time_txt = False
        start_at = time.time()
        def update_img(index):
            elapsed = time.time() - start_at
            if index:
                time_per_frame = elapsed / index
            else:
                time_per_frame = 1
            sys.stdout.flush()

            def map_image(i, j):
                #do indexing to get cam, h, and w from df
                i = int(i)
                j = int(j) 
                coords = j + (i*len(self.settings.yvalues))
                entry = map_df.iloc[coords]
                if entry.cam == -1:
                    return np.nan
                else:
                    return ics[entry.cam].imgs[index][entry.h][entry.w]

            vector_map_image = np.vectorize(map_image)
            self.logger.info('Writing frame %d' % index)
            stretched_image = np.fromfunction(vector_map_image,
                                              (len(self.settings.xvalues), len(self.settings.yvalues)))
            stretched_image = np.rot90(stretched_image, k=1)

            filename_for_date = self.settings.all_filename_lists[self.settings.cam_nums[0]][index]
            unix_timestamp_s = int(filename_for_date.split('=')[-1][:10])
            dt = datetime.datetime.utcfromtimestamp(unix_timestamp_s)
            date_string = dt.strftime('%m-%d_%H:%M:%S')

            if self.settings.timestamp_in_image:
                time_txt.set_text(date_string)

            if True:
                vmin = np.nanpercentile(stretched_image, .01)
                vmax = np.nanpercentile(stretched_image, 99.99)
                print(vmin, vmax)
            im.set_data(stretched_image)
            im.set_clim(vmin=vmin, vmax=vmax)

            if self.settings.individual_frames:
                fig_fn = os.path.join(self.out_path, ('frame_%d_%s.png' % (n, date_string)))
                plt.savefig(fig_fn)
                self.logger.info('Frame saved as %s' % fig_fn)
            self.logger.info("\r%d of %d %.1f minutes elapsed, %.1f minutes remaining" % (index, frames, elapsed/60, (frames-index) * time_per_frame / 60))
            return im, time_txt

        if not self.settings.individual_frames:
            ani = animation.FuncAnimation(fig, update_img, frames, interval=500)
            ani_fn = os.path.join(self.out_path, 'movie.mp4')
            ani.save(ani_fn, writer=self.settings.writer, dpi=self.settings.dpi)
            return ani
        else:
            for n in range(frames):
                update_img(n)
            return True


        return ani

if __name__ == "__main__":
    app = GridMapMovieApp()
    app.project_images()
