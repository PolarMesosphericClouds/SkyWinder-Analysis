from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import binning
from skywinder_analysis.lib.image_processing import flat_field
import scipy.stats
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app
import matplotlib.pyplot as plt
import numpy as np
from skywinder_analysis.lib.image_processing import stitching
import os
import pandas as pd
from scipy import interpolate
from skywinder_analysis.lib.units import coordinate_transforms


class img_container():
    def __init__(self, cam_num=0):
        self.cam_num = cam_num
        self.ff = False
        self.img_array = False
        self.fh = False
        self.fw = False


class ImageProjectionApp(skywinder_analysis_app.App):
    def project_images(self):
        self.create_output()
        self.logger.info('Creating container classes')
        ics = {}
        for cam_num in self.settings.cam_nums:
            ic = img_container(cam_num=cam_num)
            if self.settings.naive_flat_field:
                self.logger.info('Using naive flat field')
                ff = flat_field.generate_flat_field_image_from_filenames(
                    self.settings.all_flat_field_filenames[cam_num])
                img, _ = blosc_file.load_blosc_image(self.settings.all_filenames[cam_num])
                img = flat_field.apply_flat_field(img, ff)
            if self.settings.new_flat_field:
                img = flat_field.get_final_cleaned_image(self.settings.all_filenames[cam_num], self.settings.new_flat_field_window)
            img = binning.bucket(img, (self.settings.bin_factor, self.settings.bin_factor))
            ic.img_array = img
            ics[cam_num] = ic

        if self.settings.relevel_images:
            self.logger.info('Releveling images')
            # Relevel images against each other
            for cam_num in self.settings.cam_nums:
                mean = np.mean(ics[cam_num].img_array)
                ics[cam_num].img_array = ics[cam_num].img_array / mean

        self.logger.info('Building stitched image')
        full_fov_img = stitching.build_stitched_image([ics[key].img_array for key in ics.keys()])

        self.logger.info('Building interpolation functions')
        for cam_num in self.settings.cam_nums:
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

        @np.vectorize
        def get_value(x, y, bin_factor=1, verbose=False):
            alt, az = coordinate_transforms.cart_to_altaz(x, y, verbose=verbose)
            for cam_num in self.settings.cam_nums:
                coords = altaz_to_pixel(alt, az, cam_num, verbose=verbose)
                if coords == False:
                    continue
                else:
                    h, w = coords
                    return ics[cam_num].img_array[h // bin_factor][w // bin_factor]
            return np.nan

        xvalues = np.linspace(self.settings.min_x, self.settings.max_x, num=self.settings.x_increment)
        yvalues = np.linspace(self.settings.min_y, self.settings.max_y, num=self.settings.y_increment)
        xmin = min(xvalues)
        xmax = max(xvalues)
        ymin = min(yvalues)
        ymax = max(yvalues)

        xx, yy = np.meshgrid(xvalues, yvalues)

        self.logger.info('Started stretched image generation')
        stretched_image = get_value(xx, yy, bin_factor=self.settings.bin_factor)

        vmin = scipy.stats.scoreatpercentile(full_fov_img, 0.1)
        vmax = scipy.stats.scoreatpercentile(full_fov_img, 99.9)
        print('Color limits:', vmin, vmax)

        if self.settings.show_stitched:
            fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})
            axs[1].imshow(stretched_image, vmin=vmin, vmax=vmax, cmap=self.settings.color_scheme,
                          origin='upper', extent=[xmin, xmax, ymax, ymin])
            axs[1].set_xlabel('x (km)')
            axs[1].set_ylabel('y (km)')

            axs[1].xaxis.set_major_locator(plt.MultipleLocator(50))
            axs[1].xaxis.set_minor_locator(plt.MultipleLocator(25))
            axs[1].yaxis.set_major_locator(plt.MultipleLocator(50))
            axs[1].yaxis.set_minor_locator(plt.MultipleLocator(25))
            axs[1].set_aspect('equal')
            axs[1].set_axisbelow(True)
            axs[1].grid(True)
            axs[1].set_title('Projected Flatfielded Images')
            axs[0].imshow(full_fov_img, origin='upper', vmin=vmin, vmax=vmax, cmap=self.settings.color_scheme)
            axs[0].get_xaxis().set_visible(False)
            axs[0].get_yaxis().set_visible(False)
            axs[0].set_title('Flatfielded Images Side-by-Side')
            axs[0].set_aspect('equal')
            fig.set_size_inches([12, 11])
        else:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(stretched_image, vmin=vmin, vmax=vmax, cmap=self.settings.color_scheme,
                          origin='upper', extent=[xmin, xmax, ymax, ymin])
            ax.set_xlabel('x (km)')
            ax.set_ylabel('y (km)')
            ax.xaxis.labelpad = -30
            ax.yaxis.labelpad = -25

            ax.xaxis.set_major_locator(plt.MultipleLocator(50))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(25))
            ax.yaxis.set_major_locator(plt.MultipleLocator(50))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(25))
            ax.tick_params(axis='y', direction='in', pad=-20)
            ax.tick_params(axis='x', direction='in', pad=-15)
            ax.set_aspect('equal')
            ax.set_axisbelow(True)
            ax.grid(True)
            fig.set_size_inches([12, 6])

        date_string = self.settings.all_filenames[self.settings.cam_nums[0]].split('/')[-1].split('_')[0]
        time_string = self.settings.all_filenames[self.settings.cam_nums[0]].split('/')[-1].split('_')[1]
        # Switch from Eastern time to UT
        hour = int(time_string[0:2])
        hour += 4
        hour = hour % 24
        title_string = (date_string + ' ' + ('%d' % hour) + ':' + time_string[2:4] + ':' + time_string[4:])
        full_time_string = (date_string + '_' + ('%d' % hour) +  time_string[2:4] +  time_string[4:])
        if self.settings.show_stitched:
            fig.suptitle(title_string)

        fn = full_time_string + self.settings.output_name
        fig.savefig(os.path.join(self.out_path, fn), dpi=self.settings.dpi, bbox_inches='tight')


if __name__ == "__main__":
    app = ImageProjectionApp()
    app.project_images()
