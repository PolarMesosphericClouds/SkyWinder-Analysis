from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import binning
from skywinder_analysis.lib.image_processing import flat_field
import scipy.stats
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from skywinder_analysis.lib.image_processing import stitching
import os
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


class SeriesImageProjectionApp(skywinder_analysis_app.App):
    def project_images(self):
        self.create_output()
        self.logger.info('Creating container classes')
        ics = {}
        for cam_num in self.settings.cam_nums:
            ic = img_container_series(cam_num=cam_num)
            if self.settings.naive_flat_field:
                self.logger.info('Using naive flat field')
                ff = flat_field.generate_flat_field_image_from_filenames(
                    self.settings.all_flat_field_filenames[cam_num])
                for fn in self.settings.all_filename_lists[cam_num]:
                    img, _ = blosc_file.load_blosc_image(fn)
                    img = flat_field.apply_flat_field(img, ff)
                    img = binning.bucket(img, (self.settings.bin_factor, self.settings.bin_factor))
                    ic.imgs.append(img)
            if self.settings.new_flat_field:
                self.logger.info('Using sophisticated flat field')
                for fn in self.settings.all_filename_lists[cam_num]:
                    img = flat_field.get_final_cleaned_image(fn, self.settings.new_flat_field_window)
                    img = binning.bucket(img, (self.settings.bin_factor, self.settings.bin_factor))
                    ic.imgs.append(img)
            ics[cam_num] = ic

        if self.settings.relevel_images:
            self.logger.info('Releveling images')
            # Relevel images against each other
            for idx in range(len(ics[self.settings.cam_nums[0]].imgs)):
                for cam_num in self.settings.cam_nums:
                    mean = np.mean(ics[cam_num].imgs[idx])
                    ics[cam_num].imgs[idx] = ics[cam_num].imgs[idx] / mean

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
        def get_value(x, y, idx, bin_factor=1, verbose=False):
            alt, az = coordinate_transforms.cart_to_altaz(x, y, verbose=verbose)
            for cam_num in self.settings.cam_nums:
                coords = altaz_to_pixel(alt, az, cam_num, verbose=verbose)
                if coords == False:
                    continue
                else:
                    h, w = coords
                    return ics[cam_num].imgs[idx][h // bin_factor][w // bin_factor]
            return np.nan

        xvalues = np.linspace(self.settings.min_x, self.settings.max_x, num=self.settings.x_increment)
        yvalues = np.linspace(self.settings.min_y, self.settings.max_y, num=self.settings.y_increment)
        xmin = min(xvalues)
        xmax = max(xvalues)
        ymin = min(yvalues)
        ymax = max(yvalues)

        xx, yy = np.meshgrid(xvalues, yvalues)

        vmins = []
        vmaxs = []
        for cam_num in self.settings.cam_nums:
            vmin = scipy.stats.scoreatpercentile(
                ics[cam_num].imgs[0][self.settings.level_region[0]:self.settings.level_region[1],
                self.settings.level_region[2]:self.settings.level_region[3]]
                , .01)
            vmax = scipy.stats.scoreatpercentile(
                ics[cam_num].imgs[0][self.settings.level_region[0]:self.settings.level_region[1],
                self.settings.level_region[2]:self.settings.level_region[3]]
                , 99.9)
            vmins.append(vmin)
            vmaxs.append(vmax)

        vmin = min(vmins)
        vmax = max(vmaxs)

        for i in range(len(self.settings.all_filename_lists[self.settings.cam_nums[0]])):
            fig, ax = plt.subplots(1, 1)

            self.logger.info('Started stretched image generation')
            stretched_image = get_value(xx, yy, i, bin_factor=self.settings.bin_factor)

            ax.imshow(stretched_image, vmin=vmin, vmax=vmax, cmap=cm.inferno,
                      origin='upper', extent=[xmin, xmax, ymax, ymin])
            ax.set_xlabel('x (km)')
            ax.set_ylabel('y (km)')

            ax.xaxis.set_major_locator(plt.MultipleLocator(50))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(25))
            ax.yaxis.set_major_locator(plt.MultipleLocator(50))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(25))
            ax.set_aspect('equal')
            ax.set_axisbelow(True)
            ax.grid(True)
            ax.set_title('Projected Flatfielded Images')

            date_string = self.settings.all_filename_lists[self.settings.cam_nums[0]][i].split('/')[-1].split('_')[0]
            time_string = self.settings.all_filename_lists[self.settings.cam_nums[0]][i].split('/')[-1].split('_')[1]
            # Switch from Eastern time to UT
            hour = int(time_string[0:2])
            hour += 4
            hour = hour % 24
            time_string = (date_string + '_' + ('%d' % hour) + ':' + time_string[2:4] + ':' + time_string[4:])
            fig.suptitle(time_string)

            fig.set_size_inches([12, 8])
            fn = 'projected_' + time_string + '.png'
            fig.savefig(os.path.join(self.out_path, fn), dpi=self.settings.dpi, bbox_inches='tight')


if __name__ == "__main__":
    app = SeriesImageProjectionApp()
    app.project_images()
