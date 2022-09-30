import os
import sys
import time

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.ndimage
from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import binning, flat_field
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app


class MovieMakerApp(skywinder_analysis_app.App):
    def ani_frame_from_raw(self):
        self.create_output()
        self.logger.info('Starting animation of frames')
        imd = np.zeros(self.settings.raw_image_size)
        self.logger.info('Section is %r' % self.settings.section)
        if self.settings.section:
            imd = imd[self.settings.section[0]:self.settings.section[1],
                  self.settings.section[2]:self.settings.section[3]]
        frames = len(self.settings.filenames)
        self.logger.info('%d filenames passed' % frames)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        if self.settings.bin_pixels:
            img = binning.bucket(imd, self.settings.bin_pixels)
        else:
            img = imd
        if self.settings.rotate:
            self.logger.info('Shape before: %d by %d' % (img.shape[0], img.shape[1]))
            img = np.rot90(img, k=1)
            self.logger.info('Shape after: %d by %d' % (img.shape[0], img.shape[1]))
        im = ax.imshow(img, cmap=cm.inferno, interpolation='nearest')
        self.logger.info('Image shape: %d by %d' % (img.shape[0], img.shape[1]))
        vsize = img.shape[0] / self.settings.dpi
        hsize = img.shape[1] / self.settings.dpi
        self.logger.info('Size is vertical: %.1f horizontal: %.1f' % (vsize, hsize))
        fig.set_size_inches([hsize, vsize])
        xfig, yfig = fig.get_dpi() * fig.get_size_inches()
        fontsize = yfig / 32
        txt = ax.text(0.01 * xfig, 0.05 * yfig, 'time:', ha='left', va='top', color='black', fontsize=fontsize)
        start_at = time.time()
        ax = plt.gca()
        if self.settings.saved_flat_field_path:
            self.logger.info('Loading saved flat field %s' % self.settings.saved_flat_field_path)
            flat_field_image = np.load(self.settings.saved_flat_field_path)
            use_flat_field_image = True
        elif self.settings.flat_field_filenames:
            self.logger.info('Generating flat field from %d files' % len(self.settings.flat_field_filenames))
            flat_field_image = flat_field.generate_flat_field_image_from_filenames(self.settings.flat_field_filenames)
            use_flat_field_image = True
            # Generate flat_field_image from filenames
        else:
            self.logger.info('No flat field used')
            use_flat_field_image = False

        def update_img(n):
            elapsed = time.time() - start_at
            if n:
                time_per_frame = elapsed / n
            else:
                time_per_frame = 1
            sys.stdout.flush()
            try:
                img, _ = blosc_file.load_blosc_image(self.settings.filenames[n])
                if use_flat_field_image:
                    img = flat_field.apply_flat_field(img, flat_field_image)
            except Exception as e:
                self.logger.info(e)
                return
            if self.settings.section:
                img = img[self.settings.section[0]:self.settings.section[1],
                      self.settings.section[2]:self.settings.section[3]]
            if self.settings.bin_pixels:
                img = binning.bucket(img, self.settings.bin_pixels)
            if self.settings.gaussian_filter_sigma:
                img = scipy.ndimage.gaussian_filter(img, sigma=self.settings.gaussian_filter_sigma)
            if self.settings.rotate:
                img = np.rot90(img, k=1)
            im.set_data(img)

            # if vmin == 0:
            vmin = scipy.stats.scoreatpercentile(img, 0.1)
            vmax = scipy.stats.scoreatpercentile(img, 99.9)
            vmin = scipy.stats.scoreatpercentile(
                img[self.settings.level_region[0]:self.settings.level_region[1],
                self.settings.level_region[2]:self.settings.level_region[3]],
                self.settings.min_percentile)
            vmax = scipy.stats.scoreatpercentile(
                img[self.settings.level_region[0]:self.settings.level_region[1],
                self.settings.level_region[2]:self.settings.level_region[3]],
                self.settings.max_percentile)
            im.set_clim(vmin, vmax)

            try:
                time_string = self.settings.filenames[n].split('/')[-1].split('_')[1]
                # Switch from Eastern time to UT
                hour = int(time_string[0:2])
                hour += 4
                hour = hour % 24

                time_string = ('Camera %d ' % self.settings.camera_number) + ('%d' % hour) + ':' + time_string[
                                                                                                   2:4] + ':' + time_string[
                                                                                                                4:]
                txt.set_text(time_string)
            except Exception:
                pass
            self.logger.info("\r%d of %d %.1f minutes elapsed, %.1f minutes remaining" % (
                n, len(self.settings.filenames), elapsed / 60,
                (len(self.settings.filenames) - n) * time_per_frame / 60))

            return im, txt

        ani = animation.FuncAnimation(fig, update_img, frames, interval=50)

        ani_fn = os.path.join(self.out_path, self.settings.output_name)
        ani.save(ani_fn, writer=self.settings.writer, dpi=self.settings.dpi)
        return ani


if __name__ == "__main__":
    app = MovieMakerApp()
    app.ani_frame_from_raw()
    app.end()
