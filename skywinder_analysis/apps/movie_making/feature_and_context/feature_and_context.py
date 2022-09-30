import os
import sys
import time

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.stats
from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import binning, flat_field
from skywinder_analysis.lib.power_spectrum import power_spectrum_tools
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app


class FeatureAndContextMovieMakerApp(skywinder_analysis_app.App):
    def ani_frame_from_raw(self):
        self.create_output()
        self.logger.info('Starting animation of frames')
        imd = np.zeros(self.settings.raw_image_size)
        if self.settings.rotate:
            imd = np.rot90(imd, k=1)
        frames = len(self.settings.filenames)
        self.logger.info('%d filenames passed' % frames)
        fig, axs = plt.subplots(1, 2)
        for ax in axs:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        feature = imd[self.settings.section_y:self.settings.section_y + self.settings.section_ywidth,
                  self.settings.section_x:self.settings.section_x + self.settings.section_xwidth]
        img = binning.bucket(imd, self.settings.bin_pixels)
        rect = patches.Rectangle((self.settings.section_x // self.settings.bin_pixels[0],
                                  self.settings.section_y // self.settings.bin_pixels[0]),
                                  self.settings.section_xwidth // self.settings.bin_pixels[0],
                                  self.settings.section_ywidth // self.settings.bin_pixels[0],
                                  linewidth=4,
                                  edgecolor='c',
                                  facecolor='none')

        feature = binning.bucket(feature, self.settings.feature_bin_pixels)
        if self.settings.rotate:
            img = np.rot90(img, k=1)
        axs[0].add_patch(rect)
        im0 = axs[0].imshow(img, cmap=cm.inferno, interpolation='nearest')
        im1 = axs[1].imshow(feature, cmap=cm.inferno, interpolation='nearest')

        vsize = 1 * img.shape[0] / self.settings.dpi
        hsize = 2.1 * img.shape[1] / self.settings.dpi
        fig.set_size_inches([hsize, vsize])
        xfig, yfig = fig.get_dpi() * fig.get_size_inches()
        fontsize = yfig / 32
        txt = axs[0].text(0.01 * xfig, 0.05 * yfig, 'time:', ha='left', va='top', color='black', fontsize=fontsize)
        start_at = time.time()
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
            self.logger.info('X: %d, X-width: %d, Y: %d, Y-width: %d' % (self.settings.section_x, self.settings.section_xwidth, self.settings.section_y, self.settings.section_ywidth))
            feature = img[self.settings.section_y:self.settings.section_y + self.settings.section_ywidth,
                          self.settings.section_x:self.settings.section_x + self.settings.section_xwidth]
            self.logger.info('Shape of feature is %d by %d' % (feature.shape[0], feature.shape[1]))
            img = binning.bucket(img, self.settings.bin_pixels)
            feature = binning.bucket(feature, self.settings.feature_bin_pixels)
            rect.set_xy([self.settings.section_x // self.settings.bin_pixels[0], self.settings.section_y // self.settings.bin_pixels[0]])
            if self.settings.rotate:
                img = np.rot90(img, k=1)
            im0.set_data(img)
            vmin = scipy.stats.scoreatpercentile(img, self.settings.min_percentile)
            vmax = scipy.stats.scoreatpercentile(img, self.settings.max_percentile)
            self.logger.info('%.1f percentile is %.1f, %.1f percentile is %.1f' % (self.settings.min_percentile, vmin, self.settings.max_percentile, vmax))
            im0.set_clim(vmin, vmax)

            try:
                time_string = self.settings.filenames[n].split('/')[-1].split('_')[1]
                # Switch from Eastern time to UT
                hour = int(time_string[0:2])
                hour += 4
                hour = hour % 24

                time_string = ('Camera %d ' % self.settings.camera_number) + ('%d' % hour) \
                              + ':' + time_string[2:4] + ':' + time_string[4:]
                txt.set_text(time_string)
            except Exception:
                pass
            self.logger.info("\r%d of %d %.1f minutes elapsed, %.1f minutes remaining" % (
                n, len(self.settings.filenames), elapsed / 60,
                (len(self.settings.filenames) - n) * time_per_frame / 60))

            im1.set_data(feature)
            im1.set_clim(vmin, vmax)

            self.settings.section_x = self.settings.section_x + self.settings.xshift
            self.settings.section_y = self.settings.section_y + self.settings.yshift
            return im0, im1, txt

        ani = animation.FuncAnimation(fig, update_img, frames, interval=50)

        ani_fn = os.path.join(self.out_path, self.settings.output_name)
        ani.save(ani_fn, writer=self.settings.writer, dpi=self.settings.dpi)
        return ani


if __name__ == "__main__":
    app = FeatureAndContextMovieMakerApp()
    app.ani_frame_from_raw()
    app.end()
