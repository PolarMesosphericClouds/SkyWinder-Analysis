import os
import sys
import time
import datetime

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import binning, rolling_flat_field, flat_field
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app


class RollingMovieMakerApp(skywinder_analysis_app.App):
    def ani_frame_from_raw(self):
        self.create_output()
        self.logger.info('Starting animation of frames')
        imd = np.zeros(self.settings.raw_image_size)
        if self.settings.section:
            imd = imd[self.settings.section[0]:self.settings.section[1],
                  self.settings.section[2]:self.settings.section[3]]

        frames = len(self.settings.filenames)
        self.logger.info('%d filenames passed' % frames)
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

        if self.settings.bin_pixels:
            img = binning.bucket(imd, self.settings.bin_pixels)
        else:
            img = imd
        vsize = img.shape[0] / self.settings.dpi
        hsize = img.shape[1] / self.settings.dpi
        fig.set_size_inches([hsize, vsize])

        im = ax.imshow(img, cmap=cm.inferno, interpolation='nearest')

        txt = ax.text(0.1, 0.9, 'time:', ha='left', va='top', color='yellow')
        # plt.tight_layout()
        start_at = time.time()
        ax = plt.gca()

        rolling_ff = rolling_flat_field.RollingFlatField(size=self.settings.rolling_flat_field_size)
        for i in range(self.settings.rolling_flat_field_size // 2):
            rolling_ff.roll(self.settings.filenames[i])

        # Populate rolling flat field with first size/2 images.

        def update_img(n):
            elapsed = time.time() - start_at
            if n:
                time_per_frame = elapsed / n
            else:
                time_per_frame = 1
            sys.stdout.flush()
            try:
                if n > (len(self.settings.filenames) - (self.settings.rolling_flat_field_size / 2)):
                    pass
                else:
                    m = n + (self.settings.rolling_flat_field_size // 2)
                    rolling_ff.roll(self.settings.filenames[m])
                    self.logger.info('Rolling to index %d for ff on image %d' % (m, n))
                flat_field_image = rolling_ff.generate_flat_field()
                img, _ = blosc_file.load_blosc_image(self.settings.filenames[n])
                img = flat_field.apply_flat_field(img, flat_field_image)
            except Exception as e:
                self.logger.info(e)
                return
            if self.settings.section:
                img = img[self.settings.section[0]:self.settings.section[1],
                      self.settings.section[2]:self.settings.section[3]]
            if self.settings.bin_pixels:
                img = binning.bucket(img, self.settings.bin_pixels)
            im.set_data(img)

            # if vmin == 0:
            vmin = scipy.stats.scoreatpercentile(img[self.settings.level_region[0]:self.settings.level_region[1],
                                                 self.settings.level_region[2]:self.settings.level_region[3]], 0.1)
            vmax = scipy.stats.scoreatpercentile(img[self.settings.level_region[0]:self.settings.level_region[1],
                                                 self.settings.level_region[2]:self.settings.level_region[3]], 99.9)
            im.set_clim(vmin, vmax)

            unix_timestamp_s = int(self.settings.filenames[n].split('=')[-1][:10])
            dt = datetime.datetime.utcfromtimestamp(unix_timestamp_s)
            date_string = dt.strftime('%m-%d %H:%M:%S')
            txt.set_text(date_string)
            self.logger.info("\r%d of %d %.1f minutes elapsed, %.1f minutes remaining" % (
                n, len(self.settings.filenames), elapsed / 60,
                (len(self.settings.filenames) - n) * time_per_frame / 60))

            return im, txt

        ani = animation.FuncAnimation(fig, update_img, frames, interval=100)

        ani_fn = os.path.join(self.out_path, self.settings.output_name)
        ani.save(ani_fn, writer=self.settings.writer, dpi=self.settings.dpi)
        return ani


if __name__ == "__main__":
    app = RollingMovieMakerApp()
    app.ani_frame_from_raw()
    app.end()
