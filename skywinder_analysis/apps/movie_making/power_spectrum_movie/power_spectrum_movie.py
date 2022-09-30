import os
import sys
import time

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import binning, flat_field
from skywinder_analysis.lib.power_spectrum import power_spectrum_tools
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app


class PowerSpectrumMovieMakerApp(skywinder_analysis_app.App):
    def ani_frame_from_raw(self):
        self.create_output()
        self.logger.info('Starting animation of frames')
        imd = np.zeros(self.settings.raw_image_size)
        self.logger.info('Section is %r' % self.settings.section)
        if self.settings.section:
            imd = imd[self.settings.section[0]:self.settings.section[1],
                  self.settings.section[2]:self.settings.section[3]]
        if self.settings.rotate:
            imd = np.rot90(imd, k=1)
        frames = len(self.settings.filenames)
        self.logger.info('%d filenames passed' % frames)
        fig, axs = plt.subplots(1,2)
        for ax in axs:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        if self.settings.bin_pixels:
            img = binning.bucket(imd, self.settings.bin_pixels)
        else:
            img = imd
        psd = power_spectrum_tools.get_2d_psd(img)
        if self.settings.bin_psd_pixels:
            psd = binning.bucket(psd, self.settings.bin_psd_pixels)
        if self.settings.rotate:
            img = np.rot90(img, k=1)
        im0 = axs[0].imshow(img, cmap=cm.Blues, interpolation='nearest')
        im1 = axs[1].imshow(psd, cmap=cm.inferno, interpolation='nearest')

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
            if self.settings.section:
                img = img[self.settings.section[0]:self.settings.section[1],
                      self.settings.section[2]:self.settings.section[3]]
            if self.settings.bin_pixels:
                img = binning.bucket(img, self.settings.bin_pixels)
            if self.settings.rotate:
                img = np.rot90(img, k=1)
            im0.set_data(img)

            vmin = scipy.stats.scoreatpercentile(
                img[self.settings.level_region[0]:self.settings.level_region[1],
                self.settings.level_region[2]:self.settings.level_region[3]],
                self.settings.min_percentile)
            vmax = scipy.stats.scoreatpercentile(
                img[self.settings.level_region[0]:self.settings.level_region[1],
                self.settings.level_region[2]:self.settings.level_region[3]],
                self.settings.max_percentile)

            im0.set_clim(vmin, vmax)

            try:
                filename_for_date = self.settings.filenames[n]
                unix_timestamp_s = int(filename_for_date.split('=')[-1][:10])
                dt = datetime.datetime.utcfromtimestamp(unix_timestamp_s)
                date_string = dt.strftime('%m-%d_%H:%M:%S')
                txt.set_text(date_string)
            except Exception:
                pass
            self.logger.info("\r%d of %d %.1f minutes elapsed, %.1f minutes remaining" % (
                n, len(self.settings.filenames), elapsed / 60,
                (len(self.settings.filenames) - n) * time_per_frame / 60))

            psd = power_spectrum_tools.get_2d_psd(img)
            psd = np.log(psd)
            psd = psd[(psd.shape[0]//4):(3*psd.shape[0]//4), (psd.shape[1]//4):(3*psd.shape[1]//4)]
            if self.settings.bin_psd_pixels:
                psd = binning.bucket(psd, self.settings.bin_psd_pixels)
            im1.set_data(psd)
            vmin = scipy.stats.scoreatpercentile(psd, 0.1)
            vmax = scipy.stats.scoreatpercentile(psd, 99.9)
            #self.logger.info('PSD vmin is %.1f, vmax is %.1f' % (vmin, vmax))
            im1.set_clim(vmin, vmax)

            if self.settings.individual_frames:
                fig_fn = os.path.join(self.out_path, ('frame_%d_%s.png' % (n, date_string)))
                plt.savefig(fig_fn)
                self.logger.info('Frame saved as %s' % fig_fn)

            return im0, im1, txt
        if not self.settings.individual_frames:
            ani = animation.FuncAnimation(fig, update_img, frames, interval=50)

            ani_fn = os.path.join(self.out_path, self.settings.output_name)
            ani.save(ani_fn, writer=self.settings.writer, dpi=self.settings.dpi)
            return ani
        else:
            for n in range(frames):
                update_img(n)
            return True


if __name__ == "__main__":
    app = PowerSpectrumMovieMakerApp()
    app.ani_frame_from_raw()
    app.end()
