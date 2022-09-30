import os
import sys
import time

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import binning, rolling_flat_field, flat_field, stitching
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app


class StitchedRollingMovieMakerApp(skywinder_analysis_app.App):
    def ani_frame_from_raw(self):
        self.create_output()
        self.logger.info('Starting animation of frames')
        imd = np.zeros(self.settings.raw_image_size)

        frames = min([len(filenames) for filenames in self.settings.all_filenames])
        for filenames in self.settings.all_filenames:
            self.logger.info('%d filenames passed' % len(filenames))
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

        if self.settings.bin_pixels:
            img = stitching.build_stitched_image(4 * [binning.bucket(imd, self.settings.bin_pixels)])
        else:
            img = stitching.build_stitched_image(4 * [imd])
        vsize = img.shape[0] / self.settings.dpi
        hsize = img.shape[1] / self.settings.dpi
        fig.set_size_inches([hsize, vsize])

        im = ax.imshow(img, cmap=self.settings.color_scheme, interpolation='nearest')

        xfig, yfig = fig.get_dpi() * fig.get_size_inches()
        fontsize = yfig / 32
        txt0 = ax.text(0.01 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color='black', fontsize=fontsize)
        txt1 = ax.text(0.26 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color='black', fontsize=fontsize)
        txt2 = ax.text(0.51 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color='black', fontsize=fontsize)
        txt3 = ax.text(0.76 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color='black', fontsize=fontsize)
        time_txt = ax.text(0.01 * xfig, 0.05 * yfig, 'time:', ha='left', va='top', color='black', fontsize=fontsize)
        all_txt = [txt0, txt1, txt2, txt3]
        # plt.tight_layout()
        start_at = time.time()
        ax = plt.gca()

        flat_fields = []
        for j, filenames in enumerate(self.settings.all_filenames):
            self.logger.info('Creating ff %d of %d' % (j + 1, len(self.settings.all_filenames)))
            rolling_ff = rolling_flat_field.RollingFlatField(size=self.settings.rolling_flat_field_size)
            for i in range(self.settings.rolling_flat_field_size // 2):
                rolling_ff.roll(self.settings.all_filenames[j][i])
            flat_fields.append(rolling_ff)

        # Populate rolling flat fields with first size/2 images.

        def update_img(n):
            elapsed = time.time() - start_at
            if n:
                time_per_frame = elapsed / n
            else:
                time_per_frame = 1
            sys.stdout.flush()

            imgs = []
            for camera, txt, ff, filenames in zip(self.settings.cameras, all_txt, flat_fields,
                                                  self.settings.all_filenames):
                try:
                    if n > (len(filenames) - (self.settings.rolling_flat_field_size / 2)):
                        pass
                    else:
                        m = n + (self.settings.rolling_flat_field_size // 2)
                        ff.roll(filenames[m])
                        self.logger.info('Rolling to index %d for ff on image %d' % (m, n))
                    flat_field_image = ff.generate_flat_field()
                    img, _ = blosc_file.load_blosc_image(filenames[n])
                    img = flat_field.apply_flat_field(img, flat_field_image)

                    try:
                        time_string = filenames[n].split('/')[-1].split('_')[1]
                        time_string = time_string[0:2] + ':' + time_string[2:4] + ':' + time_string[4:]
                        date_string = filenames[n].split('/')[-1].split('_')[0]
                        time_string = 'Camera ' + str(camera) + ' ' + date_string + ' ' + time_string
                        txt.set_text(time_string)
                    except Exception:
                        pass

                except Exception as e:
                    self.logger.info(e)
                    return

                if self.settings.bin_pixels:
                    img = binning.bucket(img, self.settings.bin_pixels)

                imgs.append(img)

            means = [np.mean(img[self.settings.mean_region[0]:self.settings.mean_region[1],
                             self.settings.mean_region[2]:self.settings.mean_region[3],
                             ]) for img in imgs]
            mean_mean = np.mean(means)
            releveled_imgs = []
            for mean, img in zip(means, imgs):
                self.logger.info('Releveling. Mean of means is %.1f, mean of img is %.1f.' % (mean_mean, mean))
                releveled_imgs.append((mean_mean / mean) * img)

            stitched_img = stitching.build_stitched_image(releveled_imgs)
            im.set_data(stitched_img)

            vmin = scipy.stats.scoreatpercentile(
                stitched_img[self.settings.level_region[0]:self.settings.level_region[1],
                self.settings.level_region[2]:self.settings.level_region[3]],
                self.settings.min_percentile)
            vmax = scipy.stats.scoreatpercentile(
                stitched_img[self.settings.level_region[0]:self.settings.level_region[1],
                self.settings.level_region[2]:self.settings.level_region[3]],
                self.settings.max_percentile)
            self.logger.info('%.1f percentile is %.1f, %.1f percentile is %.1f' % (self.settings.min_percentile, vmin, self.settings.max_percentile, vmax))
            #vmin = scipy.stats.scoreatpercentile(stitched_img, 0.1)
            #vmax = scipy.stats.scoreatpercentile(stitched_img, 99.9)
            im.set_clim(vmin, vmax)

            self.logger.info("\r%d of %d %.1f minutes elapsed, %.1f minutes remaining" % (
                n, frames, elapsed / 60,
                (frames - n) * time_per_frame / 60))

            return im, txt0, txt1, txt2, txt3

        ani = animation.FuncAnimation(fig, update_img, frames, interval=100)

        ani_fn = os.path.join(self.out_path, self.settings.output_name)
        ani.save(ani_fn, writer=self.settings.writer, dpi=self.settings.dpi)
        return ani


if __name__ == "__main__":
    app = StitchedRollingMovieMakerApp()
    app.ani_frame_from_raw()
    app.end()
