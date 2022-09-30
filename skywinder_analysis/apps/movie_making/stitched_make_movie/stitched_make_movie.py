import os
import sys
import time
import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import binning, rolling_flat_field, flat_field, flat_field_piggyback, stitching
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app


class StitchedMovieMakerApp(skywinder_analysis_app.App):
    def ani_frame_from_raw(self):
        self.create_output()
        self.logger.info('Starting animation of frames')
        imd = np.zeros(self.settings.raw_image_size)
        if self.settings.crop:
            imd = imd[self.settings.crop[0]:self.settings.crop[1],
                      self.settings.crop[2]:self.settings.crop[3]]
        frames = min([len(filenames) for filenames in self.settings.all_filenames])
        for filenames in self.settings.all_filenames:
            self.logger.info('%d filenames passed' % len(filenames))
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        if self.settings.bin_pixels:
            img = stitching.build_stitched_image(len(self.settings.all_filenames) * [binning.bucket(imd, self.settings.bin_pixels)])
        else:
            img = stitching.build_stitched_image(len(self.settings.all_filenames) * [imd])
        if self.settings.rotate:
            self.logger.info('Shape before: %d by %d' % (img.shape[0], img.shape[1]))
            img = np.rot90(img, k=self.settings.rotate_k)
            self.logger.info('Shape after: %d by %d' % (img.shape[0], img.shape[1]))
        vsize = img.shape[0] / self.settings.dpi
        hsize = img.shape[1] / self.settings.dpi
        fig.set_size_inches([hsize, vsize])
        im = ax.imshow(img, cmap=self.settings.color_scheme, interpolation='nearest')
        xfig, yfig = fig.get_dpi() * fig.get_size_inches()
        fontsize = yfig * self.settings.font_proportion
        if len(self.settings.all_filenames) == 4:
            txt0 = ax.text(0.01 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color=self.settings.fontcolor,
                           fontsize=fontsize//2)
            txt1 = ax.text(0.26 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color=self.settings.fontcolor,
                           fontsize=fontsize//2)
            txt2 = ax.text(0.51 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color=self.settings.fontcolor,
                           fontsize=fontsize//2)
            txt3 = ax.text(0.76 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color=self.settings.fontcolor,
                           fontsize=fontsize//2)
            all_txt = [txt0, txt1, txt2, txt3]
        if len(self.settings.all_filenames) == 3:
            txt0 = ax.text(0.01 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color=self.settings.fontcolor,
                           fontsize=fontsize//2)
            txt1 = ax.text(0.34 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color=self.settings.fontcolor,
                           fontsize=fontsize//2)
            txt2 = ax.text(0.67 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color=self.settings.fontcolor,
                           fontsize=fontsize//2)
            all_txt = [txt0, txt1, txt2]
        if len(self.settings.all_filenames) == 2:
            txt0 = ax.text(0.01 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color=self.settings.fontcolor,
                           fontsize=fontsize//2)
            txt1 = ax.text(0.51 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color=self.settings.fontcolor,
                           fontsize=fontsize//2)
            all_txt = [txt0, txt1]
        if len(self.settings.all_filenames) == 1:
            txt0 = ax.text(0.01 * xfig, 0.95 * yfig, 'time:', ha='left', va='top', color=self.settings.fontcolor,
                           fontsize=fontsize//2)
            all_txt = [txt0]
        time_txt = ax.text(0.01 * xfig, 0.05 * yfig, 'time:', ha='left', va='top', color=self.settings.fontcolor,
                           fontsize=fontsize)
        start_at = time.time()
        ax = plt.gca()
        flat_fields = []
        if self.settings.saved_flat_field_paths:
            for path in self.settings.saved_flat_field_paths:
                self.logger.info('Loading saved flat field %s' % path)
                flat_field_image = np.load(path)
                flat_fields.append(flat_field_image)
            use_flat_field_image = True
            old_flat_field = True
        elif self.settings.all_flat_field_filenames:
            for filenames in self.settings.all_flat_field_filenames:
                self.logger.info('Generating flat field from %d files' % len(filenames))
                flat_field_image = flat_field.generate_flat_field_image_from_filenames(filenames)
                flat_fields.append(flat_field_image)
            use_flat_field_image = True
            old_flat_field = True
        elif self.settings.use_new_flat_field:
            self.logger.info('Using sophisticated flat field')
            use_flat_field_image = True
            flat_fields = [False, False, False, False]
            old_flat_field = False
        else:
            self.logger.info('No flat fields used')
            flat_fields = [False, False, False, False]
            use_flat_field_image = False
            old_flat_field = True

        def update_img(n):
            elapsed = time.time() - start_at
            if n:
                time_per_frame = elapsed / n
            else:
                time_per_frame = 1
            sys.stdout.flush()

            imgs = []
            for camera, txt, flat_field_image, filenames in zip(self.settings.cameras, all_txt, flat_fields,
                                                                self.settings.all_filenames):
                try:

                    if old_flat_field:
                        img, _ = blosc_file.load_blosc_image(filenames[n])
                        if use_flat_field_image:
                            img = flat_field.apply_flat_field(img, flat_field_image)
                    else:
                        if self.settings.piggyback:
                            img = flat_field_piggyback.get_final_cleaned_image(filenames[n],
                                                                               reflection_window=self.settings.new_flat_field_reflection_window)
                        else:
                            img = flat_field.get_final_cleaned_image(filenames[n],
                                                                     reflection_window=self.settings.new_flat_field_reflection_window)
                    if self.settings.crop:
                        img = img[self.settings.crop[0]:self.settings.crop[1],
                              self.settings.crop[2]:self.settings.crop[3]]

                except Exception as e:
                    self.logger.info(e)
                    return

                if self.settings.bin_pixels:
                    img = binning.bucket(img, self.settings.bin_pixels)

                imgs.append(img)

            self.logger.info(len(imgs))

            if old_flat_field:
                means = [np.mean(img[self.settings.mean_region[0]:self.settings.mean_region[1],
                                 self.settings.mean_region[2]:self.settings.mean_region[3],
                                 ]) for img in imgs]
                mean_mean = np.mean(means)
                releveled_imgs = []
                for mean, img in zip(means, imgs):
                    releveled_imgs.append((mean_mean / mean) * img)

                stitched_img = stitching.build_stitched_image(releveled_imgs)

            else:
                # No need to relevel with sophisticated flat fielding algorithm.
                stitched_img = stitching.build_stitched_image(imgs)
                if self.settings.rotate:
                    stitched_img = np.rot90(stitched_img, k=self.settings.rotate_k)

            im.set_data(stitched_img)

            filename_for_date = self.settings.all_filenames[0][n]
            unix_timestamp_s = int(filename_for_date.split('=')[-1][:10])
            dt = datetime.datetime.utcfromtimestamp(unix_timestamp_s)
            date_string = dt.strftime('%m-%d_%H:%M:%S')
            camera_string = 'Camera ' + str(camera)
            if self.settings.no_text:
                txt.set_text(' ')
                time_txt.set_text(' ')
            else:
                txt.set_text(camera_string)
                time_txt.set_text(date_string)

            if self.settings.colorscale_by_frame:
                vmin = scipy.stats.scoreatpercentile(
                    stitched_img[self.settings.level_region[0]:self.settings.level_region[1],
                    self.settings.level_region[2]:self.settings.level_region[3]],
                    0.1)
                vmax = scipy.stats.scoreatpercentile(
                    stitched_img[self.settings.level_region[0]:self.settings.level_region[1],
                    self.settings.level_region[2]:self.settings.level_region[3]],
                    99.9)
                im.set_clim(vmin, vmax)
            else:
                if not self.settings.vmin:
                    self.settings.vmin = scipy.stats.scoreatpercentile(
                        stitched_img[self.settings.level_region[0]:self.settings.level_region[1],
                        self.settings.level_region[2]:self.settings.level_region[3]],
                        0.1)
                    self.settings.vmax = scipy.stats.scoreatpercentile(
                        stitched_img[self.settings.level_region[0]:self.settings.level_region[1],
                        self.settings.level_region[2]:self.settings.level_region[3]],
                        99.9)
                im.set_clim(self.settings.vmin, self.settings.vmax)

            self.logger.info("\r%d of %d %.1f minutes elapsed, %.1f minutes remaining" % (
                n, frames, elapsed / 60,
                (frames - n) * time_per_frame / 60))
            if self.settings.individual_frames:
                fig_fn = os.path.join(self.out_path, ('frame_%d_%s.png' % (n, date_string)))
                plt.savefig(fig_fn)
                #self.logger.info('Frame saved as %s' % fig_fn)
            return im  # , txt0, txt1, txt2, txt3


        if not self.settings.individual_frames:
            if not self.settings.split_video_len:
                self.logger.info('Making continuous movie')
                ani = animation.FuncAnimation(fig, update_img, frames, interval=100)
                ani_fn = os.path.join(self.out_path, self.settings.output_name)
                ani.save(ani_fn, writer=self.settings.writer, dpi=self.settings.dpi)
                return
            else:
                full_filename_list = self.settings.all_filenames
                # Keep record of the original full set of filenames.
                num_sections = frames // self.settings.split_video_len
                self.logger.info('Number of videos: %d' % num_sections)
                for j in range(num_sections):
                    newlist = []
                    for i in range(len(full_filename_list)):
                        newlist.append(full_filename_list[i][(j*self.settings.split_video_len):((j+1)*self.settings.split_video_len)])
                    self.settings.all_filenames = newlist
                    sub_frames = len(newlist[0])
                    self.logger.info('%d sublists each of length %d' % (len(newlist), sub_frames))
                    ani = animation.FuncAnimation(fig, update_img, sub_frames, interval=100)
                    ani_fn = os.path.join(self.out_path,('movie_%d.mp4'%j))
                    ani.save(ani_fn, writer=self.settings.writer, dpi=self.settings.dpi)
                    self.logger.info("Saved movie %d" % j)
                    ani = 0
                return



        else:
            for n in range(frames):
                update_img(n)
            return


        self.logger.info('Program should never reach this point')

if __name__ == "__main__":
    app = StitchedMovieMakerApp()
    app.ani_frame_from_raw()
    app.end()
