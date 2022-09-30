import os
import sys
import time
import datetime

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd
from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import binning, flat_field
from skywinder_analysis.lib.power_spectrum import power_spectrum_tools
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app
from scipy import interpolate
from skywinder_analysis.lib.units import coordinate_transforms


class img_container_series():
    def __init__(self, cam_num=0):
        self.cam_num = cam_num
        self.ff = False
        self.imgs = []
        self.fh = False
        self.fw = False


class ProjectedPowerSpectrumMovieMakerApp(skywinder_analysis_app.App):
    def ani_frame_from_raw(self):
        self.create_output()
        self.logger.info('Creating container classes')
        frames = len(self.settings.all_filename_lists[self.settings.cam_nums[0]])
        ics = {}
        global_index = 0
        for cam_num in self.settings.cam_nums:
            ic = img_container_series(cam_num=cam_num)

            for fn in self.settings.all_filename_lists[cam_num]:
                if self.settings.use_new_flat_fields:
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
                alt, az = coordinate_transforms.cart_to_altaz(x, y, verbose=verbose)
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

        imd = np.zeros((len(self.settings.xvalues), len(self.settings.yvalues)))
        imd = np.rot90(imd, k=1)
        psd, xfreq_step, yfreq_step, xcenter, ycenter, xmaxrange, ymaxrange, xindices, yindices, xlabels, ylabels = power_spectrum_tools.get_2d_psd_and_labels(imd, m_per_pixel=self.settings.m_per_pixel,
                                                                                                                                                               min_physical_range=self.settings.min_physical_range,
                                                                                                                                                               tick_spacing=self.settings.tick_spacing,
                                                                                                                                                               wavelength_labels=self.settings.wavelength_labels)

        fig, axs = plt.subplots(1,2)
        im0 = axs[0].imshow(imd, cmap=self.settings.color_scheme,
                        origin='lower', extent=[xmin, xmax, ymax, ymin])

        im1 = axs[1].imshow(np.abs(psd), cmap=self.settings.color_scheme,
                        origin='lower')
        axs[1].set_xlim(xcenter,xcenter + xmaxrange)
        axs[1].set_ylim(ycenter - ymaxrange,ycenter + ymaxrange)
        if self.settings.wavelength_labels:
            axs[1].set_xlabel('Wavelength (km))')
            axs[1].set_ylabel('Wavelength (km))')
        else:
            axs[1].set_xlabel('Spatial Frequency (10^-5/m))')
            axs[1].set_ylabel('Spatial Frequency (10^-5/m))')
        axs[1].set_xticks(xindices)
        axs[1].set_yticks(yindices)
        axs[1].set_xticklabels(xlabels, rotation = 45)
        axs[1].set_yticklabels(ylabels, rotation = 45)
        axs[1].grid(True)

        if self.settings.grid:
            #ax.set_xlabel('x (km)')
            #ax.set_ylabel('y (km)')
            axs[0].xaxis.set_major_locator(plt.MultipleLocator(self.settings.grid_spacing))
            axs[0].yaxis.set_major_locator(plt.MultipleLocator(self.settings.grid_spacing))
            axs[0].grid(True)
        else:
            axs[0].grid(False)
            axs[0].get_xaxis().set_visible(False)
            axs[0].get_yaxis().set_visible(False)
        #fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        vsize = 1.1 * max(imd.shape[0], psd.shape[0]) / self.settings.dpi
        hsize = 2.1 * max(imd.shape[1], psd.shape[1]) / self.settings.dpi
        fig.set_size_inches([hsize, vsize])


        xfig, yfig = fig.get_dpi() * fig.get_size_inches()
        fontsize = yfig / 24
        if self.settings.timestamp_in_image:
            time_txt = axs[0].text(0.05, .95, 'time:', ha='left', va='top', color=self.settings.text_color,
                               fontsize=fontsize, transform=axs[0].transAxes)
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




            no_nan_image = stretched_image-np.nanmean(stretched_image)
            psd, xfreq_step, yfreq_step, xcenter, ycenter, xmaxrange, ymaxrange, xindices, yindices, xlabels, ylabels = power_spectrum_tools.get_2d_psd_and_labels(no_nan_image, m_per_pixel=self.settings.m_per_pixel,
                                                                                                                                           min_physical_range=self.settings.min_physical_range,
                                                                                                                                           tick_spacing=self.settings.tick_spacing)
            if self.settings.filter_freq:
                psd = power_spectrum_tools.filter_psd_by_freq(psd, xfreq_step, yfreq_step,
                                                              self.settings.low_freq_filter, self.settings.high_freq_filter)
                stretched_image = np.abs(np.fft.ifft2(psd))

            vmin = np.nanpercentile(stretched_image, .01)
            vmax = np.nanpercentile(stretched_image, 99.99)
            print(vmin, vmax)
            im0.set_data(stretched_image)
            im0.set_clim(vmin=vmin, vmax=vmax)

            im1.set_data(np.abs(psd))
            vmin = np.min(np.abs(psd))
            vmax = np.max(np.abs(psd))
            print(vmin, vmax)
            im1.set_clim(vmin=vmin, vmax=vmax)

            if self.settings.individual_frames:
                fig_fn = os.path.join(self.out_path, ('frame_%d_%s.png' % (n, date_string)))
                plt.savefig(fig_fn)
                self.logger.info('Frame saved as %s' % fig_fn)
            self.logger.info("\r%d of %d %.1f minutes elapsed, %.1f minutes remaining" % (index, frames, elapsed/60, (frames-index) * time_per_frame / 60))
            return im0, im1, time_txt

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
    app = ProjectedPowerSpectrumMovieMakerApp()
    app.ani_frame_from_raw()
    app.end()
