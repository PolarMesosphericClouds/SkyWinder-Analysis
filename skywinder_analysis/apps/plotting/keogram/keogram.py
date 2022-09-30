from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import binning
from skywinder_analysis.lib.image_processing import flat_field
import scipy.stats
import datetime
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app
import matplotlib.pyplot as plt
import numpy as np


class KeogramApp(skywinder_analysis_app.App):
    def make_keogram(self):
        slices = []

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

        for i, fn in enumerate(self.settings.filenames):
            img, _ = blosc_file.load_blosc_image(fn)
            if use_flat_field_image:
                img = flat_field.apply_flat_field(img, flat_field_image)
            if self.settings.bin_pixels:
                img = binning.bucket(img, self.settings.bin_pixels)
            slices.append(img[img.shape[0] // 2, :])
            print('%d of %d finished' % (i, len(self.settings.filenames)))

        sa = np.array(slices)
        self.logger.info(sa.shape)
        sa = np.rot90(sa, k=1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        vmin = scipy.stats.scoreatpercentile(sa, self.settings.min_percentile)
        vmax = scipy.stats.scoreatpercentile(sa, self.settings.max_percentile)

        ax.xaxis.set_major_locator(plt.MultipleLocator(15 * 60 // self.settings.interval))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(5 * 60 // self.settings.interval))

        def format_func_xaxis(value, tick_number):
            seconds = value * self.settings.interval
            timestamp = start + seconds
            return datetime.datetime.utcfromtimestamp(timestamp).strftime('%H:%M:%S')

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func_xaxis))

        def format_func_yaxis(value, tick_number):
            return int(value * bin_)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func_yaxis))

        fig.autofmt_xdate()

        ax.set_xlabel('Time')
        ax.set_ylabel('Vertical Pixel Coordinate')

        ax.imshow(sa, vmin=vmin, vmax=vmax, cmap=self.settings.colormap)

        ax.set_title('Keogram camera %d vertical' % self.settings.camera_number)
        plt.savefig(self.settings.output_name)


if __name__ == "__main__":
    app = KeogramApp()
    app.make_keogram()
    app.end()
