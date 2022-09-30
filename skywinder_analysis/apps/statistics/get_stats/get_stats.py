import os
import sys
import time

import numpy as np
import scipy.stats
from skywinder_analysis.lib.tools import blosc_file
import csv
from skywinder_analysis.lib.image_processing import rolling_flat_field, flat_field
from skywinder_analysis.lib.skywinder_analysis_app import skywinder_analysis_app


def get_mad(array):
    return np.median(np.abs(array - np.median(array)))


class GetStatsApp(skywinder_analysis_app.App):
    def get_mad(self, array):
        return np.median(np.abs(array - np.median(array)))

    def get_stats(self):
        self.create_output()
        self.logger.info('Starting acquisition of stats')
        frames = len(self.settings.filenames)
        self.logger.info('%d filenames passed' % frames)

        rolling_ff = rolling_flat_field.RollingFlatField(size=self.settings.rolling_flat_field_size)
        for i in range(self.settings.rolling_flat_field_size // 2):
            rolling_ff.roll(self.settings.filenames[i])

        # Populate rolling flat field with first size/2 images.

        percentile_keys = []
        percentiles = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
        for p in percentiles:
            key = 'percentile_%d' % p
            percentile_keys.append(key)

        def get_image_stats(n):
            try:
                if n > (len(self.settings.filenames) - (self.settings.rolling_flat_field_size / 2)):
                    pass
                else:
                    m = n + (self.settings.rolling_flat_field_size // 2)
                    rolling_ff.roll(self.settings.filenames[m])
                    self.logger.info('Rolling to index %d for ff on image %d' % (m, n))
                flat_field_image = rolling_ff.generate_flat_field()
                filename = self.settings.filenames[n]
                img, _ = blosc_file.load_blosc_image(filename)
                img = flat_field.apply_flat_field(img, flat_field_image)
            except Exception as e:
                self.logger.info(e)
                return

            epoch = filename.split('/')[-1].split('=')[-1]
            mean = np.mean(img)
            std = np.std(img)
            mad = get_mad(img)
            stats = {'epoch': epoch,
                     'mean': mean,
                     'mad': mad,
                     'std': std}
            for key, p in zip(percentile_keys, percentiles):
                percentile = scipy.stats.scoreatpercentile(img, p)
                stats[key] = percentile
            return stats

        file_name = os.path.join(self.out_path, self.settings.output_name)
        with open(file_name, 'w') as csvfile:
            fieldnames = ['epoch', 'mean', 'mad', 'std'] + percentile_keys
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(frames):
                stats = get_image_stats(i)
                writer.writerow(stats)
        return


if __name__ == "__main__":
    app = GetStatsApp()
    app.get_stats()
    app.end()
