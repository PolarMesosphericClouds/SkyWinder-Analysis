from skywinder_analysis.lib.tools import generic
import matplotlib.animation as animation
import matplotlib.cm as cm

settings = generic.Class()

settings.section = None
settings.bin_pixels = (4, 4)
settings.dpi = 100
settings.rolling_flat_field_size = 20

settings.cameras = [4, 5, 6, 7]

settings.raw_image_size = (3232, 4864)
settings.output_name = 'movie.mp4'
settings.all_filenames = []  # use glob to make list of lists of filenames
settings.writer = animation.writers['ffmpeg'](fps=10, codec='h264', bitrate=2 ** 20)
settings.level_region = [4864 // (4 * settings.bin_pixels[0]),
                         (3 * 4864) // (4 * settings.bin_pixels[0]),
                         (3232 * 2) // settings.bin_pixels[1],
                         (3232 * 3) // settings.bin_pixels[1]]
# Level full stitched by central part of camera 6

settings.mean_region = [0 // settings.bin_pixels[0],
                        3232 // settings.bin_pixels[0],
                        4864 // (4 * settings.bin_pixels[1]),
                        (3 * 4864) // (4 * settings.bin_pixels[1])]
# Compare mean of images using central 50%

settings.min_percentile = 0.1
settings.max_percentile = 99.9

settings.color_scheme = cm.inferno
