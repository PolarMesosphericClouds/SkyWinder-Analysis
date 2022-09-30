from skywinder_analysis.lib.tools import generic
import matplotlib.animation as animation

settings = generic.Class()

settings.section_x = 0
settings.section_y = 0
settings.section_xwidth = 500
settings.section_ywidth = 500

settings.bin_pixels = (4, 4)
settings.feature_bin_pixels = (2,2)
settings.dpi = 100
settings.saved_flat_field_path = None
settings.flat_field_filenames = None

settings.raw_image_size = (3232, 4864)
settings.output_name = 'movie.mp4'
settings.filenames = []  # use glob to make list of filenames
settings.writer = animation.writers['ffmpeg'](fps=20, codec='h264', bitrate=2 ** 20)
settings.camera_number = 0
settings.rotate = False
if settings.rotate:
    settings.level_region = [4864 // (4 * settings.bin_pixels[0]),
                             4864 // (settings.bin_pixels[0]),
                             0 // settings.bin_pixels[1],
                             3232 // settings.bin_pixels[1]]
else:
    settings.level_region = [0 // settings.bin_pixels[0],
                             3232 // settings.bin_pixels[0],
                             4864 // (4 * settings.bin_pixels[1]),
                             (3*4864 // (4*settings.bin_pixels[1]))]
settings.min_percentile = 0.1
settings.max_percentile = 99.9
settings.xshift = 1
settings.yshift = 1
