from skywinder_analysis.lib.tools import generic
from skywinder_analysis.lib.image_processing import flat_field

settings = generic.Class()

settings.cam_nums = [4, 5, 6, 7]
settings.bin_factor = 4
settings.dpi = 100
settings.saved_flat_field_path = None
settings.all_flat_field_filenames = None
settings.new_flat_field_window = 3000

settings.relevel_images = True

settings.pointing_directory = ''

settings.output_name = 'image.png'
settings.all_filenames = {}  # use glob to make list of filenames
settings.min_percentile = 0.1
settings.max_percentile = 99.9

settings.min_x = -100
settings.max_x = 100
settings.x_increment = 100

settings.min_y = 0
settings.max_y = 115
settings.y_increment = 100

settings.naive_flat_field = True
settings.new_flat_field = False

settings.level_region = [0 // settings.bin_factor,
                         3232 // settings.bin_factor,
                         4864 // (4 * settings.bin_factor),
                         (3*4864) // (4*settings.bin_factor)]
