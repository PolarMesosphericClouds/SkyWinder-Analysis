from skywinder_analysis.lib.tools import generic
import numpy as np
from skywinder_analysis.lib.image_processing import flat_field
import matplotlib.animation as animation
import matplotlib.cm as cm

settings = generic.Class()

settings.writer = animation.writers['ffmpeg'](fps=10, codec='h264', bitrate=2 ** 20)
settings.cam_nums = [4, 5, 6, 7]
settings.bin_factor = 4
settings.dpi = 100
settings.saved_flat_field_path = None
settings.all_flat_field_filenames = None
settings.new_flat_field_window = 600

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

settings.xvalues = np.linspace(settings.min_x, settings.max_x, num=settings.x_increment)
settings.yvalues = np.linspace(settings.min_y, settings.max_y, num=settings.y_increment)

settings.provided_grid_map = False
settings.level_region = [0 // settings.bin_factor,
                         3232 // settings.bin_factor,
                         4864 // (4 * settings.bin_factor),
                         (3*4864) // (4*settings.bin_factor)]

settings.individual_frames = False
settings.grid = False
settings.grid_spacing = 25

settings.color_scheme = cm.inferno
settings.timestamp_in_image = True
settings.text_color = 'black'
settings.use_new_flat_fields = True
settings.flat_field_filenames = False
settings.wireframes = False
settings.narrowfield_wireframe_solution_directories = {}
settings.centerline = 100
