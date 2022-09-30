from skywinder_analysis.lib.tools import generic
from skywinder_analysis.lib.image_processing import flat_field
import matplotlib.animation as animation

settings = generic.Class()

settings.pointing_dir = ''
settings.camera_numbers = [4, 5, 6, 7]
settings.bin_factor = 16
settings.reflection_window = 60
settings.min_percentile = 1
settings.max_percentile = 99
settings.clipped = 4400

settings.output_name = 'movie.mp4'
settings.writer = animation.writers['ffmpeg'](fps=10, codec='h264', bitrate=2 ** 20)
settings.dpi = 100
settings.start_time = 1531350000
settings.end_time = 1531353600
settings.interval = 12
settings.clipped = False
settings.limits = [(0,100), (0,100)]
