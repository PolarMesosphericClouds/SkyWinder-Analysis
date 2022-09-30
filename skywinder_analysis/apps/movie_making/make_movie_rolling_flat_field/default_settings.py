from skywinder_analysis.lib.tools import generic
import matplotlib.animation as animation

settings = generic.Class()

settings.section = None
settings.bin_pixels = (4, 4)
settings.dpi = 100
settings.rolling_flat_field_size = 20

settings.camera_number = 0

settings.raw_image_size = (3232, 4864)
settings.output_name = 'movie.mp4'
settings.filenames = []  # use glob to make list of filenames
settings.writer = animation.writers['ffmpeg'](fps=10, codec='h264', bitrate=2**20)
settings.level_region = [0//settings.bin_pixels[0],3232//settings.bin_pixels[0],0//settings.bin_pixels[1],4864//settings.bin_pixels[1]]
