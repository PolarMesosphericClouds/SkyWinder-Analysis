from skywinder_analysis.lib.tools import generic
import matplotlib.animation as animation
import matplotlib.cm as cm

settings = generic.Class()

settings.section = None
settings.bin_pixels = (4, 4)
settings.dpi = 100

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

settings.crop = False

settings.saved_flat_field_paths = False
settings.all_flat_field_filenames = False
settings.use_new_flat_field = False
settings.new_flat_field_reflection_window = 3000

settings.mean_region = [0 // settings.bin_pixels[0],
                        3232 // settings.bin_pixels[0],
                        4864 // (4 * settings.bin_pixels[1]),
                        (3 * 4864) // (4 * settings.bin_pixels[1])]
# Compare mean of images using central 50%

settings.color_scheme = cm.inferno
# Color scheme of image
settings.fontcolor = 'black'

settings.colorscale_by_frame = True
settings.vmin = False
settings.vmax = False
settings.individual_frames = False
settings.split_video_len = False
settings.font_proportion = 1./32.
settings.no_text = False
settings.rotate = False
settings.piggyback = False
