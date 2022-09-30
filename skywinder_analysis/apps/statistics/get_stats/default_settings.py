from skywinder_analysis.lib.tools import generic

settings = generic.Class()

settings.section = None
settings.dpi = 100
settings.rolling_flat_field_size = 20

settings.camera_number = 0

settings.raw_image_size = (3232, 4864)
settings.output_name = 'camera_stats.txt'
settings.filenames = []  # use glob to make list of filenames
