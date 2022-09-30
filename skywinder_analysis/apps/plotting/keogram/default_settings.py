from skywinder_analysis.lib.tools import generic

settings = generic.Class()

settings.saved_flat_field_path = None
settings.flat_field_filenames = None

settings.output_name = 'keogram.png'
settings.filenames = []  # use glob to make list of filenames
settings.camera_number = 0
settings.min_percentile = 0.1
settings.max_percentile = 99.9
settings.interval = 15
