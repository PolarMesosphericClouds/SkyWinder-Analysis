import numpy as np

from skywinder_analysis.lib.tools import blosc_file
from skywinder_analysis.lib.image_processing import flat_field
from collections import deque


class RollingFlatField():
    def __init__(self, size=10):
        self.images_for_flat_field = deque(maxlen=size)
        self.size = size

    def roll(self, image_filename):
        if len(self.images_for_flat_field) == self.size:
            popped = self.images_for_flat_field.popleft()
        image, _ = blosc_file.load_blosc_image(image_filename)
        self.images_for_flat_field.append(image)

    def generate_flat_field(self):
        return flat_field.generate_flat_field_image_from_raw(self.images_for_flat_field)
