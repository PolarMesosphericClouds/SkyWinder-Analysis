import numpy as np


def build_stitched_image(imgs):
    stitched_image = np.concatenate(tuple(imgs), axis=0)
    stitched_image = np.rot90(stitched_image, k=1)
    return stitched_image
