import numpy as np
from numpy.lib.stride_tricks import as_strided
import warnings


def view_as_blocks(arr_in, block_shape):
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length "
                         "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    # -- restride the array to build the block view

    if not arr_in.flags.contiguous:
        warnings.warn(RuntimeWarning("Cannot provide views on a non-contiguous input "
                                     "array without copying."))

    # Taken from https://github.com/scikit-image/scikit-image/blob/master/skimage/util/shape.py#l10
    arr_in = np.ascontiguousarray(arr_in)

    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out


def bucket(x, bucket_size):
    # from https://stackoverflow.com/questions/36269508/lets-make-a-reference-implementation-of-n-dimensional-pixel-binning-bucketing-f
    blocks = view_as_blocks(x, bucket_size)
    tup = tuple(range(-len(bucket_size), 0))
    # return blocks.sum(axis=tup)
    return blocks.mean(axis=tup)
