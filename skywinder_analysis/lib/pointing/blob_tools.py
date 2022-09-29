from skywinder_analysis.lib.pointing import blobs
import matplotlib.pyplot as plt
import numpy as np


def filter_low_contrast_blobs(image, blobs, m=1, debug=False):
    good_blobs = []
    mask = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    for blob in blobs:
        tiny_stamp = image[blob.x - 2:blob.x + 1, blob.y - 2:blob.y + 1]
        if tiny_stamp.shape == (3, 3):
            tiny_stamp = np.ma.masked_array(tiny_stamp, mask=mask)
            stamp = image[blob.x - 16:blob.x + 16, blob.y - 16:blob.y + 16]
            if debug:
                print('Local mean:', np.mean(tiny_stamp))
                print('Stamp mean:', np.mean(stamp))
                print('Stamp std:', np.std(stamp))
                print('Difference in means', np.mean(tiny_stamp) - np.mean(stamp))

            if np.mean(tiny_stamp) > (np.mean(stamp) + m * np.std(stamp)):
                good_blobs.append(blob)
    return good_blobs


def find_blobs(image, show_blobs=True, fit_blobs=True, do_low_contrast_filter=True, m=1):
    bf = blobs.BlobFinder(image, fit_blobs=fit_blobs)  # ,cell_size=64,blob_threshold=12,fit_blobs=False,kernel_sigma=1)
    myblobs = bf.blobs
    if do_low_contrast_filter == True:
        myblobs = filter_low_contrast_blobs(image, myblobs, m, debug=False)
    if show_blobs == True:
        fig = plt.figure(figsize=(9, 9))
        axs = fig.add_subplot(4, 8)
        for k, ax in enumerate(axs.flatten()):
            if k >= len(myblobs):
                break
            blob = myblobs[k]
            ax.imshow(image[blob.x - 16:blob.x + 16, blob.y - 16:blob.y + 16], cmap=plt.cm.hot, interpolation='nearest')
            ax.set_title("X: %d\nY: %d" % (blob.x, blob.y))
    return myblobs
