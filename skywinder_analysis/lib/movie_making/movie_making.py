import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats
import matplotlib.cm as cm
import time
import binning
import flat_field
import blosc_file
import sys


def ani_frame_from_intermediate_data_products(fns, output_name, bin_pixels=(1, 1), section=None):
    imd = np.zeros((3232, 4864))
    if section:
        imd = imd[section[0]:section[1], section[2]:section[3]]
    dpi = 100
    vmin = 0
    vmax = 0

    frames = len(fns)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if bin_pixels:
        img = binning.bucket(imd, bin_pixels)
    else:
        img = imd

    im = ax.imshow(img, cmap=cm.inferno, interpolation='nearest')

    fig.set_size_inches([12, 8])
    txt = ax.text(0.1, 0.9, 'time:', ha='left', va='top', color='yellow')
    plt.tight_layout()
    start_at = time.time()
    ax = plt.gca()

    def update_img(n):
        elapsed = time.time() - start_at
        if n:
            time_per_frame = elapsed / n
        else:
            time_per_frame = 1
        sys.stdout.flush()
        try:
            img = np.load(fns[n])
        except Exception:
            return
        if section:
            img = img[section[0]:section[1], section[2]:section[3]]
        if bin_pixels:
            img = binning.bucket(img, bin_pixels)
        im.set_data(img)

        # if vmin == 0:
        vmin = scipy.stats.scoreatpercentile(img, 0.1)
        vmax = scipy.stats.scoreatpercentile(img, 99.9)
        im.set_clim(vmin, vmax)

        try:
            time_string = fns[n].split('/')[-1].split('_')[1]
            time_string = time_string[0:2] + ':' + time_string[2:4] + ':' + time_string[4:]
            txt.set_text(time_string)
        except Exception:
            pass
        print("\r%d of %d %.1f minutes elapsed, %.1f minutes remaining" % (n, len(fns), elapsed / 60,
                                                                           (len(fns) - n) * time_per_frame / 60))

        return im, txt

    ani = animation.FuncAnimation(fig, update_img, frames, interval=100)
    ani.save(output_name, dpi=dpi)
    return ani


def ani_frame_from_raw(fns, output_name, saved_flat_field=None, ff_fns=None, bin_pixels=(1, 1), section=None):
    imd = np.zeros((3232, 4864))
    if section:
        imd = imd[section[0]:section[1], section[2]:section[3]]
    dpi = 100
    vmin = 0
    vmax = 0

    frames = len(fns)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if bin_pixels:
        img = binning.bucket(imd, bin_pixels)
    else:
        img = imd

    im = ax.imshow(img, cmap=cm.inferno, interpolation='nearest')

    fig.set_size_inches([12, 8])
    txt = ax.text(0.1, 0.9, 'time:', ha='left', va='top', color='yellow')
    plt.tight_layout()
    start_at = time.time()
    ax = plt.gca()

    if saved_flat_field:
        print('Loading saved flat field %s' % saved_flat_field)
        flat_field_image = np.load(saved_flat_field)
        use_flat_field_image = True
    elif ff_fns:
        print('Generating flat field')
        flat_field_image = flat_field.generate_flat_field_image_from_files(ff_fns)
        use_flat_field_image = True
        # Generate flat_field_image from filenames
    else:
        print = ('No flat field used')
        use_flat_field_image = False

    def update_img(n):
        elapsed = time.time() - start_at
        if n:
            time_per_frame = elapsed / n
        else:
            time_per_frame = 1
        sys.stdout.flush()
        try:
            img, _ = blosc_file.load_blosc_image(fns[n])
            if use_flat_field_image:
                img = flat_field.apply_flat_field(img, flat_field_image)
        except Exception as e:
            print(e)
            return
        if section:
            img = img[section[0]:section[1], section[2]:section[3]]
        if bin_pixels:
            img = binning.bucket(img, bin_pixels)
        im.set_data(img)

        # if vmin == 0:
        vmin = scipy.stats.scoreatpercentile(img, 0.1)
        vmax = scipy.stats.scoreatpercentile(img, 99.9)
        im.set_clim(vmin, vmax)

        try:
            time_string = fns[n].split('/')[-1].split('_')[1]
            time_string = time_string[0:2] + ':' + time_string[2:4] + ':' + time_string[4:]
            txt.set_text(time_string)
        except Exception:
            pass
        print("\r%d of %d %.1f minutes elapsed, %.1f minutes remaining" % (n, len(fns), elapsed / 60,
                                                                           (len(fns) - n) * time_per_frame / 60))

        return im, txt

    ani = animation.FuncAnimation(fig, update_img, frames, interval=100)
    ani.save(output_name, dpi=dpi)
    return ani


if __name__ == "__main__":
    import glob

    fns = glob.glob(sys.argv[1])
    # Include quotes around the path
    output_name = sys.argv[2]
    ff_fns = fns[::10]
    ani_frame_from_raw(fns, output_name, saved_flat_field=None, ff_fns=ff_fns, bin_pixels=(4, 4), section=None)
