import scipy.io, scipy.signal, scipy.ndimage
import numpy.fft as fft
import numpy


def window2d(width, height, window_name='hann'):
    # Makes a circular Hanning window
    win = scipy.signal.get_window(window_name, height)

    half_width = (width - 1) / 2.0
    xx = numpy.linspace(-half_width, half_width, width)

    half_height = (height - 1) / 2.0
    yy = numpy.linspace(-half_height, half_height, height)

    x, y = numpy.meshgrid(xx, yy)
    radius_matrix = numpy.sqrt(x ** 2 + y ** 2)
    window_matrix = numpy.zeros(radius_matrix.shape)

    window_matrix[radius_matrix <= half_height] = \
        numpy.interp(radius_matrix[radius_matrix <= half_height], yy, win)

    return window_matrix


def get_1d_psd(data, window_name='hann'):
    win = scipy.signal.get_window(window_name, len(data))
    data = data * win

    data = fft.fft(data)
    data = fft.fftshift(data)
    data = numpy.abs(data) ** 2
    return data


def get_2d_psd(image):
    height, width = image.shape
    hanning_window = window2d(width, height)
    image = image * hanning_window

    data = fft.fft2(image)
    data = fft.fftshift(data)
    phase = numpy.angle(data)
    data = numpy.abs(data) ** 2
    return data


def get_wavelength_labels(physical_indices, freq_step, multiplier):
    values = []
    labels = []
    for i in physical_indices:
        if i == 0:
            values.append(numpy.nan)
        else:
            values.append(multiplier/(i*freq_step))
    for value in values:
        if numpy.isnan(value):
            labels.append('Inf')
        else:
            labels.append('%.1f' % value)
    print(labels)
    return labels


def get_freq_labels(physical_indices, freq_step, multiplier):
    values = []
    for i in physical_indices:
        values.append(i*freq_step*multiplier)
    labels = ['%.1f' % value for value in values]
    return labels


def get_2d_psd_and_labels(image, m_per_pixel, min_physical_range=5e3, tick_spacing=2, wavelength_labels=False):
    # The pixels are assumed to be square!
    T = m_per_pixel  # sampling interval

    ycenter = image.shape[0]//2
    xcenter = image.shape[1]//2

    psd = fft.fft2(numpy.nan_to_num(image-numpy.nanmean(image)))
    psd = fft.fftshift(psd)

    xN = image.shape[1]
    xfreq_scale = numpy.linspace(0, 1 / T, xN)
    xfreq_step = xfreq_scale[1]
    xmax_physical_index = numpy.searchsorted(xfreq_scale, 1/min_physical_range)+1
    xphysical_indices = range(0, xmax_physical_index, tick_spacing)
    xmaxrange = max(xphysical_indices)

    xindices = [xi + xcenter for xi in xphysical_indices]

    yN = image.shape[0]
    yfreq_scale = numpy.linspace(0, 1 / T, yN)
    yfreq_step = yfreq_scale[1]
    ymax_physical_index = numpy.searchsorted(yfreq_scale, 1/min_physical_range)+1
    yphysical_indices = range(-ymax_physical_index, ymax_physical_index, tick_spacing)
    ymaxrange = max(yphysical_indices)

    yindices = [yi + ycenter for yi in yphysical_indices]

    if wavelength_labels:
        xlabels = get_wavelength_labels(xphysical_indices, xfreq_step, 1e-3)
        ylabels = get_wavelength_labels(yphysical_indices, yfreq_step, 1e-3)
    else:
        xlabels = get_freq_labels(xphysical_indices, xfreq_scale[1], 1e5)
        ylabels = get_freq_labels(yphysical_indices, yfreq_scale[1], 1e5)
    return psd, xfreq_step, yfreq_step, xcenter, ycenter, xmaxrange, ymaxrange, xindices, yindices, xlabels, ylabels


def filter_psd_by_freq(psd, xfreq_step, yfreq_step, low_freq, high_freq):
    x,y = numpy.meshgrid(numpy.arange(psd.shape[1]),numpy.arange(psd.shape[0]))
    x_center = psd.shape[1]/2
    y_center = psd.shape[0]/2
    R = numpy.sqrt(((x-x_center)**2+((yfreq_step/xfreq_step)*(y-y_center))**2))
    Rmin = low_freq/xfreq_step
    Rmax = high_freq/xfreq_step
    filtered_psd = numpy.where((R >= Rmin) & (R <= Rmax),psd,0)
    # Finds where the psd array is between the R values and sets all other elements to zero.
    return filtered_psd
