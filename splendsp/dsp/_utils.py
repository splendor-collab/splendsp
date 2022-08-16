import numpy as np
from scipy import signal, ndimage


__all__ = [
    "shift",
    "make_template",
    "slope",
    "lowpassfilter",
]


def shift(arr, num, fill_value=0):
    """
    Function for shifting the values in an array by a certain number of
    indices, filling the values of the bins at the head or tail of the
    array with fill_value.

    Parameters
    ----------
    arr : array_like
        Array to shift values in.
    num : float
        The number of values to shift by. If positive, values shift to
        the right. If negative, values shift to the left. If num is a
        non-whole number of bins, arr is linearly interpolated
    fill_value : scalar, optional
        The value to fill the bins at the head or tail of the array
        with.

    Returns
    -------
    result : ndarray
        The resulting array that has been shifted and filled in.

    """

    result = np.empty_like(arr)

    if float(num).is_integer():
        num = int(num) # force num to int type for slicing

        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result[:] = arr
    else:
        result = ndimage.shift(
            arr, num, order=1, mode='constant', cval=fill_value,
        )

    return result


def make_template(t, tau_r, tau_f, offset=0):
    """
    Function to make an ideal pulse template in time domain with single
    pole exponential rise and fall times, and a given time offset. The
    template will be returned with the maximum pulse height normalized
    to one. The pulse, by default, begins at the center of the trace,
    which can be left or right shifted via the `offset` optional
    argument.

    Parameters
    ----------
    t : ndarray
        Array of time values to make the pulse with
    tau_r : float
        The time constant for the exponential rise of the pulse
    tau_f : float
        The time constant for the exponential fall of the pulse
    offset : int
        The number of bins the pulse template should be shifted

    Returns
    -------
    template_normed : array
        the pulse template in time domain

    """

    pulse = np.exp(-t/tau_f)-np.exp(-t/tau_r)
    pulse_shifted = shift(pulse, len(t)//2 + offset)
    template_normed = pulse_shifted/pulse_shifted.max()

    return template_normed



def slope(x, y, removemeans=True):
    """
    Computes the maximum likelihood slope of a set of x and y points.

    Parameters
    ----------
    x : array_like
        Array of real-valued independent variables.
    y : array_like
        Array of real-valued dependent variables.
    removemeans : bool
        Boolean flag for if the mean of x should be subtracted. This
        should be set to True if x has not already had its mean
        subtracted. Set to False if the mean has been subtracted.
        Default is True.

    Returns
    -------
    slope : float
        Maximum likelihood slope estimate, calculated as
        sum((x-<x>)(y-<y>))/sum((x-<x>)**2)

    """

    x_mean = np.mean(x) if removemeans else 0

    return np.sum((x - x_mean) * (y - x_mean)) / np.sum((x - x_mean) ** 2)


def lowpassfilter(traces, cut_off_freq=100000, fs=625e3, order=1):
    """
    Applies a low pass filter to the inputted time series traces.

    Parameters
    ----------
    traces : ndarray
        An array of shape (# traces, # bins per trace).
    cut_off_freq : float, int, optional
        The cut off 3dB frequency for the low pass filter, defaults to
        100000 Hz.
    fs : float, int, optional
        Digitization rate of data, defaults to 625e3 Hz.
    order : int, optional
        The order of the low pass filter, defaults to 1.

    Returns
    -------
    filt_traces : ndarray
        Array of low pass filtered traces with the same shape as
        inputted traces.

    """

    nyq = 0.5 * fs
    cut_off = cut_off_freq / nyq
    b,a = signal.butter(order, cut_off)
    filt_traces = signal.filtfilt(b, a, traces, padtype='even')

    return filt_traces

