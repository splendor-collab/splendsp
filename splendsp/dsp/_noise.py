import numpy as np
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq


__all__ = [
    "foldpsd",
    "calc_psd",
    "gen_noise",
]


def foldpsd(psd, fs):
    """
    Return the one-sided version of the inputted two-sided psd.
    
    Parameters
    ----------
    psd : ndarray
        A two-sided psd to be converted to one-sided
    fs : float
        The sample rate used for the psd
            
    Returns
    -------
    f : ndarray
        The frequencies corresponding to the outputted one-sided psd
    psd_folded : ndarray
        The one-sided (folded over) psd corresponding to the inputted two-sided psd
            
    """
    
    psd_folded = np.copy(psd[:len(psd)//2+1])
    psd_folded[1:len(psd)//2+(len(psd))%2] *= 2.0
    f = rfftfreq(len(psd),d=1.0/fs)
    
    return f, psd_folded


def calc_psd(x, fs=1.0, folded_over=True):
    """Return the PSD of an n-dimensional array, assuming that we want the PSD of the last axis.
    
    Parameters
    ----------
    x : array_like
        Array to calculate PSD of.
    fs : float, optional
        Sample rate of the data being taken, assumed to be in units of Hz.
    folded_over : bool, optional
        Boolean value specifying whether or not the PSD should be folded over. 
        If True, then the symmetric values of the PSD are multiplied by two, and
        we keep only the positive frequencies. If False, then the entire PSD is 
        saved, including positive and negative frequencies. Default is to fold
        over the PSD.
            
    Returns
    -------
    f : ndarray
        Array of sample frequencies
    psd : ndarray
        Power spectral density of 'x'
        
    """
    
    # calculate normalization for correct units
    norm = fs * x.shape[-1]
    
    if folded_over:
        # if folded_over = True, we calculate the Fourier Transform for only the positive frequencies
        if len(x.shape)==1:
            psd = (np.abs(rfft(x))**2.0)/norm
        else:
            psd = np.mean(np.abs(rfft(x))**2.0, axis=0)/norm
            
        # multiply the necessary frequencies by two (zeroth frequency should be the same, as
        # should the last frequency when x.shape[-1] is odd)
        psd[1:x.shape[-1]//2+1 - (x.shape[-1]+1)%2] *= 2.0
        f = rfftfreq(x.shape[-1], d=1.0/fs)
    else:
        # if folded_over = False, we calculate the Fourier Transform for all frequencies
        if len(x.shape)==1:
            psd = (np.abs(fft(x))**2.0)/norm
        else:
            psd = np.mean(np.abs(fft(x))**2.0, axis=0)/norm
            
        f = fftfreq(x.shape[-1], d=1.0/fs)
    return f, psd


def gen_noise(psd, fs=1.0, ntraces=1):
    """
    Function to generate noise traces with random phase from a given PSD. The PSD calculated from
    the generated noise traces should be the equivalent to the inputted PSD as the number of traces
    goes to infinity.
    
    Parameters
    ----------
    psd : ndarray
        The two-sided power spectral density that will be used to generate the noise.
    fs : float, optional
        Sample rate of the data being taken, assumed to be in units of Hz.
    ntraces : int, optional
        The number of noise traces that should be generated. Default is 1.
    
    Returns
    -------
    noise : ndarray
        An array containing all of the generated noise traces from the inputted PSD. Has shape
        (ntraces, len(psd)). 
    
    """
    
    if np.isinf(psd[0]):
        psd[0] = 0
    
    traces = np.zeros((ntraces, len(psd)))
    vals = np.random.randn(ntraces, len(psd))
    noisefft = fft(vals) * np.sqrt(psd*fs)
    noise = ifft(noisefft).real
    
    return noise
