import numpy as np


__all__ = [
    "passagefraction",
    "inrange",
]


def passagefraction(x, cut, basecut=None, nbins=100,
                    lgcequaldensitybins=False):
    """
    Function for returning the passage fraction of a cut as a function
    of the specified variable `x`.

    Parameters
    ----------
    x : array_like
        Array of values to be binned and plotted
    cut : array_like
        Mask of values to calculate passage fraction for.
    basecut : NoneType, array_like, optional
        A cut to use as the comparison for the passage fraction. If
        left as None, then the passage fraction is calculated using
        all of the inputted data.
    nbins : int, optional
        The number of bins that should be created. Default is 100.
    lgcequaldensitybins : bool, optional
        If set to True, the bin widths are set such that each bin has
        the same number of data points within it. If left as False,
        then a constant bin width is used.

    Returns
    -------
    x_binned : ndarray
        The corresponding `x` values for each passage fraction, given
        as the edges of each bin.
    frac_binned_mp : ndarray
        The most probable (measured) passage fraction for each value
        of `x_binned` for the given `cut` and `basecut`.
    frac_binned_biased : ndarray
        The expected value of the passage fraction for each value of
        `x_binned` for the given `cut` and `basecut`. See Notes for
        more information.
    frac_binned_err : ndarray
        The standard deviation of the passage fraction for each value
        of `x_binned` for the given `cut` and `basecut`, where this is
        centered on the values of `frac_binned_biased`. See Notes for
        more information.

    Notes
    -----
    The errors are based on the derivation in
        https://arxiv.org/pdf/physics/0701199v1.pdf.

    Let \eps be the passage fraction. Then we have that:

    P(\eps | n, k) = (n + 1)! / (k! (n - k)!) \eps^k (1 - \eps)^(n - k)

    E[\eps] = Integral(\eps * P(\eps | n, k), 0, 1) = (k + 1) / (n + 2)

    This is a biased estimator of the efficiency. An unbiased estimator
    is the solution  to dP/d\eps = 0, which is \eps_{mp} = k / n
    (the same as the measured passage fraction).

    The variance is given by:

    V[\eps] = E[\eps^2] - E[\eps]^2 = (k + 1) / (n + 2) *
        ( (k + 2) / (n + 3) - (k + 1) / (n + 2))

    Then the standard deviation is \sigma_{\eps} = \sqrt{V[\eps]}.

    """

    if basecut is None:
        basecut = np.ones(len(x), dtype=bool)

    if lgcequaldensitybins:
        histbins_equal = lambda var, nbin: np.interp(
            np.linspace(0, len(var), nbin + 1),
            np.arange(len(var)),
            np.sort(var),
        )
        nbins = histbins_equal(x[basecut], nbins)

    hist_vals_base, x_binned = np.histogram(x[basecut], bins=nbins)
    hist_vals, _ = np.histogram(x[basecut & cut], bins=x_binned)

    frac_binned_biased = (hist_vals + 1)/(hist_vals_base + 2)
    frac_binned_err = np.sqrt(
        frac_binned_biased * ((hist_vals + 2)/(hist_vals_base + 3) - frac_binned_biased)
    )

    hist_vals_base[hist_vals_base==0] = 1

    frac_binned_mp = hist_vals/hist_vals_base

    return x_binned, frac_binned_mp, frac_binned_biased, frac_binned_err


def inrange(vals, lwrbnd, uprbnd, include_lwrbnd=True, include_uprbnd=True):
    """
    Function for returning a boolean mask that specifies which values
    in an array are between the specified bounds (inclusive of the
    bounds).

    Parameters
    ----------
    vals : ndarray
        A 1-d ndarray of values.
    lwrbnd : float
        The lower bound of the range that we are checking if vals is
        between.
    uprbnd : float
        The upper bound of the range that we are checking if vals is
        between.
    include_lwrbnd : bool, optional
        Boolean flag for including or excluding the lower bound in the
        range. Default is True, meaning that we include the lower bound
        in the specified range.
    include_uprbnd : bool, optional
        Boolean flag for including or excluding the upper bound in the
        range. Default is True, meaning that we include the upper bound
        in the specified range.

    Returns
    -------
    mask : ndarray
        A boolean array of the same shape as vals. True means that the
        value was between the bounds, False means that the value was
        not.

    """

    return (
        vals >= lwrbnd if include_lwrbnd else vals > lwrbnd
    ) & (
        vals <= uprbnd if include_uprbnd else vals < uprbnd
    )
