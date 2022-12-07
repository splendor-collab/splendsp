import numpy as np
from scipy.optimize import least_squares
from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq
import matplotlib.pyplot as plt

from splendsp.dsp._utils import shift


__all__ = [
    "OptimalFilter",
    "OFnonlin",
]


def _argmin_chi2(chi2, nconstrain=None, lgcoutsidewindow=False,
                 constraint_mask=None, windowcenter=0):
    """
    Helper function for finding the index for the minimum of a chi^2.
    Includes options for constraining the values of chi^2.

    Parameters
    ----------
    chi2 : ndarray
        An array containing the chi^2 to minimize. If `chi2` has
        dimension greater than 1, then it is minimized along the last
        axis.
    nconstrain : NoneType, int, optional
        This is the length of the window (in bins) out of which to
        constrain the possible values to in the chi^2 minimization,
        centered on the middle value of `chi2`. Default is None,
        where `chi2` is uncontrained.
    lgcoutsidewindow : bool, optional
        If False, then the function will minimize the chi^2 in the bins
        inside the constrained window specified by `nconstrain`, which
        is the default behavior. If True, the function will minimize
        the chi^2 in the bins outside the range specified by
        `nconstrain`.
    constraint_mask : NoneType, boolean ndarray, optional
        An additional constraint on the chi^2 to apply, which should be
        in the form of a boolean mask. If left as None, no additional
        constraint is applied.
    windowcenter : int, optional
        The bin, relative to the center bin of the trace, on which the
        delay window specified by `nconstrain` is centered. Default of
        0 centers the delay window in the center of the trace.
        Equivalent to centering the `nconstrain` window on
        `chi2.shape[-1]//2 + windowcenter`.

    Returns
    -------
    bestind : int, ndarray, float
        The index of the minimum of `chi2` given the constraints
        specified by `nconstrain` and `lgcoutsidewindow`. If the
        dimension of `chi2` is greater than 1, then this will be an
        ndarray of ints.

    """

    nbins = chi2.shape[-1]

    if not -(nbins//2) <= windowcenter <= nbins//2 - (nbins+1)%2:
        raise ValueError(
            f"windowcenter must be between {-(nbins//2)} "
            f"and {nbins//2 - (nbins + 1)%2}"
        )

    if nconstrain is not None:
        if nconstrain>nbins:
            nconstrain = nbins
        elif nconstrain <= 0:
            raise ValueError(
                f"nconstrain must be a positive integer less than {nbins}"
            )

        win_start = nbins//2 - nconstrain//2 + windowcenter
        if lgcoutsidewindow:
            win_end = -nbins//2 + nconstrain//2 + nconstrain%2 + windowcenter
            inds = np.r_[0:win_start, win_end:0]
            inds[inds < 0] += nbins
        else:
            win_end = nbins//2 + nconstrain//2 + nconstrain%2 + windowcenter
            inds = np.arange(win_start, win_end)
            inds = inds[(inds>=0) & (inds<nbins)]

        if constraint_mask is not None:
            inds = inds[constraint_mask[inds]]
        if len(inds)!=0:
            bestind = np.argmin(chi2[..., inds], axis=-1)
            bestind = inds[bestind]
        else:
            bestind = np.nan
    else:
        if constraint_mask is None:
            bestind = np.argmin(chi2, axis=-1)
        else:
            inds = np.flatnonzero(constraint_mask)
            if len(inds)!=0:
                bestind = np.argmin(chi2[..., constraint_mask], axis=-1)
                bestind = inds[bestind]
            else:
                bestind = np.nan

    return bestind


def _get_pulse_direction_constraint_mask(amps, pulse_direction_constraint=0):
    """
    Helper function for returning the constraint mask for positive or
    negative-only going pulses.

    Parameters
    ----------
    amps : ndarray
        Array of the OF amplitudes to use when getting the
        `constraint_mask`.
    pulse_direction_constraint : int, optional
        Sets a constraint on the direction of the fitted pulse. If 0,
        then no constraint on the pulse direction is set. If 1, then a
        positive pulse constraint is set for all fits. If -1, then a
        negative pulse constraint is set for all fits. If any other
        value, then an ValueError will be raised. 

    Returns
    -------
    constraint_mask : NoneType, ndarray
        If no constraint is set, this is set to None. If
        `pulse_direction_constraint` is 1 or -1, then this is the
        boolean array of the constraint.

    """

    if pulse_direction_constraint not in (-1, 0, 1):
        raise ValueError(
            "pulse_direction_constraint should be set to 0, 1, or -1",
        )

    if pulse_direction_constraint == 0:
        return None

    return pulse_direction_constraint * amps > 0


class OptimalFilter(object):
    """
    Class for efficient calculation of the various different
    Optimal Filters. Written to minimize the amount of repeated
    computations when running multiple on the same data.

    Attributes
    ----------
    psd : ndarray
        The two-sided psd that will be used to describe the noise in
        the signal (in Amps^2/Hz)
    psd0 : float
        The value of the inputted PSD at the zero frequency bin. Used
        for ofamp_baseline in the case that `OptimalFilter` is
        initialized with "AC" coupling.
    nbins : int
        The length of the trace/psd/template in bins.
    fs : float
        The sample rate of the data being taken (in Hz).
    df : float
        Equivalent to df/nbins, the frequency spacing of the Fourier
        Tranforms.
    s : ndarray
        The template converted to frequency space, with the
        normalization specified by the `integralnorm` parameter in the
        initialization.
    phi : ndarray
        The optimal filter in frequency space.
    norm : float
        The normalization for the optimal filtered signal.
    v : ndarray
        The signal converted to frequency space.
    signalfilt : ndarray
        The optimal filtered signal in frequency space.
    chi0 : float
        The chi^2 value for just the signal part.
    chit_withdelay : ndarray
        The fitting part of the chi^2 for `ofamp_withdelay`.
    amps_withdelay : ndarray
        The possible amplitudes for `ofamp_withdelay`.
    chi_withdelay : ndarray
        The full chi^2 for `ofamp_withdelay`.
    signalfilt_td : ndarray
        The filtered signal converted back to time domain.
    templatefilt_td : ndarray
        The filtered template converted back to time domain.
    times : ndarray
        The possible time shift values.
    freqs : ndarray
        The frequencies matching the Fourier Transform of the data.

    """

    def __init__(self, signal, template, psd, fs, coupling="AC",
                 integralnorm=False):
        """
        Initialization of the OptimalFilter class.

        Parameters
        ----------
        signal : ndarray
            The signal that we want to apply the optimal filter to
            (units should be Amps).
        template : ndarray
            The pulse template to be used for the optimal filter
            (should be normalized to a max height of 1 beforehand).
        psd : ndarray
            The two-sided psd that will be used to describe the noise
            in the signal (in Amps^2/Hz)
        fs : ndarray
            The sample rate of the data being taken (in Hz).
        coupling : str, optional
            String that determines if the zero frequency bin of the psd
            should be ignored (i.e. set to infinity) when calculating
            the optimal amplitude. If set to 'AC', then ths zero
            frequency bin is ignored. If set to anything else, then the
            zero frequency bin is kept. Default is 'AC'.
        integralnorm : bool, optional
            If set to True, then `OptimalFilter` will normalize the
            template to an integral of 1, and any optimal filters will
            instead return the optimal integral in units of Coulombs.
            If set to False, then the usual optimal filter amplitudes
            will be returned (in units of Amps).

        """

        self.psd = np.zeros(len(psd))
        self.psd[:] = psd
        self.psd0 = psd[0]

        if coupling=="AC":
            self.psd[0] = np.inf

        self.nbins = signal.shape[-1]
        self.fs = fs
        self.df = self.fs / self.nbins

        self.s = fft(template) / self.nbins / self.df

        if integralnorm:
            self.s /= self.s[0]

        self.phi = self.s.conjugate() / self.psd
        self.norm = np.real(np.dot(self.phi, self.s)) * self.df

        self.v = fft(signal, axis=-1) / self.nbins / self.df
        self.signalfilt = self.phi * self.v / self.norm

        self.chi0 = None

        self.chit_withdelay = None
        self.amps_withdelay = None
        self.chi_withdelay = None

        self.signalfilt_td = None
        self.templatefilt_td = None

        self.times = None
        self.freqs = None

    def _check_freqs(self):
        """
        Hidden method for checking if we have initialized the FFT
        frequencies.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        if self.freqs is None:
            self.freqs = fftfreq(self.nbins, d=1.0/self.fs)

    @staticmethod
    def _interpolate_parabola(vals, bestind, delta, t_interp=None):
        """
        Precomputed equation of a parabola given 3 equally spaced
        points. Returns the coordinates of the extremum of the
        parabola.

        """

        sf = 1 / (2 * delta**2)

        a = sf * (vals[bestind + 1] - 2 * vals[bestind] + vals[bestind - 1])
        b = sf * delta * (vals[bestind + 1] - vals[bestind - 1])
        c = sf * 2 * delta**2 * vals[bestind]

        if t_interp is None:
            t_interp = - b / (2 * a)
        vals_interp = a * t_interp**2 + b * t_interp + c

        return t_interp, vals_interp

    @staticmethod
    def _interpolate_of(amps, chi2, bestind, delta):
        """
        Helper function for running `_interpolate_parabola` twice,
        in the correct order.

        """
    
        t_interp, chi2_interp = OptimalFilter._interpolate_parabola(
            chi2, bestind, delta,
        )
        _, amps_interp = OptimalFilter._interpolate_parabola(
            amps, bestind, delta, t_interp=t_interp,
        )

        return amps_interp, t_interp, chi2_interp

    def update_signal(self, signal):
        """
        Method to update `OptimalFilter` with a new signal if the PSD
        and template are to remain the same.

        Parameters
        ----------
        signal : ndarray
            The signal that we want to apply the optimal filter to
            (units should be Amps).

        """

        self.v = fft(signal, axis=-1)/self.nbins/self.df
        self.signalfilt = self.phi * self.v / self.norm

        self.chi0 = None
        self.chit_withdelay = None
        self.signalfilt_td = None
        self.amps_withdelay = None
        self.chi_withdelay = None

    def energy_resolution(self):
        """
        Method to return the energy resolution for the optimal filter.

        Returns
        -------
        sigma : float
            The energy resolution of the optimal filter.

        """

        sigma = 1.0 / np.sqrt(self.norm)

        return sigma

    def time_resolution(self, amp):
        """
        Method to return the time resolution for the optimal filter for
        a specific fit.

        Parameters
        ----------
        amp : float
            The OF amplitude of the fit to use in the time resolution
            calculation.

        Returns
        -------
        sigma : float
            The time resolution of the optimal filter.

        """

        self._check_freqs()

        sigma = 1.0 / np.sqrt(
            amp**2 * np.sum(
                (2 * np.pi * self.freqs)**2 * np.abs(self.s)**2 / self.psd
            ) * self.df
        )

        return sigma


    def chi2_nopulse(self):
        """
        Method to return the chi^2 for there being no pulse in the signal.

        Returns
        -------
        chi0 : float
            The chi^2 value for there being no pulse.

        """

        # signal part of chi2
        if self.chi0 is None:
            self.chi0 = np.real(
                np.dot(self.v.conjugate() / self.psd, self.v) * self.df
            )

        return self.chi0

    def chi2_lowfreq(self, amp, t0, fcutoff=10000):
        """
        Method for calculating the low frequency chi^2 of the optimal
        filter, given some cut off frequency.

        Parameters
        ----------
        amp : float
            The optimal amplitude calculated for the trace (in Amps).
        t0 : float
            The time shift calculated for the pulse (in s).
        fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz.

        Returns
        -------
        chi2low : float
            The low frequency chi^2 value (cut off at fcutoff) for the
            inputted values.

        """

        self._check_freqs()

        chi2tot = self.df * np.abs(
            self.v - amp * np.exp(-2.0j * np.pi * t0 * self.freqs) * self.s
        )**2 / self.psd

        chi2inds = np.abs(self.freqs) <= fcutoff

        chi2low = np.sum(chi2tot[chi2inds])

        return chi2low

    def ofamp_nodelay(self, windowcenter=0):
        """
        Function for calculating the optimal amplitude of a pulse in
        data with no time shifting, or at a specific time.

        Parameters
        ----------
        windowcenter : int, optional
            The bin, relative to the center bin of the trace, at which
            to calculate the OF amplitude. Default of 0 calculates the
            usual no delay optimal filter. Equivalent to calculating
            the OF amplitude at the bin `self.nbins//2 + windowcenter`.
            Useful for calculating amplitudes at specific times, if
            there is some prior knowledge.

        Returns
        -------
        amp : float
            The optimal amplitude calculated for the trace (in Amps)
            with no time shifting allowed (or at the time specified by
            `windowcenter`).
        chi2 : float
            The chi^2 value calculated from the optimal filter with no
            time shifting (or at the time specified by `windowcenter`).

        """

        if windowcenter != 0:
            self._check_freqs()
            t0 = windowcenter / self.fs
            amp = np.real(
                np.sum(
                    self.signalfilt * np.exp(2.0j * np.pi * t0 * self.freqs),
                    axis=-1,
                )
            ) * self.df
        else:
            amp = np.real(np.sum(self.signalfilt, axis=-1)) * self.df

        # signal part of chi2
        if self.chi0 is None:
            self.chi0 = np.real(
                np.dot(self.v.conjugate() / self.psd, self.v) * self.df
            )

        # fitting part of chi2
        chit = (amp**2) * self.norm

        chi2 = self.chi0 - chit

        return amp, chi2

    def ofamp_withdelay(self, nconstrain=None, lgcoutsidewindow=False,
                        pulse_direction_constraint=0, windowcenter=0,
                        interpolate_t0=False):
        """
        Function for calculating the optimal amplitude of a pulse in
        data with time delay.

        Parameters
        ----------
        nconstrain : int, NoneType, optional
            The length of the window (in bins) to constrain the
            possible t0 values to. By default centered on the unshifted
            trigger, non-default center choosen with windowcenter. If
            left as None, then t0 is uncontrained. If `nconstrain` is
            larger than `self.nbins`, then the function sets
            `nconstrain` to `self.nbins`,  as this is the maximum
            number of values that t0 can vary over.
        lgcoutsidewindow : bool, optional
            Boolean flag that is used to specify whether the optimal
            Filter should look inside `nconstrain` or outside it. If
            False, the filter will minimize the chi^2 in the bins
            specified by `nconstrain`, which is the default behavior.
            If True, then it will minimize the chi^2 in the bins that
            do not contain the constrained window.
        pulse_direction_constraint : int, optional
            Sets a constraint on the direction of the fitted pulse.
            If 0, then no constraint on the pulse direction is set.
            If 1, then a positive pulse constraint is set for all fits.
            If -1, then a negative pulse constraint is set for all
            fits. If any other value, then a ValueError will be raised.
        windowcenter : int, optional
            The bin, relative to the center bin of the trace, on which
            the delay window specified by `nconstrain` is centered.
            Default of 0 centers the delay window in the center of the
            trace. Equivalent to centering the `nconstrain` window on
            `self.nbins//2 + windowcenter`.
        interpolate_t0 : bool, optional
            If True, then a precomputed solution to the parabolic
            equation is used to find the interpolated time-of-best-fit.
            Default is False.

        Returns
        -------
        amp : float
            The optimal amplitude calculated for the trace (in Amps).
        t0 : float
            The time shift calculated for the pulse (in s).
        chi2 : float
            The chi^2 value calculated from the optimal filter.

        """

        if self.signalfilt_td is None:
            self.signalfilt_td = np.real(
                ifft(self.signalfilt * self.nbins, axis=-1)
            ) * self.df

        # signal part of chi2
        if self.chi0 is None:
            self.chi0 = np.real(
                np.dot(self.v.conjugate() / self.psd, self.v) * self.df
            )

        # fitting part of chi2
        if self.chit_withdelay is None:
            self.chit_withdelay = (self.signalfilt_td**2) * self.norm

        # sum parts of chi2
        if self.chi_withdelay is None:
            chi = self.chi0 - self.chit_withdelay
            self.chi_withdelay = np.roll(chi, self.nbins//2, axis=-1)

        if self.amps_withdelay is None:
            self.amps_withdelay = np.roll(
                self.signalfilt_td, self.nbins//2, axis=-1,
            )

        constraint_mask = _get_pulse_direction_constraint_mask(
            self.amps_withdelay,
            pulse_direction_constraint=pulse_direction_constraint,
        )

        bestind = _argmin_chi2(
            self.chi_withdelay,
            nconstrain=nconstrain,
            lgcoutsidewindow=lgcoutsidewindow,
            constraint_mask=constraint_mask,
            windowcenter=windowcenter,
        )

        if np.isnan(bestind):
            amp = 0.0
            t0 = 0.0
            chi2 = self.chi0
        elif interpolate_t0:
            amp, dt_interp, chi2 = self._interpolate_of(
                self.amps_withdelay, self.chi_withdelay, bestind, 1 / self.fs,
            )
            t0 = (bestind - self.nbins//2) / self.fs + dt_interp
        else:
            amp = self.amps_withdelay[bestind]
            t0 = (bestind - self.nbins//2) / self.fs
            chi2 = self.chi_withdelay[bestind]

        return amp, t0, chi2

    def ofamp_pileup_iterative(self, a1, t1, nconstrain=None,
                               lgcoutsidewindow=True,
                               pulse_direction_constraint=0, windowcenter=0,
                               interpolate_t0=False):
        """
        Function for calculating the optimal amplitude of a pileup
        pulse in data given the location of the triggered pulse.

        Parameters
        ----------
        a1 : float
            The OF amplitude (in Amps) to use for the "main" pulse,
            e.g. the triggered pulse.
        t1 : float
            The corresponding time offset (in seconds) to use for the
            "main" pulse, e.g. the triggered pulse.
        nconstrain : int, NoneType, optional
            This is the length of the window (in bins) out of which to
            constrain the possible t2 values to for the pileup pulse,
            centered on the unshifted trigger. If left as None, then t2
            is uncontrained. The value of nconstrain2 should be less
            than nbins.
        lgcoutsidewindow : bool, optional
            Boolean flag that is used to specify whether
            `OptimalFilter` should look for the pileup pulse inside the
            bins specified by `nconstrain` or outside them. If True,
            the filter will minimize the chi^2 in the bins outside the
            range specified by `nconstrain`, which is the default
            behavior. If False, then it will minimize the chi^2 in the
            bins inside the constrained window specified by
            `nconstrain`.
        pulse_direction_constraint : int, optional
            Sets a constraint on the direction of the fitted pulse.
            If 0, then no constraint on the pulse direction is set.
            If 1, then a positive pulse constraint is set for all fits.
            If -1, then a negative pulse constraint is set for all
            fits. If any other value, then a ValueError will be raised.
        windowcenter : int, optional
            The bin, relative to the center bin of the trace, on which
            the delay window specified by `nconstrain` is centered.
            Default of 0 centers the delay window in the center of the
            trace. Equivalent to centering the `nconstrain` window on
            `self.nbins//2 + windowcenter`.
        interpolate_t0 : bool, optional
            If True, then a precomputed solution to the parabolic
            equation is used to find the interpolated time-of-best-fit.
            Default is False.

        Returns
        -------
        a2 : float
            The optimal amplitude calculated for the pileup pulse (in
            Amps).
        t2 : float
            The time shift calculated for the pileup pulse (in s)
        chi2 : float
            The chi^2 value calculated for the pileup optimal filter.

        """

        self._check_freqs()

        if self.signalfilt_td is None:
            self.signalfilt_td = np.real(
                ifft(self.signalfilt * self.nbins, axis=-1)
            ) * self.df

        templatefilt_td = np.real(
            ifft(
                np.exp(
                    -2.0j * np.pi * self.freqs * t1
                ) * self.phi * self.s * self.nbins
            )
        ) * self.df

        if self.times is None:
            self.times = np.arange(
                -(self.nbins//2), self.nbins//2 + self.nbins%2,
            ) / self.fs

        # signal part of chi^2
        if self.chi0 is None:
            self.chi0 = np.real(
                np.dot(self.v.conjugate() / self.psd, self.v)
            ) * self.df

        a2s = self.signalfilt_td - a1 * templatefilt_td / self.norm

        if t1<0:
            t1ind = int(t1 * self.fs + self.nbins)
        else:
            t1ind = int(t1 * self.fs)

        # do a1 part of chi2
        chit = (
            a1**2 * self.norm
        ) - (
            2 * a1 * self.signalfilt_td[t1ind] * self.norm
        )

        # do a1, a2 combined part of chi2
        chil = (
            a2s**2 * self.norm
        ) + (
            2 * a1 * a2s * templatefilt_td
        ) - (
            2 * a2s * self.signalfilt_td * self.norm
        )

        # add all parts of chi2
        chi = self.chi0 + chit + chil

        a2s = np.roll(a2s, self.nbins//2)
        chi = np.roll(chi, self.nbins//2)

        # apply pulse direction constraint
        constraint_mask = _get_pulse_direction_constraint_mask(
            a2s,
            pulse_direction_constraint=pulse_direction_constraint,
        )

        # find time of best fit
        bestind = _argmin_chi2(
            chi,
            nconstrain=nconstrain,
            lgcoutsidewindow=lgcoutsidewindow,
            constraint_mask=constraint_mask,
            windowcenter=windowcenter,
        )

        if np.isnan(bestind):
            a2 = 0.0
            t2 = 0.0
            chi2 = self.chi0 + chit
        elif interpolate_t0:
            a2, dt_interp, chi2 = self._interpolate_of(
                a2s, chi, bestind, 1 / self.fs,
            )
            t2 = self.times[bestind] + dt_interp
        else:
            a2 = a2s[bestind]
            t2 = self.times[bestind]
            chi2 = chi[bestind]

        return a2, t2, chi2


class OFnonlin(object):
    """
    This class provides the user with a non-linear optimal filter to
    estimate the amplitude, rise time (optional), fall time, and time
    offset of a pulse.

    Attributes:
    -----------
    psd : ndarray
        The power spectral density corresponding to the pulses that
        will be used in the fit. Must be the full psd (positive and
        negative frequencies), and should be properly normalized to
        whatever units the pulses will be in.
    fs : int or float
        The sample rate of the ADC
    df : float
        The delta frequency
    freqs : ndarray
        Array of frequencies corresponding to the psd
    time : ndarray
        Array of time bins corresponding to the pulse
    template : ndarray
        The time series pulse template to use as a guess for initial
        parameters
    data : ndarray
        FFT of the pulse that will be used in the fit
    lgcdouble : bool
        If False, only the Pulse hight, fall time, and time offset will
        be fit. If True, the rise time of the pulse will be fit in
        addition to the above.
    taurise : float
        The user defined risetime of the pulse
    error : ndarray
        The uncertianty per frequency (the square root of the psd,
        divided by the errorscale)
    dof : int
        The number of degrees of freedom in the fit
    norm : float
        Normalization factor to go from continuous to FFT
    scale_amplitude : bool
        If using the 1- or 2-pole fit, whether the parameter, A, should
        be treated as the pulse height (`scale_amplitude` = True,
        default) or as a scale parameter in the functional expression.
        See `twopole` and `twopoletime` for details.

    """

    def __init__(self, psd, fs, template=None):
        """
        Initialization of OFnonlin object

        Parameters
        ----------
        psd : ndarray
            The power spectral density corresponding to the pulses that
            will be used in the fit. Must be the full psd (positive and
            negative frequencies), and should be properly normalized to
            whatever units the pulses will be in.
        fs : int, float
            The sample rate of the ADC
        template : ndarray, NoneType, optional
            The time series pulse template to use as a guess for
            initial parameters, if inputted.

        """

        self.psd = np.zeros(len(psd))
        self.psd[:] = psd
        self.psd[0] = 1e40

        self.fs = fs
        self.df = fs / len(psd)
        self.freqs = np.fft.fftfreq(len(psd), 1 / fs)
        self.time = np.arange(len(psd)) / fs
        self.template = template

        self.data = None
        self.npolefit = 1
        self.scale_amplitude = True

        self.taurise = None
        self.error = None
        self.dof = None
        self.norm = np.sqrt(fs * len(psd))


    def fourpole(self, A, B, C, tau_r, tau_f1, tau_f2, tau_f3, t0):
        """
        Functional form of pulse in frequency domain with 1 rise time
        and three fall times. The fall times have independent
        amplitudes (A,B,C). The condition f(0)=0 requires the rise time
        to have amplitude (A+B+C). Therefore, the "amplitudes" take on
        different meanings than in other n-pole functions. The
        functional form (time-domain) is:

            A*(exp(-t/tau_fall1)) + B*(exp(-t/tau_fall2)) +
            C*(exp(-t/tau_fall3)) - (A+B+C)*(exp(-t/tau_rise))

        4 rise/fall times, 3 amplitudes, and time offset allowed to
        float.

        Parameters
        ----------
        A : float
            Amplitude for first fall time
        B : float
            Amplitude for second fall time
        C : float
            Amplitude for third fall time
        tau_r : float
            Rise time of pulse
        tau_f1 : float
            First fall time of pulse
        tau_f2 : float
            Second fall time of pulse
        tau_f3 : float
            Third fall time of pulse
        t0 : float
            Time offset of four pole pulse

        Returns
        -------
        pulse : ndarray, complex
            Array of amplitude values as a function of frequency

        """

        omega = 2 * np.pi * self.freqs
        phaseTDelay = np.exp(-(0 + 1j) * omega * t0)
        pulse = (
            (
                A * (tau_f1 / (1 + omega * tau_f1 * (0 + 1j)))
            ) + (
                B * (tau_f2 / (1 + omega * tau_f2 * (0 + 1j)))
            ) + (
                C * (tau_f3 / (1 + omega * tau_f3 * (0 + 1j)))
            ) - (
                (A + B + C) * (tau_r / (1 + omega * tau_r * (0 + 1j)))
            )
        ) * phaseTDelay
        return pulse * np.sqrt(self.df)

    def fourpoletime(self, A, B, C, tau_r, tau_f1, tau_f2, tau_f3, t0):
        """
        Functional form of pulse in time domain with 1 rise time and
        three fall times The fall times have independent amplitudes
        (A,B,C). The condition f(0)=0 requires the rise time to have
        amplitude (A+B+C). Therefore, the "amplitudes" take on
        different meanings than in other n-pole functions. The
        functional form (time-domain) is:

            A*(exp(-t/tau_fall1)) + B*(exp(-t/tau_fall2)) +
            C*(exp(-t/tau_fall3)) - (A+B+C)*(exp(-t/tau_rise))

        4 rise/fall times, 3 amplitudes, and time offset allowed to
        float.

        Parameters
        ----------
        A : float
            Amplitude for first fall time
        B : float
            Amplitude for second fall time
        C : float
            Amplitude for third fall time
        tau_r : float
            Rise time of pulse
        tau_f1 : float
            First fall time of pulse
        tau_f2 : float
            Second fall time of pulse
        tau_f3 : float
            Third fall time of pulse
        t0 : float
            Time offset of four pole pulse

        Returns
        -------
        pulse : ndarray
            Array of amplitude values as a function of time

        """

        pulse = (
            A * (np.exp(-self.time / tau_f1))
        ) + (
            B * (np.exp(-self.time / tau_f2))
        ) + (
            C * (np.exp(-self.time / tau_f3))
        ) - (
            (A + B + C) * (np.exp(-self.time / tau_r))
        )
        return shift(pulse, int(t0 * self.fs))

    def threepole(self, A, B, tau_r, tau_f1, tau_f2, t0):
        """
        Functional form of pulse in frequency domain with 1 rise time
        and two fall times. The  fall times have independent amplitudes
        (A,B) and the condition f(0)=0 constrains the rise time to have
        amplitude (A+B). The functional form (time domain) is:

            A*(exp(-t/\tau_fall1)) + B*(exp(-t/\tau_fall2)) -
            (A+B)*(exp(-t/\tau_rise))

        and therefore the "amplitudes" take on different meanings than
        in the other n-pole functions

        3 rise/fall times, 2 amplitudes, and time offset allowed to
        float.

        Parameters
        ----------
        A : float
            Amplitude for first fall time
        B : float
            Amplitude for second fall time
        tau_r : float
            Rise time of pulse
        tau_f1 : float
            First fall time of pulse
        tau_f2 : float
            Second fall time of pulse
        t0 : float
            Time offset of three pole pulse

        Returns
        -------
        pulse : ndarray, complex
            Array of amplitude values as a function of frequency

        """

        omega = 2 * np.pi * self.freqs
        phaseTDelay = np.exp(-(0 + 1j) * omega * t0)
        pulse = (
            (
                A * (tau_f1 / (1 + omega * tau_f1 * (0 + 1j)))
            ) + (
                B * (tau_f2 / (1 + omega * tau_f2 * (0 + 1j)))
            ) - (
                (A + B) * (tau_r / (1 + omega * tau_r * (0 + 1j)))
            )
        ) * phaseTDelay
        return pulse * np.sqrt(self.df)


    def threepoletime(self, A, B, tau_r, tau_f1, tau_f2, t0):
        """
        Functional form of pulse in time domain with 1 rise time and
        two fall times. The  fall times have independent amplitudes
        (A,B) and the condition f(0)=0 constrains the rise time to have
        amplitude (A+B). The functional form (time domain) is:

            A*(exp(-t/\tau_fall1)) + B*(exp(-t/\tau_fall2)) - 
            (A+B)*(exp(-t/\tau_rise))

        and therefore the "amplitudes" take on different meanings than
        in the other n-pole functions

        3 rise/fall times, 2 amplitudes, and time offset allowed to
        float.

        Parameters
        ----------
        A : float
            Amplitude for first fall time
        B : float
            Amplitude for second fall time
        tau_r : float
            Rise time of pulse
        tau_f1 : float
            First fall time of pulse
        tau_f2 : float
            Second fall time of pulse
        t0 : float
            Time offset of three pole pulse

        Returns
        -------
        pulse : ndarray
            Array of amplitude values as a function of time

        """

        pulse = (
            A * (np.exp(-self.time / tau_f1))
        ) + (
            B * (np.exp(-self.time / tau_f2))
        ) - (
            (A + B) * (np.exp(-self.time / tau_r))
        )
        return shift(pulse, int(t0 * self.fs))


    def twopole(self, A, tau_r, tau_f, t0):
        """
        Functional form of pulse in frequency domain with the
        amplitude, rise time, fall time, and time offset allowed to
        float. The functional form (time domain) is:

            A*(exp(-t/\tau_fall)) - A*(exp(-t/\tau_rise))

        Note that there are 2 ways to interpret the 'A' parameter input
        to this function (see below).

        This is meant to be a private function

        Parameters
        ----------
        A : float
            Amplitude paramter or pulse height. If self.scale_amplitude
            is true, A represents the pulse height, if false, A is the
            amplitude parameter in the time domain expression above.
        tau_r : float
            Rise time of two-pole pulse
        tau_f : float
            Fall time of two-pole pulse
        t0 : float
            Time offset of two-pole pulse

        Returns
        -------
        pulse : ndarray, complex
            Array of amplitude values as a function of frequency

        """

        omega = 2 * np.pi * self.freqs

        if(self.scale_amplitude):
            delta = tau_r - tau_f
            rat = tau_r / tau_f
            amp = A / (rat**(-tau_r / delta) - rat**(-tau_f / delta))
            pulse = amp * np.abs(
                tau_r-tau_f
            ) / (
                1 + omega * tau_f * 1j
            ) / (
                1 + omega * tau_r * 1j
            ) * np.exp(-omega * t0 * 1.0j)
        else:
            pulse = (
                (
                    A * (tau_f / (1 + omega * tau_f * (0 + 1j)))
                ) - (
                    A * (tau_r / (1 + omega * tau_r * (0 + 1j)))
                )
            ) * np.exp(-omega * t0 * 1.0j)

        return pulse * np.sqrt(self.df)



    def twopoletime(self, A, tau_r, tau_f, t0):
        """
        Functional form of pulse in time domain with the amplitude,
        rise time, fall time, and time offset allowed to float. The
        functional form (time domain) is:

            A*(exp(-t/\tau_fall)) - A*(exp(-t/\tau_rise))

        Note that there are 2 ways to interpret the 'A' parameter input
        to this function (see below).

        This is meant to be a private function

        Parameters
        ----------
        A : float
            Amplitude paramter or pulse height. If self.scale_amplitude
            is true, A represents the pulse height, if false, A is the
            amplitude parameter in the time domain expression above.
        tau_r : float
            Rise time of two-pole pulse
        tau_f : float
            Fall time of two-pole pulse
        t0 : float
            Time offset of two pole pulse

        Returns
        -------
        pulse : ndarray
            Array of amplitude values as a function of time

        """

        if(self.scale_amplitude):
            delta = tau_r - tau_f
            rat = tau_r / tau_f
            amp = A / (rat**(-tau_r / delta) - rat**(-tau_f / delta))
            pulse = amp * (
                np.exp(-(self.time) / tau_f) - np.exp(-(self.time) / tau_r)
            )
        else:
            pulse = (
                A * (np.exp(-self.time / tau_f))
            ) - (
                A * (np.exp(-self.time / tau_r))
            )

        return shift(pulse, int(t0 * self.fs))


    def onepole(self, A, tau_f, t0):
        """
        Functional form of pulse in time domain with the amplitude,
        fall time, and time offset allowed to float, and the rise time
        held constant

        Parameters
        ----------
        A : float
            Amplitude of pulse
        tau_f : float
            Fall time of two-pole pulse
        t0 : float
            Time offset of two pole pulse

        Returns
        -------
        pulse : ndarray, complex
            Array of amplitude values as a function of frequency

        """

        tau_r = self.taurise
        return self.twopole(A, tau_r, tau_f, t0)

    def residuals(self, params):
        """
        Function to calculate the weighted residuals to be minimized

        Parameters
        ----------
        params : tuple
            Tuple containing fit parameters

        Returns
        -------
        z1d : ndarray
            Array containing residuals per frequency bin. The complex
            data is flatted into a single array

        """

        if (self.npolefit==4):
            A, B, C, tau_r, tau_f1, tau_f2, tau_f3, t0 = params
            delta = (self.data - self.fourpole(
                A, B, C, tau_r, tau_f1, tau_f2, tau_f3, t0,
            ))
        elif (self.npolefit==3):
            A, B, tau_r, tau_f1, tau_f2, t0 = params
            delta = (self.data - self.threepole(
                A, B, tau_r, tau_f1, tau_f2, t0,
            ))
        elif (self.npolefit==2):
            A,tau_r, tau_f, t0 = params
            delta = (self.data - self.twopole(
                A, tau_r, tau_f, t0,
            ))
        else:
            A, tau_f, t0 = params
            delta = (self.data - self.onepole(A, tau_f, t0))
        z1d = np.zeros(self.data.size * 2, dtype=np.float64)
        z1d[0:z1d.size:2] = delta.real / self.error
        z1d[1:z1d.size:2] = delta.imag / self.error

        return z1d


    def calcchi2(self, model):
        """
        Function to calculate the reduced chi square

        Parameters
        ----------
        model : ndarray
            Array corresponding to pulse function (twopole or onepole)
            evaluated at the optimal values

        Returns
        -------
        chi2 : float
            The reduced chi squared statistic

        """

        return sum(
            np.abs(self.data - model)**2 / self.error**2
        ) / (
            len(self.data) - self.dof
        )

    def fit_falltimes(self, pulse, npolefit=1, errscale=1, guess=None,
                      bounds=None, taurise=None, scale_amplitude=True,
                      lgcfullrtn=False, lgcplot=False):
        """
        Function to do the fit

        Parameters
        ----------
        pulse : ndarray
            Time series traces to be fit. Should be a 1-dimensional
            array.
        npolefit: int, optional
            The number of poles to fit.
            If 1, the one pole fit is done, the user must provide the
            value of taurise
            If 2, the two pole fit is done
            If 3, the three pole fit is done (1 rise 2 fall). Second
            fall time amplitude is independent
            If 4, the four pole fit is done (1 rise 3 fall). Second and
            third fall time amplitudes are independent
        errscale : float or int, optional
            A scale factor for the psd. For example, if fitting an
            average, the errscale should be set to the number of traces
            used in the average.
        guess : tuple, optional
            Guess of initial values for fit, must be the same size as
            the model being used for fit.
        bounds : 2-tuple of array_like, optional
            Lower and upper bounds on independent variables. Each array
            must match the size of guess. Use np.inf with an
            appropriate sign to disable bounds on all or some
            variables. If None, bounds are automatically set to within
            a factor of 100 of amplitude guesses, a factor of 10 of
            rise/fall time guesses, and within 30 samples of start time
            guess.
        taurise : float, optional
            The value of the rise time of the pulse if the single pole
            function is being use for fit
        scale_amplitude : bool, optional
            If using the 1- or 2-pole fit, whether the parameter, A,
            should be treated as the pulse height 
            (`scale_amplitude`=True, default) or as a scale parameter
            in the functional expression. See `twopole` and
            `twopoletime` for details.
        lgcfullrtn : bool, optional
            If False, only the best fit parameters are returned. If
            True, the errors in the fit parameters, the covariance
            matrix, and chi squared statistic are returned as well.
        lgcplot : bool, optional
            If True, diagnostic plots are returned.

        Returns
        -------
        variables : tuple
            The best fit parameters
        errors : tuple, optional
            The corresponding fit errors for the best fit parameters.
            Returned if `lgcfullrtn` is True.
        cov : ndarray, optional
            The convariance matrix returned from the fit. Returned if
            `lgcfullrtn` is True.
        chi2 : float, optional
            The reduced chi squared statistic evaluated at the optimal
            point of the fit. Returned if `lgcfullrtn` is True.
        success : bool, optional
           The success flag from `scipy.optimize.curve_fit`. True if
           the fit converged. Returned if `lgcfullrtn` is True.

        Raises
        ------
        ValueError
            if length of guess does not match the number of parameters
            needed in fit

        """

        self.data = np.fft.fft(pulse) / self.norm
        self.error = np.sqrt(self.psd / errscale)

        self.npolefit = npolefit
        self.scale_amplitude = scale_amplitude

        if (self.npolefit==1):
            if taurise is None:
                raise ValueError(
                    'taurise must not be None if doing 1-pole fit.'
                )
            else:
                self.taurise = taurise

        if guess is not None:
            if (self.npolefit==4):
                if len(guess) != 8:
                    raise ValueError(
                        "Length of guess not compatible with 4-pole fit. "
                        "Must be of format: guess = (A,B,C,taurise,taufall1,"
                        "taufall2,taufall3,t0)"
                    )
                else:
                    (Aguess, Bguess, Cguess, tauriseguess, taufall1guess,
                     taufall2guess, taufall3guess, t0guess) = guess
            elif (self.npolefit==3):
                if len(guess) != 6:
                    raise ValueError(
                        'Length of guess not compatible with 3-pole fit. '
                        'Must be of format: guess = (A,B,taurise,taufall1,'
                        'taufall2,t0)'
                    )
                else:
                    (Aguess, Bguess, tauriseguess, taufall1guess,
                     taufall2guess, t0guess) = guess
            elif (self.npolefit==2):
                if len(guess) != 4:
                    raise ValueError(
                        'Length of guess not compatible with 2-pole fit. '
                        'Must be of format: guess = (A,taurise,taufall,t0)'
                    )
                else:
                    ampguess, tauriseguess, taufallguess, t0guess = guess
            else:
                if len(guess) != 3:
                    raise ValueError(
                        'Length of guess not compatible with 1-pole fit. '
                        'Must be of format: guess = (A,taufall,t0)'
                    )
                else:
                    ampguess, taufallguess, t0guess = guess
        else:
            # before making guesses, if self.template
            # has been defined then define maxind,
            # ampscale, and amplitudes using the template.
            # otherwise use the pulse
            if self.template is not None:
                ampscale = np.max(pulse) - np.min(pulse)
                templateforguess = self.template
            else:
                ampscale = 1
                templateforguess = pulse

            maxind = np.argmax(templateforguess)

            if (self.npolefit==4):
                # guesses need to be tuned depending
                # on the detector being analyzed.
                # good guess for t0 particularly important to provide
                Aguess = np.mean(
                    templateforguess[maxind - 7:maxind + 7]
                ) * ampscale
                Bguess = Aguess / 3
                Cguess = Aguess / 3
                tauriseguess = 20e-6
                taufall1guess = 100e-6
                taufall2guess = 300e-6
                taufall3guess = 500e-6
                t0guess = maxind / self.fs
            elif (self.npolefit==3):
                Aguess = np.mean(
                    templateforguess[maxind - 7:maxind + 7]
                ) * ampscale
                Bguess = Aguess / 3
                tauriseguess = 20e-6
                taufall1guess = 100e-6
                taufall2guess = 300e-6
                t0guess = maxind / self.fs
            else:
                ampguess = np.mean(
                    templateforguess[maxind-7:maxind+7]
                ) * ampscale
                tauval = 0.37 * ampguess
                endt_val = int(300e-6 * self.fs)
                tauind = np.argmin(
                    np.abs(
                        pulse[maxind + 1:maxind + 1 + endt_val] - tauval
                    )
                ) + maxind + 1
                taufallguess = (tauind - maxind) / self.fs
                tauriseguess = 20e-6
                t0guess = maxind / self.fs


        if (self.npolefit==4):
            self.dof = 8
            p0 = (Aguess, Bguess, Cguess, tauriseguess, taufall1guess,
                  taufall2guess, taufall3guess, t0guess)
            if bounds is None:
                boundslower = (Aguess / 100, Bguess / 100, Cguess / 100,
                               tauriseguess / 10, taufall1guess / 10,
                               taufall2guess / 10, taufall3guess / 10,
                               t0guess - 30 / self.fs)
                boundsupper = (Aguess * 100, Bguess * 100, Cguess * 100,
                               tauriseguess * 10, taufall1guess * 10,
                               taufall2guess * 10, taufall3guess * 10,
                               t0guess + 30 / self.fs)
                bounds = (boundslower, boundsupper)
        elif (self.npolefit==3):
            self.dof = 6
            p0 = (Aguess, Bguess, tauriseguess, taufall1guess, taufall2guess,
                  t0guess)
            if bounds is None:
                boundslower = (Aguess / 100, Bguess / 100, tauriseguess / 10,
                               taufall1guess / 10, taufall2guess / 10,
                               t0guess - 30 / self.fs)
                boundsupper = (Aguess * 100, Bguess * 100, tauriseguess * 10,
                               taufall1guess * 10, taufall2guess * 10,
                               t0guess + 30 / self.fs)
                bounds = (boundslower, boundsupper)
        elif (self.npolefit==2):
            self.dof = 4
            p0 = (ampguess, tauriseguess, taufallguess, t0guess)
            if bounds is None:
                boundslower = (ampguess / 100, tauriseguess / 10,
                               taufallguess / 10, t0guess - 30 / self.fs)
                boundsupper = (ampguess * 100, tauriseguess * 10,
                               taufallguess * 10, t0guess + 30 / self.fs)
                bounds = (boundslower, boundsupper)
        else:
            self.dof = 3
            p0 = (ampguess, taufallguess, t0guess)
            if bounds is None:
                boundslower = (ampguess / 100, taufallguess / 10,
                               t0guess - 30 / self.fs)
                boundsupper = (ampguess * 100, taufallguess * 10,
                               t0guess + 30 / self.fs)
                bounds = (boundslower, boundsupper)


        result = least_squares(
            self.residuals,
            x0=p0,
            bounds=bounds,
            x_scale=p0,
            jac='3-point',
            loss='linear',
            xtol=2.3e-16,
            ftol=2.3e-16,
        )
        variables = result['x']
        success = result['success']


        if (self.npolefit==4):
            chi2 = self.calcchi2(
                self.fourpole(
                    variables[0],
                    variables[1],
                    variables[2],
                    variables[3],
                    variables[4],
                    variables[5],
                    variables[6],
                    variables[7],
                )
            )
        elif (self.npolefit==3):
            chi2 = self.calcchi2(
                self.threepole(
                    variables[0],
                    variables[1],
                    variables[2],
                    variables[3],
                    variables[4],
                    variables[5],
                )
            )
        elif (self.npolefit==2):
            chi2 = self.calcchi2(
                self.twopole(
                    variables[0],
                    variables[1],
                    variables[2],
                    variables[3],
                )
            )
        else:
            chi2 = self.calcchi2(
                self.onepole(
                    variables[0],
                    variables[1],
                    variables[2],
                )
            )

        jac = result['jac']
        cov = np.linalg.pinv(np.dot(np.transpose(jac), jac))
        errors = np.sqrt(cov.diagonal())

        if lgcplot:
            self._plotnonlin(pulse, variables, errors)
        if lgcfullrtn:
            return variables, errors, cov, chi2, success
        else:
            return variables

    def _plotnonlin(self, pulse, params, errors):
        """
        Diagnostic plotting of non-linear pulse fitting

        Parameters
        ----------
        pulse : ndarray
            The raw trace to be fit
        params : tuple
            Tuple containing best fit parameters
        errors : tuple
            The corresponding statistical errors of the fit parameters

        """

        if (self.npolefit==4):
            A, B, C, tau_r,tau_f1, tau_f2, tau_f3, t0 = params
            A_err, B_err, C_err, tau_r_err, tau_f1_err, tau_f2_err, tau_f3_err, t0_err = errors
        elif (self.npolefit==3):
            A, B, tau_r, tau_f1, tau_f2,t0 = params
            A_err, B_err, tau_r_err, tau_f1_err, tau_f2_err, t0_err = errors
        elif (self.npolefit==2):
            A,tau_r, tau_f, t0 = params
            A_err, tau_r_err, tau_f_err, t0_err = errors
        else:
            A, tau_f, t0 = params
            A_err, tau_f_err, t0_err = errors
            tau_r = self.taurise
            tau_r_err = 0.0

        if (self.npolefit==4):
            variables = [A, B, C, tau_r, tau_f1, tau_f2, tau_f3, t0]
        elif (self.npolefit==3):
            variables = [A, B, tau_r, tau_f1, tau_f2, t0]
        else:
            variables = [A, tau_r, tau_f, t0]
        ## get indices to define window ##
        t0ind = int(t0 * self.fs) #location of timeoffset

        nmin = t0ind - int(5 * tau_r * self.fs) # 5 risetimes before offset
        if (self.npolefit==3 or self.npolefit==4):
            nmax = t0ind + int(9 * tau_f1 * self.fs) # 9 falltimes after offset
        else:
            nmax = t0ind + int(7 * tau_f * self.fs) # 7 falltimes after offset

        nbaseline = int(self.fs * t0) - 1000
        if nbaseline > 0:
            pulse = pulse - np.mean(pulse[:nbaseline])
        else:
            pulse = pulse - np.mean(pulse[nbaseline + 10000:])

        f = self.freqs
        cf = f > 0
        f = f[cf]
        error = self.error[cf]

        fig, axes = plt.subplots(2, 2, figsize = (12,8))
        if (self.npolefit==4):
            fig.suptitle('Non-Linear Four Pole Fit', fontsize=18)
        elif (self.npolefit==3):
            fig.suptitle('Non-Linear Three Pole Fit', fontsize=18)
        elif (self.npolefit==2):
            fig.suptitle('Non-Linear Two Pole Fit', fontsize=18)
        elif (self.npolefit==1):
            fig.suptitle('Non-Linear Two Pole Fit (Fixed Rise Time)', fontsize=18)

        axes[0][0].grid(linestyle='dashed')
        axes[0][0].set_title(f'Frequency Domain Trace')
        axes[0][0].set_xlabel(f'Frequency [Hz]')
        axes[0][0].set_ylabel('Amplitude [A/$\sqrt{\mathrm{Hz}}$]')
        axes[0][0].loglog(
            f, np.abs(self.data[cf]), c='g', label='Pulse', alpha=0.75,
        )
        if (self.npolefit==4):
            axes[0][0].loglog(
                f, np.abs(self.fourpole(*variables))[cf], c='r', label='Fit',
            )
        elif (self.npolefit==3):
            axes[0][0].loglog(
                f, np.abs(self.threepole(*variables))[cf], c='r', label='Fit',
            )
        else:
            axes[0][0].loglog(
                f, np.abs(self.twopole(*variables))[cf], c='r', label='Fit',
            )

        axes[0][0].loglog(f, error, c='b', label='$\sqrt{PSD}$', alpha=0.75)
        axes[0][0].tick_params(which='both', direction='in', right=True, top=True)

        axes[0][1].grid(linestyle = 'dashed')
        axes[0][1].set_title(f'Time Series Trace (Zoomed)')
        axes[0][1].set_xlabel(f'Time [ms]')
        axes[0][1].set_ylabel(f'Amplitude [Amps]')

        axes[0][1].plot(
            self.time[nmin:nmax] * 1e3,
            pulse[nmin:nmax],
            c='g',
            label='Pulse',
            alpha=0.75,
        )

        if (self.npolefit==4):
            axes[0][1].plot(
                self.time[nmin:nmax] * 1e3,
                self.fourpoletime(*variables)[nmin:nmax],
                c='r',
                label='Time Domain',
            )
        elif (self.npolefit==3):
            axes[0][1].plot(
                self.time[nmin:nmax] * 1e3,
                self.threepoletime(*variables)[nmin:nmax],
                c='r',
                label='Time Domain',
            )
        else:
            axes[0][1].plot(
                self.time[nmin:nmax] * 1e3,
                self.twopoletime(*variables)[nmin:nmax],
                c='r',
                label='Time Domain',
            )
        axes[0][1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[0][1].tick_params(which='both', direction='in', right=True, top=True)


        axes[1][0].grid(linestyle='dashed')
        axes[1][0].set_title(f'Time Series Trace (Full)')
        axes[1][0].set_xlabel(f'Time [ms]')
        axes[1][0].set_ylabel(f'Amplitude [Amps]')

        axes[1][0].plot(
            self.time * 1e3, pulse, c='g', label='Pulse', alpha=0.75,
        )

        if (self.npolefit==4):
            axes[1][0].plot(
                self.time * 1e3,
                self.fourpoletime(*variables),
                c='r',
                label='Time Domain',
            )
        elif (self.npolefit==3):
            axes[1][0].plot(
                self.time * 1e3,
                self.threepoletime(*variables),
                c='r',
                label='Time Domain',
            )
        else:
            axes[1][0].plot(
                self.time * 1e3,
                self.twopoletime(*variables),
                c='r',
                label='Time Domain',
            )
        axes[1][0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[1][0].tick_params(which='both', direction='in', right=True, top=True)

        axes[1][1].plot([], [], c='r', label='Best Fit')
        axes[1][1].plot([], [], c='g', label='Raw Data')
        axes[1][1].plot([], [], c='b', label='$\sqrt{PSD}$')

        for ii in range(len(params)):
            axes[1][1].plot([], [], linestyle=' ')

        if (self.npolefit==4):
            labels = [
                f'A: ({A * 1e6:.4f} +\- {A_err * 1e6:.4f}) [$\mu$A]',
                f'B: ({B * 1e6:.4f} +\- {B_err * 1e6:.4f}) [$\mu$A]',
                f'C: ({C * 1e6:.4f} +\- {C_err * 1e6:.4f}) [$\mu$A]',
                f'τ, f1: ({tau_f1 * 1e6:.4f} +\- {tau_f1_err * 1e6:.4f}) [$\mu$s]',
                f'τ, f2: ({tau_f2 * 1e6:.4f} +\- {tau_f2_err * 1e6:.4f}) [$\mu$s]',
                f'τ, f3: ({tau_f3 * 1e6:.4f} +\- {tau_f3_err * 1e6:.4f}) [$\mu$s]',
                f'$t_0$: ({t0 * 1e3:.4f} +\- {t0_err * 1e3:.4f}) [ms]',
                f'τ$_r$: ({tau_r * 1e6:.4f} +\- {tau_r_err * 1e6:.4f}) [$\mu$s]',
            ]
        elif (self.npolefit==3):
            labels = [
                f'A: ({A * 1e6:.4f} +\- {A_err * 1e6:.4f}) [$\mu$A]',
                f'B: ({B * 1e6:.4f} +\- {B_err * 1e6:.4f}) [$\mu$A]',
                f'τ, f1: ({tau_f1 * 1e6:.4f} +\- {tau_f1_err * 1e6:.4f}) [$\mu$s]',
                f'τ, f2: ({tau_f2 * 1e6:.4f} +\- {tau_f2_err * 1e6:.4f}) [$\mu$s]',
                f'$t_0$: ({t0 * 1e3:.4f} +\- {t0_err * 1e3:.4f}) [ms]',
                f'τ$_r$: ({tau_r * 1e6:.4f} +\- {tau_r_err * 1e6:.4f}) [$\mu$s]',
            ]
        else:
            labels = [
                f'A: ({A * 1e6:.4f} +\- {A_err * 1e6:.4f}) [$\mu$A]',
                f'τ$_f$: ({tau_f * 1e6:.4f} +\- {tau_f_err * 1e6:.4f}) [$\mu$s]',
                f'$t_0$: ({t0 * 1e3:.4f} +\- {t0_err * 1e3:.4f}) [ms]',
                f'τ$_r$: ({tau_r * 1e6:.4f} +\- {tau_r_err * 1e6:.4f}) [$\mu$s]',
            ]
        lines = axes[1][1].get_lines()
        legend1 = plt.legend(
            [lines[i] for i in range(3, 3 + len(params))],
            [labels[ii] for ii  in range(len(params))],
            loc=1,
        )
        legend2 = plt.legend(
            [lines[i] for i in range(0, 3)],
            ['Best Fit', 'Raw Data', '$\sqrt{PSD}$'],
            loc=2,
        )

        axes[1][1].add_artist(legend1)
        axes[1][1].add_artist(legend2)
        axes[1][1].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
