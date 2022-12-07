import numpy as np


__all__ = [
    "OFNxMCov",
]


class OFNxMCov(object):
    """
    Class for calculating the optimal ampltiudes for N signals and
    M templates including the full noise description of the system.

    """

    def __init__(self, signals, templates, psd_matrix, channel_matrix, fs):
        """
        Initialization of OF NxM Cov

        Parameters
        ----------
        signals : ndarray, list of ndarrays
            List of all signals, should be length N
        templates : ndarray, list of ndarrays
            List of all templates, should be length M
        psd_matrix : ndarray
            Matrix of PSDs and CSDs, diagonals are PSDs, off-diagonals
            are CSDs. NxN dimensions.
        channel_matrix : ndarray
            Matrix of ones and zeros, denoting which templates will be
            applied to which channels, should be NxM.
        fs : float
            Digitization rate.

        """

        self._nbins = signals[0].shape[-1]
        self._fs = fs
        self._df = self._fs / self._nbins
        num_channels = len(signals)
        num_templates = len(templates)

        if isinstance(templates, list):
            templates = np.stack(templates, axis=0)
        if isinstance(signals, list):
            signals = np.stack(signals, axis=0)

        self._psd = psd_matrix
        self._inv_psd = np.linalg.pinv(
            np.rollaxis(psd_matrix, -1),
        )
        self._inv_psd[0] = 0
        
        s_stack = np.fft.fft(templates).T / self._nbins / self._df
        self._s = np.zeros(
            (self._nbins, num_templates, num_templates),
            dtype=np.complex128,
        )
        self._s[:, range(num_templates), range(num_templates)] = s_stack

        v_stack = np.fft.fft(signals).T / self._nbins / self._df
        self._v = np.zeros(
            (self._nbins, num_channels, 1),
            dtype=np.complex128,
        )
        self._v[:, range(num_channels), range(1)] = v_stack

        self._c = channel_matrix[None]

        self._pmatrix = self._s.conjugate() @ np.transpose(
            self._c, axes=[0, 2, 1],
        )  @ self._inv_psd @ self._c @ self._s

        self._inv_pmatrix = np.linalg.pinv(
            self._pmatrix.sum(axis=0) * self._df
        ).real

        self._q = self._s.conjugate() @ np.transpose(
            self._c, axes=[0, 2, 1],
        ) @ self._inv_psd @ self._v

    def nodelay(self):
        """
        Method to calculate the OFNxMCov ampltiudes and chi-square
        without a time-shift of the templates.

        """

        qvec = np.real(self._q.sum(axis=0) * self._df)
        
        amps = self._inv_pmatrix @ qvec

        chi20 = np.sum(np.transpose(
            self._v, axes=[0, 2, 1],
        ).conjugate() @ self._inv_psd @ self._v).real * self._df
        chi2amp = np.transpose(amps) @ (
            self._pmatrix.sum(axis=0)[None] * self._df
        ) @ amps

        chi2 = np.real(chi20 - chi2amp)

        return np.squeeze(amps), np.squeeze(chi2)


    def withdelay(self):
        """
        Method to calculate the OFNxMCov ampltiudes and chi-square
        with an allowed time-shift of the templates over all
        possible times.

        """
        
        amps = np.real(np.fft.ifft(
            self._inv_pmatrix @ self._q * self._nbins, axis=0,
        )) * self._df

        chi20 = np.sum(np.transpose(
            self._v, axes=[0, 2, 1],
        ).conjugate() @ self._inv_psd @ self._v).real * self._df

        chi2amp = np.transpose(amps, axes=[0, 2, 1]) @ (
            self._pmatrix.sum(axis=0)[None] * self._df
        ) @ amps

        chi2 = np.real(chi20 - chi2amp)

        bestind = np.argmin(chi2)
        amps_best = amps[bestind]
        chi2_best = chi2[bestind]
        t0_best = np.arange(self._nbins)[bestind] / self._fs

        if t0_best > self._nbins//2 / self._fs:
            t0_best -= self._nbins / self._fs

        return amps_best, t0_best, chi2_best

    def energy_covariance(self):
        """
        The expected energy covariance matrix given the inputted noise
        matrix and given templates.

        """

        return self._inv_pmatrix


