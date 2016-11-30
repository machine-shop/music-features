from __future__ import division
from music_feats.features import extractor
from music_feats.features import tonotopyExtractor
import numpy as np
import scipy as sp
import numpy.testing as npt
import librosa, math

import os
from os.path import join as pjoin

sr = 44100
n_fft = 2048

percentError = 0.1 # percentage error within MIR value

test_data_path = pjoin(os.path.dirname(__file__), 'data')

# Sine signal: 10 sine waves, over 10**6 samples (f = 10**-5)
sinsig = np.sin(np.linspace(0, 20*np.pi, 10**6))

# Ones signal: constant signal of ones, over 10**6 samples
onesig = np.ones(10**6)

# Sawtooth signal: 10 sawtooth waves, over 10**6 samples (f = 10**-5)
sawsig = np.tile(np.linspace(0, 1, 10**5), 5)

# Alternating signal: flips between 1 and -1, over 10**6 samples
altsig = np.empty(10**6)
altsig[::2] = 1
altsig[1::2] = -1

# Toy signal: Sampled at 1000 Hz that is composed of a f=6 Hz, f=10 Hz, f=13
# Hz components.
t = np.linspace(0, 1, 1000)
signal = np.sin(2*np.pi*t*6) + np.sin(2*np.pi*t*10) + np.sin(2*np.pi*t*13)

# Toy signal2: Sampled at 50 Hz composed of 1Hz and 10 Hz with added, 3 second
fs = 50.                        # Sampling rate (Hz)
f = 1.                          # Base signal frequency (Hz)
t = np.arange(0.0, 3.0, 1/fs)   # Alternative: np.linspace(0., 3.0, len(y))
signal2 = np.sin(2*np.pi*f*t) + np.sin(2*np.pi*10*f*t)

# Toy signal3: Sampled at 44100 Hz composed of 1Hz, 10 seconds
fs = 44100.                        # Sampling rate (Hz)
f = 1.                          # Base signal frequency (Hz)
t = np.arange(0.0, 7.0, 1/fs)
signal3 = np.sin(2*np.pi*f*t)

# NOTE: librosa v. 1.6.1 used

class TestRMS:

    def test_one(self):
        val = extractor.rms(onesig, decomposition=False)
        npt.assert_equal(val, 1)

    def test_scale(self):
        for _ in range(10):
            scalar = np.random.random_sample()
            val = extractor.rms(onesig*scalar, decomposition=False)
            npt.assert_allclose(val, scalar, rtol=1e-3)
            val = extractor.rms(sawsig*scalar, decomposition=False)
            npt.assert_allclose(val, scalar/np.sqrt(3), rtol=1e-3)

    def test_alt(self):
        val = extractor.rms(altsig, decomposition=False)
        npt.assert_equal(val, 1)

    def test_sawtooth(self):
        val = extractor.rms(sawsig, decomposition=False)
        npt.assert_allclose(val, 1/np.sqrt(3), rtol=1e-3)

    def test_sine(self):
        val = extractor.rms(sinsig, decomposition=False)
        npt.assert_allclose(val, 1/np.sqrt(2), rtol=1e-3)

    def test_scrambled(self):
        #permutations should not change rms
        val = extractor.rms(np.random.permutation(sinsig), decomposition=False)
        npt.assert_allclose(val, 1/np.sqrt(2), rtol=1e-3)

    def test_sine_windows(self):
        val = extractor.rms(sinsig, sr=1, win_length=10**5, hop_length=10**5/5,
            decomposition=True)
        #each window contains one sine wave
        npt.assert_allclose(val, 1/np.sqrt(2)*np.ones(46), 1e-4)

class TestZCR:

    def test_one(self):
        val = extractor.zcr(onesig, sr=1, decomposition=False)
        npt.assert_equal(val, 0)

    def test_alt(self):
        val = extractor.zcr(altsig, sr=1, p='sample', d='both', decomposition=False)
        npt.assert_allclose(val, 1, rtol=1e-3)

    def test_sine(self):
        val = extractor.zcr(sinsig, sr=1, p='sample', d='both', decomposition=False)
        npt.assert_equal(val, 19/10**6)

    def test_sine_windows(self):
        val = extractor.zcr(sinsig, sr=1, win_length=10**5+1, hop_length=10**5/5,
            p='sample', d='both', decomposition=True)
        npt.assert_array_equal(val, 2/(10**5+1)*np.ones(45))


class TestSpectralCentroid:

    def test_one(self):
        val = extractor.spectralCentroid(onesig, sr=1, decomposition=False)
        npt.assert_allclose(val, 0, atol=1e-10)

    def test_sine(self):
        val = extractor.spectralCentroid(sinsig, sr=1, decomposition=False)
        npt.assert_allclose(val, 10**-5, rtol=1e-3)

    def test_sine_againstLIBROSA(self):
        my_val = extractor.spectralCentroid(sinsig, win_length=n_fft/sr, sr=sr, decomposition=True)
        lib_val = librosa.feature.spectral_centroid(y=signal3, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    # def test_againstLIBROSA_testToySig3Pure(self):
    #     my_val = extractor.spectralCentroid(signal3, win_length=n_fft/sr, sr=sr, decomposition=True)
    #     lib_val = librosa.feature.spectral_centroid(y=signal3, n_fft=n_fft, hop_length=n_fft/2)
    #     corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
    #     assert corr >= 0.95 # assert 95% correlation b/w zscores

class TestSpectralSpread:

    def test_againstMIR_beethoven(self):
        val = extractor.spectralSpread(beet, sr, decomposition=False)
        MIRVAL = 1359.8841
        assert np.abs(val-MIRVAL) <= 0.15 * MIRVAL

    # def test_againstMIR_test(self):
    #     val = extractor.spectralSpread(test, sr, decomposition=False)
    #     MIRVAL = 282.3409
    #     assert np.abs(val-MIRVAL) <= 0.15 * MIRVAL

    def test_againstMIR_test_alt(self):
        val = extractor.spectralSpread(test_alt, sr, decomposition=False)
        MIRVAL = 376.773
        assert np.abs(val-MIRVAL) <= 0.15 * MIRVAL

    def test_againstLIBROSA_beethoven(self):
        my_val = extractor.spectralSpread(beet, n_fft=n_fft, sr=sr, decomposition=True)
        lib_val = librosa.feature.spectral_bandwidth(y=beet, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    # def test_againstLIBROSA_test(self):
    #     my_val = extractor.spectralSpread(test, n_fft=n_fft, sr=sr, decomposition=True)
    #     lib_val = librosa.feature.spectral_bandwidth(y=test, n_fft=n_fft, hop_length=n_fft/2)
    #     corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
    #     assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_test_alt(self):
        my_val = extractor.spectralSpread(test_alt, n_fft=n_fft, sr=sr, decomposition=True)
        lib_val = librosa.feature.spectral_bandwidth(y=test_alt, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_testToySig3Pure(self):
        my_val = extractor.spectralSpread(signal3, n_fft=n_fft/sr, sr=sr, decomposition=True)
        lib_val = librosa.feature.spectral_bandwidth(y=signal3, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

class TestSpectralFlatness:

    def test_againstMIR_beethoven(self):
        val = extractor.spectralFlatness(beet, sr, decomposition=False)
        MIRVAL = 0.024353
        assert np.abs(val-MIRVAL) <= 0.15 * MIRVAL
        # npt.assert_allclose(val, 0.024353, significant=1,
        #     err_msg='Not equal up to one significant figure.')

    # def test_againstMIR_test(self):
    #     val = extractor.spectralFlatness(test, sr, decomposition=False)
    #     MIRVAL = 0.00095308
    #     assert np.abs(val-MIRVAL) <= 0.15 * MIRVAL

    def test_againstMIR_test_alt(self):
        val = extractor.spectralFlatness(test_alt, sr, decomposition=False)
        MIRVAL = 4.0096e-05
        assert np.abs(val-MIRVAL) <= 0.15 * MIRVAL

class TestTonotopyLabelExtractor:

	#TODO: complete this test
    def test_tonotopyExtractor(self):
        pass
        #val = tonotopyExtractor.

####### UTIL FUNCTIONS #######
def calculateZcorr(x, y):
    """Returns the correlation coefficient between two arrays."""
    # convert all nans to a number before calculating the zscore
    xz = sp.stats.mstats.zscore(np.nan_to_num(x))
    yz = sp.stats.mstats.zscore(np.nan_to_num(y))
    coeffvals = np.corrcoef(xz, yz)
    return coeffvals[0,1]

def retrieveLibrosaValue(libval):
    """Returns a 1D array from librosa outputs."""
    # currently it returns values as an array with the first element as value
    return libval[0]
