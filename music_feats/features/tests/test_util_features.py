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

    def test_constant(self):
        npt.assert_equal(extractor.rms(np.ones(10**6),
                                          decomposition=False), 1)

    def test_sawtooth(self):
        val = extractor.rms(np.linspace(0, 1, 10**6), decomposition=False)
        npt.assert_approx_equal(val, 1/np.sqrt(3), significant=4)

    def test_sine(self):
        val = extractor.rms(np.sin(np.linspace(0, 2*np.pi, 10**6)), decomposition=False)
        npt.assert_approx_equal(val, 1/np.sqrt(2), significant=4)

    def test_sines(self):
        val = extractor.rms(np.sin(np.linspace(0, 10*np.pi, 10**6)), win_length=10**5/sr, hop_length=10**5/sr/5, decomposition=True)
        npt.assert_allclose(val, 1/np.sqrt(2)*np.ones(46), 1e-4)

class TestZCR:

    def test_ones(self):
        a = np.ones(10)
        npt.assert_equal(extractor.zcr(a, decomposition=False), 0.0)

    def test_ones_and_minusones_sample_both(self):
        a = np.ones(10)
        a[1::2] = -1
        npt.assert_equal(extractor.zcr(a, p='sample', d='both',
                                          decomposition=False), 0.9)

    def test_ones_and_minusones_second_both(self):
        a = np.ones(10)
        a[1::2] = -1
        npt.assert_equal(extractor.zcr(a, sr=2, d='both',
                                          decomposition=False), 0.9 * 2)

    def test_ones_and_minusones_second_one(self):
        a = np.ones(10)
        a[1::2] = -1
        sr = 22050
        npt.assert_equal(extractor.zcr(a, sr=sr, decomposition=False),
                         0.9 * sr / 2)

    def test_ones_and_minusones_sample_one(self):
        a = np.ones(10)
        a[1::2] = -1
        npt.assert_equal(extractor.zcr(a, p='sample', decomposition=False),
                         0.9 / 2)

    def test_againstMIR_beethoven(self):
        val = extractor.zcr(beet, decomposition=False)
        MIRVAL = 632.3395
        # within percent error of the MIR value
        assert np.abs(val-MIRVAL) <= percentError * MIRVAL

    def test_againstMIR_beethoven_sample_both(self):
        val = extractor.zcr(beet, sr, p='sample', d='both',
                                          decomposition=False)
        MIRVAL = 0.028678
        assert np.abs(val-MIRVAL) <= percentError * MIRVAL

    # def test_againstMIR_test(self):
    #     val = extractor.zcr(test, decomposition=False)
    #     MIRVAL = 99.93
    #     assert np.abs(val-MIRVAL) <= percentError * MIRVAL

    # def test_againstMIR_test_sample_both(self):
    #     val = extractor.zcr(test, p='sample', d='both', decomposition=False)
    #     MIRVAL = 0.0045
    #     assert np.abs(val-MIRVAL) <= percentError * MIRVAL

    def test_againstMIR_testalt(self):
        val = extractor.zcr(test_alt, decomposition=False)
        MIRVAL = 536
        assert np.abs(val-MIRVAL) <= percentError * MIRVAL

    def test_againstMIR_testalt_sample_both(self):
        val = extractor.zcr(test_alt, p='sample', d='both', decomposition=False)
        MIRVAL = 0.0243
        assert np.abs(val-MIRVAL) <= percentError * MIRVAL

    def test_againstLIBROSA_beethoven(self):
        my_val = extractor.zcr(beet, n_fft=n_fft, sr=sr, decomposition=True)
        lib_val = librosa.feature.zero_crossing_rate(y=beet,
                frame_length=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    # def test_againstLIBROSA_test(self):
    #     my_val = extractor.zcr(test, n_fft=n_fft, sr=sr, decomposition=True)
    #     lib_val = librosa.feature.zero_crossing_rate(y=test,
    #             frame_length=n_fft, hop_length=n_fft/2)
    #     corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
    #     assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_testalt(self):
        my_val = extractor.zcr(test_alt, n_fft=n_fft, sr=sr, decomposition=True)
        lib_val = librosa.feature.zero_crossing_rate(y=test_alt,
                frame_length=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_testToySig3Pure(self):
        my_val = extractor.zcr(signal3, n_fft=n_fft, sr=sr, decomposition=True)
        lib_val = librosa.feature.zero_crossing_rate(y=signal3,
                frame_length=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

class TestSpectralCentroid:

    def test_againstMIR_beethoven(self):
        val = extractor.spectralCentroid(beet, sr,
                         decomposition=False)
        MIRVAL = 1067.1637
        assert np.abs(val-MIRVAL) <= percentError * MIRVAL

    # def test_againstMIR_test(self):
    #     val = extractor.spectralCentroid(test, sr,
    #                      decomposition=False)
    #     MIRVAL = 112.3587
    #     assert np.abs(val-MIRVAL) <= percentError * MIRVAL

    def test_againstMIR_test_alt(self):
        val = extractor.spectralCentroid(test_alt, sr,
                         decomposition=False)
        MIRVAL = 666.6773
        assert np.abs(val-MIRVAL) <= percentError * MIRVAL

    def test_againstLIBROSA_beethoven(self):
        my_val = extractor.spectralCentroid(beet, n_fft=n_fft, sr=sr, decomposition=True)
        lib_val = librosa.feature.spectral_centroid(y=beet, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    # def test_againstLIBROSA_test(self):
    #     my_val = extractor.spectralCentroid(test, n_fft=n_fft, sr=sr, decomposition=True)
    #     lib_val = librosa.feature.spectral_centroid(y=test, n_fft=n_fft, hop_length=n_fft/2)
    #     corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
    #     assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_test_alt(self):
        my_val = extractor.spectralCentroid(test_alt, n_fft=n_fft, sr=sr, decomposition=True)
        lib_val = librosa.feature.spectral_centroid(y=test_alt, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_testToySig3Pure(self):
        my_val = extractor.spectralCentroid(signal3, n_fft=n_fft, sr=sr, decomposition=True)
        lib_val = librosa.feature.spectral_centroid(y=signal3, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

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
        my_val = extractor.spectralSpread(signal3, n_fft=n_fft, sr=sr, decomposition=True)
        lib_val = librosa.feature.spectral_bandwidth(y=signal3, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores


class TestSpectralFlatness:

    def test_againstMIR_beethoven(self):
        val = extractor.spectralFlatness(beet, sr, decomposition=False)
        MIRVAL = 0.024353
        assert np.abs(val-MIRVAL) <= 0.15 * MIRVAL
        # npt.assert_approx_equal(val, 0.024353, significant=1,
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
    """Function basically exists for abstraction purposes. To
    retrieve the actual array value from what librosa spits out."""
    # currently it returns values as an array with the first element as value
    return libval[0]
