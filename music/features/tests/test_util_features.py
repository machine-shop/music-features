from __future__ import division
from music.features import extractor
from music.features import tonotopyExtractor
import numpy as np
import scipy as sp
import numpy.testing as npt
import librosa, math

import os
from os.path import join as pjoin

sr = 44100
n_fft = 2048

percentError = 0.1 # percentage error within MIR value

# first 2 lines are just when debugging on computer - DELETE
# test_data_path = pjoin(os.getcwd(), 'music', 'features', 'tests', 'data')
# test_data_path = pjoin(os.getcwd(), 'data')
test_data_path = pjoin(os.path.dirname(__file__), 'data')

# load beethoven test file
beet, sr = librosa.load(pjoin(test_data_path,
                     'Beethoven_Op031No2-03_002_20090916-SMD.mp3'), sr=sr)
# load test.wav file
# pure tone 100Hz (pretty sure, but based off of after the fact calculations)
test, sr = librosa.load(pjoin(test_data_path,
                      'test.wav'), sr=sr)

# load test_1000,350,350,0,1000.wav
# alternating tones of [1000, 350, 350, 0, 1000] Hz
test_alt, sr = librosa.load(pjoin(test_data_path,
                        'test_1000,350,350,0,1000.wav'), sr=sr)

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

    def test_ones(self):
        npt.assert_equal(extractor.rms(np.ones(100),
                                          decomposition=False), 1)

    def test_simple_array(self):
        val = extractor.rms(np.array([1, 2, 3, 4, 5]), decomposition=False)
        npt.assert_equal(val, 3.3166247903554)

    def test_againstMIR_beethoven(self):
        val = extractor.rms(beet, decomposition=False)
        npt.assert_almost_equal(val, 0.073394, decimal=4,
                err_msg='Returned value not within 4 decimal places of MIR output')

    # def test_againstMIR_test(self):
    #     val = extractor.rms(test, decomposition=False)
    #     npt.assert_almost_equal(val, 0.70711, decimal=4,
    #             err_msg='Returned value not within 4 decimal places of MIR output')

    def test_againstMIR_testalt(self):
        val = extractor.rms(test_alt, decomposition=False)
        npt.assert_almost_equal(val, 0.61381, decimal=4,
                err_msg='Returned value not within 4 decimal places of MIR output')

    def test_againstLIBROSA_beethoven(self):
        my_val = extractor.rms(beet, win_length=n_fft / sr, decomposition=True)
        lib_val = librosa.feature.rmse(y=beet, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    # def test_againstLIBROSA_test(self):
    #     my_val = extractor.rms(test, win_length=n_fft / sr, decomposition=True)
    #     lib_val = librosa.feature.rmse(y=test, n_fft=n_fft, hop_length=n_fft/2)
    #     corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
    #     assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_testalt(self):
        my_val = extractor.rms(test_alt, win_length=n_fft / sr, decomposition=True)
        lib_val = librosa.feature.rmse(y=test_alt, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_testToySig3Pure(self):
        my_val = extractor.rms(signal3, win_length=n_fft / sr, decomposition=True)
        lib_val = librosa.feature.rmse(y=signal3, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

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
        my_val = extractor.zcr(beet, win_length=n_fft / sr, decomposition=True)
        lib_val = librosa.feature.zero_crossing_rate(y=beet,
                frame_length=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    # def test_againstLIBROSA_test(self):
    #     my_val = extractor.zcr(test, win_length=n_fft / sr, decomposition=True)
    #     lib_val = librosa.feature.zero_crossing_rate(y=test,
    #             frame_length=n_fft, hop_length=n_fft/2)
    #     corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
    #     assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_testalt(self):
        my_val = extractor.zcr(test_alt, win_length=n_fft / sr, decomposition=True)
        lib_val = librosa.feature.zero_crossing_rate(y=test_alt,
                frame_length=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_testToySig3Pure(self):
        my_val = extractor.zcr(signal3, win_length=n_fft / sr, decomposition=True)
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
        my_val = extractor.spectralCentroid(beet, win_length=n_fft / sr, decomposition=True)
        lib_val = librosa.feature.spectral_centroid(y=beet, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    # def test_againstLIBROSA_test(self):
    #     my_val = extractor.spectralCentroid(test, win_length=n_fft / sr, decomposition=True)
    #     lib_val = librosa.feature.spectral_centroid(y=test, n_fft=n_fft, hop_length=n_fft/2)
    #     corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
    #     assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_test_alt(self):
        my_val = extractor.spectralCentroid(test_alt, win_length=n_fft / sr, decomposition=True)
        lib_val = librosa.feature.spectral_centroid(y=test_alt, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_testToySig3Pure(self):
        my_val = extractor.spectralCentroid(signal3, win_length=n_fft / sr, decomposition=True)
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
        my_val = extractor.spectralSpread(beet, win_length=n_fft / sr, decomposition=True)
        lib_val = librosa.feature.spectral_bandwidth(y=beet, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    # def test_againstLIBROSA_test(self):
    #     my_val = extractor.spectralSpread(test, win_length=n_fft / sr, decomposition=True)
    #     lib_val = librosa.feature.spectral_bandwidth(y=test, n_fft=n_fft, hop_length=n_fft/2)
    #     corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
    #     assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_test_alt(self):
        my_val = extractor.spectralSpread(test_alt, win_length=n_fft / sr, decomposition=True)
        lib_val = librosa.feature.spectral_bandwidth(y=test_alt, n_fft=n_fft, hop_length=n_fft/2)
        corr = calculateZcorr(my_val, retrieveLibrosaValue(lib_val))
        assert corr >= 0.95 # assert 95% correlation b/w zscores

    def test_againstLIBROSA_testToySig3Pure(self):
        my_val = extractor.spectralSpread(signal3, win_length=n_fft / sr, decomposition=True)
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

####### DEPRECATED CODE #######

class TestKeyStrength:

    a_major = np.matrix([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

    krumhanslProf_major = np.matrix([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    Tmajor = a_major * np.transpose(krumhanslProf_major)
    Tmajor_normed = np.array(np.transpose(Tmajor[:, 0]))[0] / max(
        np.array(np.transpose(Tmajor[:, 0]))[0])

    def test_gomezprof_major(self):
        # FIXME: undefined name Tmajor_normed
        # npt.assert_equal(Tmajor_normed, gomezProf_major)
        pass

    def check_gomezprof_minor(self):
        # FIXME: undefined name Tmajor_normed
        # npt.assert_equal(Tminor_normed, gomezProf_minor)
        pass
