from __future__ import division
import math
import librosa
import numpy as np
import scipy as sp
from music_feats.features.util.utils import *

__all__ = ['rms',
           'zcr',
           'spectralCentroid',
           'spectralSpread',
           'CQT',
		   'temporalEnvelope',
           'temporalFlatness',
           'chromagram',
           'tonality',
           'fluctuationPatterns',
           'fluctuationEntropy',
           'fluctuationFocus',
           'fluctuationCentroid',
		   'MPS']

def rms(y, sr=44100, win_length=0.05, hop_length=None,
    pad=None, decomposition=True):
    '''
    Calculate root-mean-square energy from a time-series signal
        :usage:
                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> # Calculate the RMS of a time-series
                >>> rms = extractor.rms(y, sr=sr,
                        win_length=None, hop_length=512, decomposition='True')

        :parameters:
            - y : np.ndarray [shape=(n,)]. Time series to calculate the RMS of.
            - sr : integer. sampling rate of the audio file
            - win_length : integer. The frame length of the music time series
                           (in s) to be considered.  Default 50 ms.
            - hop_length : integer. The amount of overlap between the frames
                           (in s).  Default is half the window length.
            - pad : integer. Amount which to pad by before frame decomposition.
            - decomposition : boolean. Whether or not to do a framewise
                              analysis of the time series

        :returns:
            If decomposition = 'False':
            - A float representing the root-mean-square of the time-series of
              the signal
            If decomposition = 'True':
            - A numpy array representing the root-mean-square of the
              time-series of the signal per frame.
    '''
    if decomposition:
        if hop_length is None:
            hop_length = win_length/2
        win_length, hop_length = int(win_length*sr), int(hop_length*sr)
        return framewise(rms, y, win_length, hop_length,
            padAmt=pad, decomposition=False)
    else:
        return np.sqrt(np.sum(y**2)/len(y))


def zcr(y, sr=44100, p='second', d='one', n_fft=2048, hop_length=None,
        pad=None, decomposition=True): # win_length=0.05
    '''
    Calculate the zero-crossing rate from a time-series signal.

        :usage:
                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> # Calculate a zero-crossing rate of the time-series of a
                >>> # signal
                >>> zcr = extractor.zcr(y, sr=sr, p='second', d='one',
                                        win_length=0.05, hop_length=None,
                                        decomposition='True')

        :parameters:
            - y : np.ndarray [shape=(n,)] Time series of the audio file.
            - sr : integer. Sampling rate of the audio file.
            - p : Number of zero crossings either per 'second' or per 'sample'.
                  Default: 'second'.
            - d : Number of zero crossings from negative to positive (or
                  equivalently positive to negative) only using d='one'
                  (default) or both directions using d='both'.
            - win_length : integer. The frame length of the music time series
                   (in s) to be considered.  Default 50 ms.
            - hop_length : integer. The amount of overlap between the frames
                    (in samples).  Default is half the window length.
            - decomposition : boolean. Whether or not to do a framewise
                    analysis of the time series

        :returns:
            If decomposition = 'False':
            - A float representing the zcr of the time-series of the signal.
            If decomposition = 'True':
            - A numpy array representing the zcr of the time-series of the
              signal per frame.
    '''
    if decomposition:
        # win_length = sr * win_length
        if hop_length is None:
            # hop_length = int(win_length / 2)
            hop_length = int(n_fft / 2)
        return framewise(zcr, y, n_fft, hop_length, padAmt=pad,
                         sr=sr, p=p, d=d, decomposition=False) # win_length
    else:
        zcrate = y[1:] * y[:len(y)-1]
        # All zero crossings can be identified with a negative number
        # zcrate = sum(zcrate < 0) / len(y)
        zcrate = len(np.where(zcrate < 0)[0]) / len(y) # np.where() speed boost
        if p == 'second':
            zcrate = zcrate * sr
        if d == 'one':
            zcrate = zcrate / 2
        return zcrate


def spectralCentroid(y, sr=44100, n_fft=2048, hop_length=None,
                     toWin=True, pad=None, decomposition=True): #win_length=0.05
    '''
    Calculate the spectral centroid (mean) of a time-series signal. Commonly
    used as the brightness of a sound.

        :usage:
                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> # Calculate the spectral centroid of a time-series
                >>> spectralCentroid = extractor.spectralCentroid(y,
                    sr=sr, win_length=0.05, hop_length=None,
                    decomposition=True)

        :parameters:
            - y : A numpy array [shape=(n,)] of time series to calculate the
                  spectral centroid of.
            - sr : Sampling rate of the audio file. (Default = 22050)
            - win_length : integer. The frame length of the music time series
              (in s) to be considered.  Default 50 ms.
            - hop_length : integer. The amount of overlap between the frames
              (in samples).  Default is half the window length.
            - decomposition: boolean. Whether or not to do a framewise
              analysis of the time-series.

        :returns:
            If decomposition=False:
            - A float representing the spectral centroid (mean) of the signal.
            If decomposition=True:
            - A numpy array representing the spectral centroid of the signal
              per frame (window).

        :notes:
            - Beauchamp J. W., Synthesis by Spectral Amplitude and
            'Brightness' Matching of Analyzed Musical Instrument Tones
    '''
    if decomposition:
        # win_length = sr * win_length
        if hop_length is None:
            hop_length = int(n_fft/2)
            # hop_length = int(win_length/2)
        return framewise(spectralCentroid, y, n_fft, hop_length,
                         toWin=toWin, padAmt=pad, sr=sr,
                         decomposition=False) #win_length
    else:
        Y = np.fft.fft(y)
        magns = np.abs(Y[:np.ceil(len(Y)/2)])
        # Calculate the frequency bin values
        freqs = np.fft.fftfreq(len(Y), 1/sr)[:np.ceil(len(Y)/2)]
        # freqs = np.linspace(0, 1, len(Y))[:len(Y)/2] * sr
        return np.dot(freqs, magns) / np.sum(magns)


def spectralSpread(y, sr=44100, n_fft=2048, hop_length=None,
                   toWin=True, pad=None, decomposition=True):
    '''
    Calculate the spectral spread of a time-series signal.
        :usage:
                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> # Calculate the spectral spread of a time-series
                >>> spectralSpread = extractor.spectralSpread(y, sr=sr,
                    win_length=0.05, hop_length=None, decomposition=True)

        :parameters:
            - y : A numpy array [shape=(n,)] of time series to calculate the
              spectral spread of.
            - sr : Sampling rate of the audio file. (Default = 22050)
            - win_length : integer. The frame length of the music time series
              (in s) to be considered.  Default 50 ms.
            - hop_length : integer. The amount of overlap between the frames
              (in s).  Default is half the window length.
            - decomposition: boolean. Whether or not to do a framewise analysis
              of the time series.

        :returns:
            If decomposition=False:
            - A float representing the spectral spread (standard deviation) of
              the signal.
            If decomposition=True:
            - A numpy array representing the spectral spread of the signal per
              frame
    '''
    if decomposition:
        if hop_length is None:
            hop_length = int(n_fft/2)
        return framewise(spectralSpread, y, n_fft, hop_length,
                         toWin=toWin, padAmt=pad, sr=sr,
                         decomposition=False)
    else:
        # Calculate the spectrum
        Y = np.fft.fft(y)
        magns = np.abs(Y[:np.ceil(len(Y))/2])
        # Calculate the frequency bin values
        freqs = np.fft.fftfreq(len(Y), 1/sr)
        # freqs = np.linspace(0, 1, len(Y)) * sr
        # Calculate SC
        SC = np.dot(freqs[:np.ceil(len(Y)/2)], magns) / np.sum(magns)
        scs = np.ones(len(Y))*SC
        # Calculate the squared deviation from the spectral centroid
        spread = (freqs - scs)**2
        return np.sqrt(np.dot(spread[:len(Y)/2], magns) / np.sum(magns))
        # bins_var = np.linspace(0,1,len(Y)) * sr - np.ones(len(Y)) * SC
        # bins_var = bins_var ** 2
        # temp = np.dot(bins_var[:len(Y)/2], abs(Y[:len(Y)/2]))
        # return (temp / np.sum(abs(Y[:len(Y)/2]))) ** 0.5

def spectralFlatness(y, sr=44100, n_fft=2048, hop_length=None,
                     toWin=True, pad=None, decomposition=True): # win_length=0.05
    '''
    Calculate the spectral flatness from a time-series signal
        :usage:
                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> # Calculate the spectral flatness of a time-series
                >>> spectralFlatness = extractor.spectralFlatness(y,
                                    sr=sr, win_length=0.05,
                                    hop_length=None, decomposition=True)

        :parameters:
            - y : np.ndarray [shape=(n,)]. Time series to calculate the
              spectral flatness of.
            - sr : integer. sampling rate of the audio file
            - win_length : integer. The frame length of the music time series
              (in s) to be considered.  Default 50 ms.
            - hop_length : integer. The amount of overlap between the frames
              (in s).  Default is half the window length.
            - decomposition : boolean. Whether or not to do a framewise
              analysis of the time series.

        :returns:
            If decomposition = False:
            - A float representing the spectral flatness of the signal
            If decomposition = True:
            - A numpy array representing the spectral flatness of the signal
              per frame
    '''
    if decomposition:
        # win_length = sr * win_length
        if hop_length is None:
            hop_length = int(n_fft/2)
            # hop_length = int(win_length/2)
        return framewise(spectralFlatness, y, n_fft, hop_length,
                         toWin=toWin, padAmt=pad, sr=sr,
                         decomposition=False) # win_length
    else:
        Y = np.fft.fft(y)
        Y_abs = abs(Y[:len(Y)/2])
        return sp.stats.mstats.gmean(Y_abs) / np.mean(Y_abs)

def CQT(y, sr=44100, cqt_hop=1024, seconds=2.0, n_bins=30, bins_per_octave=4, fmin=27.5,
        use_han=False):
    """
    Get the constant-q transform of the audio file. Takes ((seconds*sr)//cqt_hop) * cqt_hop
    sample long chunks of the audiofile before doing the cqt computation. Hop length between
    these chunks is frame_length - cqt_hop, where frame_length is the size of the chunks of the
    audiofile. These chunks are necessary because librosa's cqt function can only handle short
    duration audio files in a reasonable amount of time.
    	:usage:
    		>>> # Load a file
    		>>> y, sr = librosa.load('file.mp3')
    		>>> # Calculate the constant q transform of a time-series
    		>>> CQTlog = extractor.CQT(y, sr=sr, ...)

    	:parameters:
    		- cqt_hop : integer. The hop length between adjacent frames for when extracting
    					the cqt feature.
    		- seconds : float. The time window to intially chunk the audio file into before
    					feeding into the librosa cqt function.
    		- n_bins : integer. The number of cqt frequency bands to extract.
    		- bins_per_octave : interger. The number of cqt frequency bands that comprise
    							an octave. The number of octaves is n_bins/float(bins_per_octave).
    		- fmin : integer. The lowest frequency in the range of frequencies covered by the constant
    						q transform.
    		- use_han : boolean. True, window each frame with a hanning window before extracting CQT.
    					As of 06/22/2016, librosa's util.frame() function already applies a hanning window.

    	:returns:
    		- CQTlog : np.ndarray [shape=(n_bins, n)]. The time series of the constant-q transform of the
    				   audio file.
    """
    frame_length = seconds * sr
    frame_length = (frame_length//cqt_hop) * cqt_hop
    frame_hop = frame_length - cqt_hop

    padded_y = np.append(y, np.zeros(frame_length))

    y_frames = librosa.util.frame(padded_y, frame_length=frame_length, hop_length=frame_hop)

    if use_han:
    	han_win = signal.hanning(frame_length)

    CQT_frames = []
    for frame in range(y_frames.shape[1]):
    	if not use_han:
    		sig = y_frames[:, frame]
    	else:
    		sig = y_frames[:, frame] * han_win
    	CQTf = np.abs(librosa.cqt(sig, sr=sr, n_bins=n_bins, hop_length=cqt_hop,
    							  bins_per_octave=bins_per_octave, fmin=fmin))
    	CQT_frames.append(CQTf[:,1:-1])

    CQT = np.hstack(CQT_frames)
    CQTlog = librosa.logamplitude(CQT**2, ref_power=np.max)
    return CQTlog

def temporalEnvelope(y, sr=44100):
    '''
    Calculate the temporal envelope of the signal.
        :usage:
            >>> # Load a file
            >>> y, sr = librosa.load('file.mp3')
            >>> # Calculate the temporal envelope of a time-series
            >>> tempEnv = extractor.temporalEnvelope(y, sr=sr)

        :parameters:
            - y : np.ndarray [shape=(n,)]. Time series to calculate the
                temporal envelope of.
            - sr : integer. sampling rate of the audio file.

        :returns:
            - ndarray [shape=(n,1)]. The time series of the temporal
                envelope of the audio file.
    '''
    return np.abs(y)

def temporalFlatness(y, sr=44100, n_fft=2048, hop_length=None,
                     pad=None, decomposition=True): #win_length=0.05
    '''
    Calculate the temporal flatness from a time-series signal
        :usage:
                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> # Calculate the temporal flatness of a time-series
                >>> temporalFlatness = extractor.temporalFlatness(y,
                                    sr=sr, win_length=0.05, hop_length=None,
                                    pad=None, decomposition=True)

        :parameters:
            - y : np.ndarray [shape=(n,)]. Time series to calculate the
              temporal flatness of.
            - sr : integer. sampling rate of the audio file
            - win_length : integer. The frame length of the music time series
              (in s) to be considered.  Default 50 ms.
            - hop_length : integer. The amount of overlap between the frames
              (in s).  Default is half the window length.
            - pad : integer. How much to pad audio by (beginning and end)
              before doing computation
            - decomposition : boolean. Whether or not to do a framewise
              analysis of the time series.

        :returns:
            If decomposition = False:
            - A float representing the temporal flatness of the signal
            If decomposition = True:
            - A numpy array representing the temporal flatness of the signal
              per frame
    '''
    if decomposition:
        # win_length = sr * win_length
        if hop_length is None:
            # hop_length = int(win_length/2)
            hop_length = int(n_fft/2)
        return framewise(temporalFlatness, y, n_fft, hop_length,
                         padAmt=pad, sr=sr, decomposition=False) #win_length
    else:
        y_abs = np.abs(y)
        return sp.stats.mstats.gmean(y_abs) / np.mean(y_abs)

def chromagram(y=None, sr=44100, S=None,  norm=np.inf, n_fft=2048,
                hop_length=None, seconds=4, tuning=None,
                center=True, **kwargs):
    """
    Derivation of chromagram from librosa python package. Bins spectrogram
    on a larger frame size than it was originally calculated with.
        :parameters:
            - y : np.ndarray. The signal to calculate the chromagram of.
                Default is None.
            - sr : integer. The sampling rate of the audiofile. Default is
                44100 Hz.
            - S : np.ndarray. The spectrogram from which to calculate
                the chromagram. Default is None (function calculates
                    spectrogram first).
            - norm : float or None. Column-wise normalization. Default np.inf.
            - n_fft : integer. The window size with which to calculate
                the spectrogram. Default is 2048.
            - hop_length : integer. The amount of overlap between frames.
                Default is half-overlap.
            - seconds : integer. The amount of seconds to bin the spectrogram
                into before calculating the chromagram. Default is 4 seconds.
            - tuning : float in '[-0.5, 0.5]' or None. Deviation from A440
                tuning in fractional bins. Default is None (automatically
                estimated)
            - center : boolean. Whether or not to center the spectrogram
                before calculating the chromagram. Default is True.
            - kwargs : the arguments for librosa.filter.chroma()
    """

    n_chroma = 12 # defining variable for use below
    if hop_length is None:
        hop_length = int(n_fft / 2)
    if S is None:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    else:
        n_fft = 2 * (S.shape[0] - 1)

    if tuning is None:
        tuning = librosa.estimate_tuning(S=S, sr=sr, bins_per_octave=n_chroma)

    if 'A440' not in kwargs:
        kwargs['A440'] = 440.0 * 2.0**(float(tuning) / n_chroma)

    chromafb = librosa.filters.chroma(sr, n_fft, **kwargs)

    segment_length = sr * seconds / hop_length # n_fft??

    # make it a power of two
    segment_length = 2**prevPow(segment_length) #alt: nextPow()
    if center:
        npad = ((0,0), (int(segment_length/4), int(segment_length/4)))
        S = np.pad(S, npad, mode='reflect')
    num_segments = math.floor((S.shape[1] - segment_length) /
                                        (segment_length/2) + 1)

    bin_S = np.zeros((S.shape[0], num_segments))
    # Calculate binned chromagram
    for i in range(int(num_segments)):
        start, end = calculateStartEnd(segment_length / 2, segment_length, iterV=i)
        bin_S[:,i] = np.mean(S[:, start:end], axis=1)

    # Compute raw chroma
    raw_chroma = np.dot(chromafb, bin_S)

    # Compute normalization factor for each frame
    return librosa.util.normalize(raw_chroma, norm=norm, axis=0)


def tonality(y, sr=44100, profiles='gomez', n_fft=2048, hop_length=1024,
                seconds=4, center=True, use_librosa=True, full=False):
    '''
    Calculates the average correlation coefficients of different tonalities
    (key & mode)
        :usage:
                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> # Calculate key strength values and a time-series
                    analysis of the tonality/modality
                >>> audio_keys, mode, major, minor, keys =
                            extractor.tonality(y, sr=sr,
                            profiles='gomez', n_fft=2048, hop_length=64,
                            full=True)
                >>> audio_keys, mode = extractor.tonality(y, sr=sr,
                            profiles='gomez', n_fft=2048, hop_length=64,
                            full=False)

        :parameters:
            - y : np.ndarray [shape=(n,)] Time series of the audio file
            - sr : integer. sampling rate of the audio file
            - profiles : default is gomez. The key profiles to be used.
              Profiles available:
                - Gomez, 2006; Krumhansl, Cognitive Foundations of Pitch;
                Temperley The Krumhansl-Schmuckler Key-Finding Algorithm
                Revisted; Temperley, MIREX; Wei Chai MIT PhD Thesis
            - n_fft : integer. FFT window size for STFT
            - hop_length : integer. The amount of overlap between the frames
              of the STFT (in samples).  Default is half the window length.
            - seconds : integer. The bin width to bin the spectrogram with
                before calculating chroma. Number is rounded down to nearest
                power of 2. (Default is 4 seconds -> ~3 seconds)
            - center : boolean. For extractor.chromagram(). Whether or not to
                center the spectrogram before calculating the chromagram.
                Default is True.
            - librosa : boolean. Whether to use extractor.chromagram() or
                librosa.chromagram(). Default is True.
            - full : boolean. Whether or not to include cumulative information
              about the tonality (key + mode) for the full song

        :returns:
            - A numpy array representing the key of the signal per frame
            - A numpy array representing the mode of the signal per frame
            If full = True, also include:
            - A numpy array with the average correlation values for major
              tonalities
            - A numpy array with the average correlation values for minor
              tonalities
            - An array of keys
    '''

    major_prof, minor_prof = loadProfiles(name=profiles)
    keys = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    if use_librosa:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft,
                                        hop_length=hop_length)
    else:
        chroma = chromagram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                            seconds=seconds, center=center)

    audio_keys, mode = [], []
    if full:
        major, minor = {}, {}
        for key in keys:
            major[key], minor[key] = 0, 0

    # Create the profile matrix
    prof = major_prof
    for i in range(1, 12):
        prof = np.vstack((prof, np.roll(major_prof, i)))
    for i in range(12):
        prof = np.vstack((prof, np.roll(minor_prof, i)))

    prev_key, prev_mode = '', ''
    for j in range(np.shape(chroma)[1]):
        # Calculate the correlation matrix with the profile matrix
        tmp = np.corrcoef(np.vstack((np.transpose(chroma[:, j]), prof)))
        outcomeMajor = tmp[0, 1:13]
        outcomeMinor = tmp[0, 13:]
        if full:
            for ind, key in enumerate(keys):
                major[key] += outcomeMajor[ind]
                minor[key] += outcomeMinor[ind]
        maxMajorind = np.argmax(outcomeMajor)
        maxMinorind = np.argmax(outcomeMinor)
        # Based on which correlation value was greater, append key/mode
        if outcomeMajor[maxMajorind] > outcomeMinor[maxMinorind]:
            mode.append('Major')
            audio_keys.append(keys[maxMajorind])
            prev_mode, prev_key = 'Major', keys[maxMajorind]
        elif outcomeMinor[maxMinorind] > outcomeMajor[maxMajorind]:
            mode.append('Minor')
            audio_keys.append(keys[maxMinorind])
            prev_mode, prev_key = 'Minor', keys[maxMinorind]
        else:
            # if neither larger, append previous key/mode
            mode.append(prev_mode)
            audio_keys.append(prev_key)

    if full:
        major_vals, minor_vals = [], []
        for key in keys:
            major_vals.append(major[key])
            minor_vals.append(minor[key])
        major_vals = np.array(major_vals)/np.shape(chroma)[1]
        minor_vals = np.array(minor_vals)/np.shape(chroma)[1]
        # normalize
        max_val = max(max(abs(major_vals)), max(abs(minor_vals)))
        major_vals = np.array(major_vals)/max_val
        minor_vals = np.array(minor_vals)/max_val
        return audio_keys, mode, major_vals, minor_vals, keys
    else:
        return audio_keys, mode


def fluctuationPatterns(y, sr=44100, n_fft=512, hop_length=512, mel_count=36,
                        seconds=3, band_num=12, max_freq=10, center=True,
                        padAmt=0.25, Pampalk=True, terhardt=False):
    '''
    Calculates the fluctuation patterns of the piece based on E. Pampalk's
    PhD thesis (2006).
        :usage:
                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> # Calculate the fluctuation patterns
                >>> fluctuation_patterns =
                        extractor.fluctuationPatterns(y, sr=sr, n_fft=512,
                                hop_length=512, mel_count=12, seconds=3,
                                band_num=12, max_freq=10, Pampalk=True,
                                terhardt=False)

        :parameters:
            - y : np.ndarray [shape=(n,)] Time series of the audio file
            - sr : integer. sampling rate of the audio file. Default 44100.
            - n_fft : integer. FFT window size for
                librosa.feature.melspectrogram() [samples]
            - hop_length : integer. The amount of overlap [samples] between
                the frames for librosa.feature.melspectrogram(). Default is
                same values as n_fft.
            - mel_count : integer. The number of mel bands to consider when
                generating the melspectrogram. Default is 36 bands.
            - seconds : integer. The length of each segment when calculating
                the fluctuation patterns. Default is 3 seconds.
            - band_num : integer. The number of bands to consider when calculating
                the fluctuation patterns. The mel bands resulting from the initial
                melspectrogram will be combined together to get this many bands.
                Default is 12 bands.
            - max_freq : integer. The maximum modulation frequency to be considered.
                Default is 10 Hz.
            - center : boolean. Whether or not to center the signal. Default True.
            - padAmt : float. What factor of the segment_length to pad by.
                Will pad from left and from right by the same amount. Default is 0.25.
            - Pampalk: boolean. Whether to use Pampalk's algorithm straight from his
                thesis (i.e. with the hardcoded gaussian values) or to use an
                alternate method involving a built-in gaussian filter. Default is
                using Pampalk's method.
            - terhardt: boolean. Whether or not to apply the Terhardt perception model
                (1979). Default is False.

        :returns:
            - np.ndarray[shape=(segment_num, band_num*resolution)]:
                Fluctuation patterns. Each row is a segment in time
                (row 0 being first). Resolution is a value calculated
                in the function; it corresponds to the resolution of the
                modulation frequency domain (bin number between 0-max_freq Hz).
                Note the fluctuation patterns are encoded in vector format.
    '''

    # calculate log mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                            hop_length=hop_length, n_mels=mel_count)
    log_S = librosa.core.logamplitude(S, ref_power=np.max)

    # apply auditory perception weights
    # Terhardt perception model (1979)
    if terhardt == True:
        f = np.linspace(0, sr/n_fft, num=n_fft)[:n_fft/2 + 1]
        w = 3.64 * np.power(f, -0.8) - \
            6.5 * np.exp(-0.6 * np.power((f - 3.3), 2)) + \
            10**(-3)*np.power(f, 4)
        w = np.tile(w,(1,1))
        w[0,0]=0.0
        W = np.tile(np.transpose(w),(1,np.shape(log_S)[1]))
        W = np.dot(librosa.filters.mel(sr, n_fft, n_mels=mel_count), W)
        log_S = np.multiply(log_S,W)

    # calculate the segment length using the effective sampling rate
    # segment_length = sr * seconds / n_fft
    # TODO: verify accuracy of this calculation
    # I think this is the correct way to calculate the new effective sampling rate
    segment_length = sr * seconds / hop_length
    # make it a power of two
    segment_length = 2**prevPow(segment_length) #alt: nextPow()
    # pad to center signal
    if center:
        npad = ((0,0),
                (int(segment_length * padAmt), int(segment_length * padAmt)))
        log_S = np.pad(log_S, npad, mode='reflect')
    num_segments = math.floor((np.shape(log_S)[1] - segment_length) /
                                        (segment_length/2) + 1)
    # Calculate the number of frequency bins from 0 to maxfreq
    resolution = math.ceil(max_freq / (sr / n_fft) * segment_length)

    f = np.linspace(0, sr/n_fft, num=segment_length)
    tmp = 1 / (f[1:2+resolution] / 4 + 4 / f[1:2+resolution])
    flux = np.tile(tmp, (band_num, 1)) # band_num used to originally be 12: hardcode??

    # Creating filters following method in Pampalk PhD Thesis (2006)
    if Pampalk:
        vals = [0.05, 0.1, 0.25, 0.5, 1, 0.5, 0.25, 0.1, 0.05]
        filt_one = sp.signal.convolve2d(np.identity(band_num),
                                    np.tile(vals[::-1],(1,1)), mode='same')
        tmp = np.transpose(np.tile(np.sum(filt_one, axis=1),(1,1)))
        filt_one = np.divide(filt_one, np.tile(tmp, (1,band_num)))
        filt_two = sp.signal.convolve2d(np.identity(resolution),
                                    np.tile(vals[::-1],(1,1)), mode='same')
        tmp = np.transpose(np.tile(np.sum(filt_two, axis=1),(1,1)))
        filt_two = np.divide(filt_two, np.tile(tmp, (1,resolution)))

    t = np.zeros(mel_count)

    # Combine freq. bands of melspectrogram
    # Combine according to values from Pampalk Thesis (2006)
    step = np.concatenate((np.array([1,1,2,2,2,2,2,2]), np.arange(4,20)))
    cur_ind, i, curr = 0, 0, 0
    while cur_ind < mel_count and curr < band_num:
        t[cur_ind:cur_ind+step[i]] = curr
        cur_ind += step[i]
        curr += 1
        i += 1
    log_S_merged = np.zeros((band_num, np.shape(log_S)[1]))
    for i in range(band_num):
        log_S_merged[i, :] = np.sum(log_S[t==i,:], 0)

    # Initialize an array to hold the calculated fluctuation patterns
    all_fp = np.zeros((num_segments, band_num*resolution))
    # Calculate fluctuation patterns
    for i in range(int(num_segments)):
        # Calculate chunk start/end indices
        start, end = calculateStartEnd(segment_length / 2, segment_length, iterV=i)
        # start = (segment_length / 2) * i
        # end = segment_length + (segment_length / 2) * i
        Y = np.fft.fft(log_S_merged[:, int(start):int(end)],
                                    n=int(segment_length), axis=1)
        # Fluctuation perception weights, Fastl (1982)
        # Apply filters to smooth and emphasize major fluctuation contrasts
        Y2 = np.multiply(abs(Y[:,1:2+resolution]), flux)
        if Pampalk:
            Y2 = np.dot(filt_one, np.dot(abs(np.diff(Y2, n=1, axis=1)),filt_two))
        else:
            Y2 = abs(np.diff(Y2, n=1, axis=1))
            Y2 = sp.ndimage.filters.gaussian_filter(Y2, 2)
        # Store fluctuation patterns in vectorized format
        all_fp[i, :] = np.transpose(Y2.flatten())
    return all_fp

# TODO: verify if computation of fluctuation entropy is accurate
def fluctuationEntropy(y=None, sr=44100, all_fp=None, decomposition=True,
                        n_fft=512, hop_length=512, mel_count=36, seconds=3,
                        band_num=12, max_freq=10, Pampalk=True,
                        terhardt=False):
    '''
    Calculates the entropy of the fluctuation patterns of the audio piece.
    Based on computation from V. Alluri (2012) paper, with slight modification.
    Can calculate for either the median fluctuation pattern or for all
    fluctuation patterns.
        :usage:
                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> # Calculate the fluctuation entropy of an audiofile
                >>> fp_entropy = extractor.fluctuationEntropy(y, sr=sr,
                        decomposition=True, n_fft=512, hop_length=512,
                        mel_count=36, seconds=3, band_num=12, max_freq=10,
                        Pampalk=True, terhardt=False)

                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> fluctuation_patterns =
                                    extractor.fluctuationPatterns(y, sr=sr)
                >>> # Calculate the fluctuation entropy from fp values
                >>> fp_entropy =
                      extractor.fluctuationEntropy(all_fp=fluctuation_patterns)

        :parameters:
            - y : np.ndarray [shape=(n,)] Time series of the audio file.
                Default None.
            - sr : integer. sampling rate of the audio file. Default 44100.
            - all_fp: np.ndarray[shape=(segment_num, band_num*resolution)].
                Output of fluctuation_patterns. Default None.
            - decomposition: boolean. Whether to only consider the median
                fluctuation pattern or all fluctuation patterns.
            - n_fft : integer. FFT window size for
                librosa.feature.melspectrogram() [samples]. Default 512.
            - hop_length : integer. The amount of overlap [samples] between
                the frames for librosa.feature.melspectrogram(). Default is
                same values as n_fft. Default 512.
            - mel_count : integer. The number of mel bands to consider when
                generating the melspectrogram. Default is 36 bands.
            - seconds : integer. The length of each segment when calculating
                the fluctuation patterns. Default is 3 seconds.
            - band_num : integer. The number of bands to consider when calculating
                the fluctuation patterns. The mel bands resulting from the initial
                melspectrogram will be combined together to get this many bands.
                Default is 12 bands.
            - max_freq : integer. The maximum modulation frequency to be considered.
                Default is 10 Hz.
            - Pampalk: boolean. Whether to use Pampalk's algorithm straight from his
                thesis (i.e. with the hardcoded gaussian values) or to use an
                alternate method involving a built-in gaussian filter. Default is
                using Pampalk's method.
            - terhardt: boolean. Whether or not to apply the Terhardt perception model
                (1979). Default is False.

        :returns:
            - np.ndarray[shape=(num_segments,)]: if decomposition == True
            - float: if decomposition == False
    '''
    if all_fp is None and y is None:
        print('Invalid paramters: need either audio or fluctuation patterns')
    if all_fp is None:
        all_fp = fluctuationPatterns(y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                        mel_count=mel_count, seconds=seconds, band_num=band_num,
                        max_freq=max_freq, Pampalk=Pampalk, terhardt=terhardt)
    # Make segment length a power of two; use same method as fluctuation patterns
    segment_length = 2**prevPow(sr * seconds / n_fft) # alt: nextPow()
    # Calculate the number of frequency bins from 0 to maxfreq
    resolution = math.ceil(max_freq / (sr / n_fft) * segment_length)

    if not decomposition:
        fp = np.median(all_fp,axis=0)
        fp = fp.reshape((band_num,resolution))
        fp = np.sum(fp,axis=0)/np.sum(np.sum(fp,axis=0))
        return -1 * np.dot(fp, np.log10(fp)) / log10(resolution)
    else:
        entropy_vals = np.zeros(np.shape(all_fp)[0]) #  int(num_segments)
        # Calculate fluctuation entropy for each frame
        for i in range(np.shape(all_fp)[0]):
            fp = all_fp[i,:].reshape((band_num,resolution))
            # Calculate a probability value for each fluctuation frequency
            fp = np.sum(fp, axis=0) / np.sum(np.sum(fp, axis=0))
            # Calculate entropy value
            entropy_vals[i] = -1 * np.dot(fp, np.log10(fp)) / np.log10(resolution)
        return entropy_vals

def fluctuationFocus(y=None, sr=44100, all_fp=None, n_fft=512, hop_length=512,
                        mel_count=36, seconds=3, band_num=12, max_freq=10,
                        Pampalk=True, terhardt=False, decomposition=True):
    '''
    Calculates the focus of the fluctuation patterns of the audio piece.
    Based on computation from E. Pampalk's PhD thesis (2006) paper.
    Can calculate for either the median fluctuation pattern or for all
    fluctuation patterns.
        :usage:
                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> # Calculate the fluctuation entropy of an audiofile
                >>> fp_focus = extractor.fluctuationFocus(y, sr=sr,
                        decomposition=True, n_fft=512, hop_length=512,
                        mel_count=36, seconds=3, band_num=12, max_freq=10,
                        Pampalk=True, terhardt=False)

                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> fluctuation_patterns =
                                extractor.fluctuationPatterns(y, sr=sr)
                >>> # Calculate the fluctuation entropy from fp values
                >>> fp_focus =
                      extractor.fluctuationFocus(all_fp=fluctuation_patterns)

        :parameters:
            - y : np.ndarray [shape=(n,)] Time series of the audio file.
                Default None.
            - sr : integer. sampling rate of the audio file. Default 44100.
            - all_fp: np.ndarray[shape=(segment_num, band_num*resolution)].
                Output of fluctuation_patterns. Default None.
            - decomposition: boolean. Whether to only consider the median
                fluctuation pattern or all fluctuation patterns.
            - n_fft : integer. FFT window size for
                librosa.feature.melspectrogram() [samples]. Default 512.
            - hop_length : integer. The amount of overlap [samples] between
                the frames for librosa.feature.melspectrogram(). Default is
                same values as n_fft. Default 512.
            - mel_count : integer. The number of mel bands to consider when
                generating the melspectrogram. Default is 36 bands.
            - seconds : integer. The length of each segment when calculating
                the fluctuation patterns. Default is 3 seconds.
            - band_num : integer. The number of bands to consider when calculating
                the fluctuation patterns. The mel bands resulting from the initial
                melspectrogram will be combined together to get this many bands.
                Default is 12 bands.
            - max_freq : integer. The maximum modulation frequency to be considered.
                Default is 10 Hz.
            - Pampalk: boolean. Whether to use Pampalk's algorithm straight from his
                thesis (i.e. with the hardcoded gaussian values) or to use an
                alternate method involving a built-in gaussian filter. Default is
                using Pampalk's method.
            - terhardt: boolean. Whether or not to apply the Terhardt perception model
                (1979). Default is False.

        :returns:
            - np.ndarray[shape=(num_segments,)]: if decomposition == True
            - float: if decomposition == False
    '''
    if all_fp is None and y is None:
        print('Invalid args: need either audio file or fluctuation patterns')
    if all_fp is None:
        all_fp = fluctuationPatterns(y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                        mel_count=mel_count, seconds=seconds, band_num=band_num,
                        max_freq=max_freq, Pampalk=Pampalk, terhardt=terhardt)
    if not decomposition:
        fp = np.median(all_fp,axis=0)
        return np.mean(np.divide(fp, max(fp)))
    else:
        # Initialize array to hold results
        foci = np.zeros(np.shape(all_fp)[0]) #int(num_segments)
        for i in range(np.shape(all_fp)[0]):
            # Calculate focus value for each frame
            # Average normalized fluctuation value
            foci[i] = np.mean(np.divide(all_fp[i,:], max(all_fp[i,:])))
        return foci

def fluctuationCentroid(y=None, sr=44100, all_fp=None, n_fft=512,
                        hop_length=512, mel_count=36, seconds=3, band_num=12,
                        max_freq=10, Pampalk=True, terhardt=False,
                        decomposition=True):
    '''
    Calculates the centroid of the fluctuation patterns of the audio piece.
    Based on computation from E. Pampalk's PhD thesis (2006) paper.
    Can calculate for either the median fluctuation pattern or for all
    fluctuation patterns.
        :usage:
                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> # Calculate the fluctuation entropy of an audiofile
                >>> fp_centroid = extractor.fluctuationCentroid(y, sr=sr,
                        decomposition=True, n_fft=512, hop_length=512,
                        mel_count=36, seconds=3, band_num=12, max_freq=10,
                        Pampalk=True, terhardt=False)

                >>> # Load a file
                >>> y, sr = librosa.load('file.mp3')
                >>> fluctuation_patterns =
                                    extractor.fluctuationPatterns(y, sr=sr)
                >>> # Calculate the fluctuation entropy from fp values
                >>> fp_centroid =
                    extractor.fluctuationCentroid(all_fp=fluctuation_patterns)

        :parameters:
            - y : np.ndarray [shape=(n,)] Time series of the audio file.
                Default None.
            - sr : integer. sampling rate of the audio file. Default 44100.
            - all_fp: np.ndarray[shape=(segment_num, band_num*resolution)].
                Output of fluctuation_patterns. Default None.
            - decomposition: boolean. Whether to only consider the median
                fluctuation pattern or all fluctuation patterns.
            - n_fft : integer. FFT window size for
                librosa.feature.melspectrogram() [samples]. Default 512.
            - hop_length : integer. The amount of overlap [samples] between
                the frames for librosa.feature.melspectrogram(). Default is
                same values as n_fft. Default 512.
            - mel_count : integer. The number of mel bands to consider when
                generating the melspectrogram. Default is 36 bands.
            - seconds : integer. The length of each segment when calculating
                the fluctuation patterns. Default is 3 seconds.
            - band_num : integer. The number of bands to consider when calculating
                the fluctuation patterns. The mel bands resulting from the initial
                melspectrogram will be combined together to get this many bands.
                Default is 12 bands.
            - max_freq : integer. The maximum modulation frequency to be considered.
                Default is 10 Hz.
            - Pampalk: boolean. Whether to use Pampalk's algorithm straight from his
                thesis (i.e. with the hardcoded gaussian values) or to use an
                alternate method involving a built-in gaussian filter. Default is
                using Pampalk's method.
            - terhardt: boolean. Whether or not to apply the Terhardt perception model
                (1979). Default is False.

        :returns:
            - np.ndarray[shape=(num_segments,)]: if decomposition == True
            - float: if decomposition == False
    '''
    if all_fp is None and y is None:
        print('Invalid args: need either audio file or fluctuation patterns')
    if all_fp is None:
        all_fp = fluctuationPatterns(y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                        mel_count=mel_count, seconds=seconds, band_num=band_num,
                        max_freq=max_freq, Pampalk=Pampalk, terhardt=terhardt)
    # Make segment length a power of two; use same method as fluctuation patterns
    segment_length = 2**prevPow(sr * seconds / n_fft)  # alt: nextPow()
    # Calculate the number of frequency bins from 0 to maxfreq
    resolution = math.ceil(max_freq / (sr / n_fft) * segment_length)

    if not decomposition:
        fp = np.median(all_fp,axis=0)
        fp = fp.reshape((band_num,resolution))
        return np.dot(np.sum(fp,axis=0),np.arange(resolution)) / np.sum(fp)
    else:
        # Initialize array to hold results
        centroids = np.zeros(np.shape(all_fp)[0]) #int(num_segments)
        for i in range(np.shape(all_fp)[0]):
            fp = all_fp[i,:].reshape((band_num,resolution))
            # Calculate the fluctuation pattern centroid (center of mass)
            centroids[i] = np.dot(np.sum(fp,axis=0),np.arange(resolution)) \
                                    / np.sum(fp)
        return centroids

def MPS():
    pass
