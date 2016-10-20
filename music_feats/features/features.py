from __future__ import division
import librosa
import numpy as np
from music.features import extractor
from music.features.util.utils import *

__all__ = ['Features']

class Features(object):

    def __init__(self, audiofile, TR=2.0045, n_fft=2048, sr=44100, pad=None):
        """
        Initializes a Features object that can be used to extract desired
        acoustic features from the provided audiofile.

            :parameters:
                - audiofile : .wav file to associate with this feature
                    generator object.
                - TR : float. The TR length used for the fMRI scans.
                    Default is 2.0045 seconds.
                - n_fft : integer. The windowing length to be used for feature
                    extraction. Default is 2048 samples.
                - sr : integer. The sampling rate to be used.
                    Default is 44100 Hz.
                - pad : integer. How much to pad the audio file by before
                    extracting features. Default is half n_fft value.
        """
        self.audiofile = audiofile
        (self.audio, self.sr) = self.load_audio(sr=sr)

        self.n_fft = n_fft
        self.TR = TR
        self.all_features = {}
        self.featureParams = {}
        self.featureDim = {}
        if pad is None:
            pad = int(n_fft / 2)
        self.pad = pad

    def load_audio(self, sr=44100):
        """ 
        Load the audio file using librosa's load function.
            :parameters:
                - sr : integer. Sampling rate to be used.
                    Default is 44100 Hz.
        """
        return librosa.load(self.audiofile, sr=sr)

    def saveFeatures(self):
        """
        Returns the extracted features to be saved by user.
            :returns:
                - all_features : dict. Dictionary of extracted features.
                - featureParams : dict. Dictionary of paramters used to
                                  extract features.
                - featureDim : dict. Dictionary of the dimensions of the
                               extracted features.
        """
        return self.all_features, self.featureParams, self.featureDim

    def insertFeature(self, val, ft, hop_length, full=False):
        """
        Inserts a feature into the data structure storing all of the extracted
        features. Also inserts the feature dimension into a separate dictionary.
            :parameters:
                - val : np.ndarray. Extracted feature value
                - ft : string. Feature name
                - hop_length : integer. The hop_length used when extracting
                  the feature.
                - full : boolean. Whether or not it is a time-series/windowed
                  analysis. Default is False.
        """
        if not full:
            val = np.tile(val, (1,1))
            self.featureDim[ft] = val.shape
        self.all_features[ft] = val

    def returnFeature(self, feature):
        """
        Returns the value of an extracted feature.
            :parameters:
                - feature : string. The feature of interest.
        """
        return self.all_features[feature]

    def returnFeatureDim(self, feature):
        """
        Returns the dimension of an extracted feature.
            :parameters:
                - feature : string. The feature of interest.
        """
        return self.featureDim[feature]

    def returnParams(self, feature):
        """
        Returns the parameters used to extract the feature.
            :parameters:
                - feature: string. Return the extraction parameters
                    of this feature.
        """
        return self.featureParams[feature]

    def featureExists(self, f):
        """
        Returns whether or not a feature exists.
            :parameters:
                - f : string. Which feature to check existance of.
        """
        return (f in self.all_features)

    def extractedFeatures(self):
        """
        Returns a list of all extracted features.
            :Returns:
                - list of features extracted
        """
        return self.all_features.keys()

    def insertExtractionParameters(self, feature, params):
        """
        Inserts the set of extraction parameters used for this feature into the
        featureParams dictionary 
            :parameters:
                - feature : string. The feature whose extraction parameters
                                    we are inserting.
                - kwargs: each extraction parameter should be treated as a
                          parameter of the function.
            :usage:
                >>> insertExtractionParameters('RMS', n_fft=n_fft, hop_length=hop_length)
        """
        params['sr'] = self.sr
        params['audio_len_samples'] = len(self.audio)
        self.featureParams[feature] = params	

    ### LOUDNESS FEATURES ###
    def rms(self, use_librosa=False, decomposition=True, hop_length=None,
            n_fft=None):
        """
        Get the root mean square (RMS) energy feature.
            :parameters:
                - use_librosa : boolean. Whether to use librosa or extractor
                    feature extraction function. Default False.
                - decomposition : boolean. Whether to look at a time-series or
                    an overall root mean square analysis of the piece.
                - win_length : float. Window length for the frames in the
                    time series analysis. [seconds]. Default is 0.05 s.
                - hop_length : float. Hop length (i.e. overlap) between the
                    frames in the time series analysis [samples].
                    Default is half-overlap.
                - n_fft : integer. The window length [samples].
                          Default 2048 [samples].
        """

        if n_fft is None:
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = int(n_fft / 2)
        if use_librosa:
            tmp = librosa.feature.rmse(self.audio, n_fft=n_fft,
                                        hop_length=hop_length)
            self.insertFeature(tmp, 'RMS', hop_length, full=False)
            self.insertExtractionParameters('RMS',
                                            dict(n_fft=n_fft, librosa=use_librosa,
											hop_length=hop_length, decomposition=True))
        else:
            tmp = extractor.rms(self.audio, sr=self.sr, n_fft=n_fft, hop_length=hop_length,
                                        pad=self.pad, decomposition=decomposition)
            self.insertFeature(tmp, 'RMS', hop_length, full=not decomposition)
            self.insertExtractionParameters('RMS',
                                            dict(n_fft=n_fft, hop_length=hop_length,
											pad=self.pad, decomposition=decomposition,
											librosa=use_librosa))
        return tmp

    def temporalEnvelope(self):
        """
        Get temporal amplitude feature.
        """

        self.all_features['temporalEnv'] = \
                            extractor.temporalEnvelope(self.audio)
		
        self.insertExtractionParameters('temporalEnv', dict(librosa=False))
        return self.all_features['temporalEnv']

    def temporalFlatness(self, n_fft=None, hop_length=None,
                            decomposition=True):
        """
        Get temporal flatness feature.
            :parameters:
                - n_fft : integer. The window length [samples].
                    Default 2048 [samples].
                - hop_length : float. Hop length (i.e. overlap) between the
                    frames in the time series analysis [samples].
                    Default is half-overlap.
                - decomposition : boolean. Whether to look at a time-series or
                    an overall root mean square analysis of the piece.
        """
        if n_fft is None:
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = int(self.n_fft / 2)

        tmp = \
            extractor.temporalFlatness(self.audio, sr=self.sr,
                n_fft=n_fft, hop_length=hop_length, pad=self.pad,
                decomposition=decomposition)
        
        self.insertFeature(tmp, 'temporalFlatness', hop_length, full=not decomposition)
        self.insertExtractionParameters('temporalFlatness',
                                        dict(hop_length=hop_length,
										pad=self.pad, n_fft=n_fft,
										decomposition=decomposition, librosa=False))
        return tmp

    ### TIMBRAL FEATURES ###
    def zcr(self, use_librosa=False, decomposition=True, p='second', d='one',
            hop_length=None, n_fft=None):
        """
        Get the zero crossing rate feature.
        :parameters:
            - use_librosa : boolean. Whether to use librosa or extractor
                feature extraction function. Default False.
            - decomposition : boolean. Whether to look at a time-series or
                an overall root mean square analysis of the piece.
            - p : boolean. p='second' or p='sample'. Calculate zero crossing
                rate per these two units of time. Default is 'second'.
            - d : boolean. d='one' or d='both'. Calculate zero crossing rate
                from one direction or both directions of approach to x-axis.
                Default is 'one'
            - n_fft : integer. The window length [samples].
                Default 2048 [samples].
            - hop_length : float. Hop length (i.e. overlap) between the
                frames in the time series analysis [samples].
                Default is half-overlap.
            - n_fft : integer. The window length [samples]. For using with
                librosa. Default 2048 [samples].
        """

        if n_fft is None:
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = int(n_fft / 2)

        if use_librosa:
            tmp = librosa.feature.zero_crossing_rate(self.audio,
                                    n_fft=n_fft, hop_length=hop_length)
            self.insertFeature(tmp, 'ZCR', hop_length, full=False)
            self.insertExtractionParameters('ZCR',
                                            dict(n_fft=n_fft, hop_length=hop_length,
											librosa=use_librosa, decomposition=True))
        else:
            tmp = extractor.zcr(self.audio, sr=self.sr, p=p,
                        d=d, n_fft=n_fft, hop_length=hop_length,
                        pad=self.pad, decomposition=decomposition)
            self.insertFeature(tmp, 'ZCR', hop_length, full=not decomposition)
            self.insertExtractionParameters('ZCR',
                                            dict(hop_length=hop_length,
											decomposition=decomposition, 
											n_fft=n_fft, per=p, direction=d,
											pad=self.pad, librosa=use_librosa))
        return tmp

    # SPECTRAL FEATURES #
    def spectralCentroid(self, use_librosa=False, decomposition=True,
                        n_fft=None, hop_length=None):
        """
        Get spectral centroid feature.
        :parameters:
            - use_librosa : boolean. Whether to use librosa or extractor
                feature extraction function. Default False.
            - decomposition : boolean. Whether to look at a time-series or
                an overall root mean square analysis of the piece.
            - n_fft : integer. The window length [samples]. For using with
                librosa. Default 2048 [samples].
            - win_length : float. Window length for the frames in the
                time series analysis. [seconds]. Default is 0.05 s.
            - hop_length : float. Hop length (i.e. overlap) between the
                frames in the time series analysis [samples].
                Default is half-overlap.
        """

        if n_fft is None:
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = int(n_fft / 2)

        if use_librosa:
            tmp = librosa.feature.spectral_centroid(y=self.audio,
                        sr=self.sr, n_fft=n_fft, hop_length=hop_length)
            self.insertFeature(tmp, 'spectralCentroid', hop_length, full=False)
            self.insertExtractionParameters('spectralCentroid',
                                            dict(hop_length=hop_length,
											n_fft=n_fft, librosa=use_librosa,
											decomposition=True))
        else:
            tmp = extractor.spectralCentroid(self.audio, sr=self.sr,
                            n_fft=n_fft, hop_length=hop_length,
                            pad=self.pad, decomposition=decomposition)
            self.insertFeature(tmp, 'spectralCentroid', hop_length, full=not decomposition)
            self.insertExtractionParameters('spectralCentroid',
                                            dict(hop_length=hop_length,
											n_fft=n_fft, pad=self.pad,
											decomposition=decomposition, librosa=use_librosa))
        return tmp

    def spectralSpread(self, use_librosa=False, decomposition=True,
                       hop_length=None, n_fft=None):
        """
        Get spectral spread/bandwidth feature.
        :parameters:
            - use_librosa : boolean. Whether to use librosa or extractor
                feature extraction function. Default False.
            - decomposition : boolean. Whether to look at a time-series or
                an overall root mean square analysis of the piece.
            - win_length : float. Window length for the frames in the
                time series analysis. [seconds]. Default is 0.05 s.
            - hop_length : float. Hop length (i.e. overlap) between the
                frames in the time series analysis [samples].
                Default is half-overlap.
            - n_fft : integer. The window length [samples]. For using with
                librosa. Default 2048 [samples].
        """

        if n_fft is None:
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = int(n_fft / 2)
        if use_librosa:
            tmp = librosa.feature.spectral_bandwidth(y=self.audio,
                            sr=self.sr, n_fft=n_fft, hop_length=hop_length)
            self.insertFeature(tmp, 'spectralSpread', hop_length, full=False)
            self.insertExtractionParameters('spectralSpread',
                                            dict(hop_length=hop_length,
											n_fft=n_fft, librosa=use_librosa,
											decomposition=True))
        else:
            tmp = extractor.spectralSpread(self.audio,
                            sr=self.sr, n_fft=n_fft,
                            hop_length=hop_length, pad=self.pad,
                            decomposition=decomposition)
            self.insertFeature(tmp, 'spectralSpread', hop_length, full=not decomposition)
            self.insertExtractionParameters('spectralSpread',
                                            dict(hop_length=hop_length,
											n_fft=n_fft, librosa=use_librosa,
											decomposition=decomposition, pad=self.pad))
        return tmp

    def spectralFlatness(self, decomposition=True, n_fft=None,
                            hop_length=None): #win_length=None
        """
        Get spectral flatness feature.
        :parameters:
            - decomposition : boolean. Whether to look at a time-series or
                an overall root mean square analysis of the piece.
            - win_length : float. Window length for the frames in the
                time series analysis. [seconds]. Default is 0.05 s.
            - hop_length : float. Hop length (i.e. overlap) between the
                frames in the time series analysis [samples].
                Default is half-overlap.
        """
        if n_fft is None:
            n_fft = self.n_fft
        # if win_length is None:
        #     win_length = self.n_fft / self.sr
        if hop_length is None:
            hop_length = int(self.n_fft / 2)

        tmp = extractor.spectralFlatness(self.audio, sr=self.sr,
                            n_fft=n_fft, hop_length=hop_length,
                            pad=self.pad, decomposition=decomposition) # win_length=win_length,

        self.insertFeature(tmp, 'spectralFlatness', hop_length, full=not decomposition)
        self.insertExtractionParameters('spectralFlatness',
                                        dict(hop_length=hop_length,
										n_fft=n_fft, pad=self.pad,
										decomposition=decomposition, librosa=False)) # win_length=win_length
        return tmp

    def spectralContrast(self, n_fft=None, hop_length=None, fmin=200.0,
                                n_bands=6, linear=False):
        """
        Get spectral contrast feature.
        :parameters:
            - n_fft : integer. The window length [samples]. For using with
                librosa. Default 2048 [samples].
            - hop_length : integer. Hop length (i.e. overlap) between the
                frames in the time series analysis [samples].
                Default is 1024 [samples].
            - fmin : float. Frequency cutoff for the first bin `[0, fmin]`.
                Default 200.0.
            - n_bands : integer. Number of frequency bands. Default 6.
            - linear : boolean. If `True`, return the linear difference of
                magnitudes. If `False`, return the logarithmic difference.
                Default is False.
        """

        if n_fft is None:
            n_fft = int(self.n_fft)
        if hop_length is None:
            hop_length = int(n_fft / 2)
        tmp = librosa.feature.spectral_contrast(y=self.audio, sr=self.sr,
                        n_fft=n_fft, hop_length=hop_length,
                        fmin=fmin, n_bands=n_bands, linear=linear)
        self.insertFeature(tmp, 'spectralContrast', hop_length, full=False)
        self.insertExtractionParameters('spectralContrast',
                                        dict(hop_length=hop_length,
										n_fft=n_fft, fmin=fmin, n_bands=n_bands,
										band_diff_linear=linear, band_diff_log=not linear,
										decomposition=True, librosa=True))
        return tmp

    def spectralRolloff(self, n_fft=None, hop_length=None, roll_percent=0.85):
        """"
        Get spectral rollof feature.
        :parameters:
            - n_fft : integer. The window length [samples]. For using with
                librosa. Default 2048 [samples].
            - hop_length : integer. Hop length (i.e. overlap) between the
                frames in the time series analysis [samples].
                Default is 1024 [samples].
            - roll_percent : float. Roll-off percentage (between 0 and 1).
                Default is 0.85.
        """

        if n_fft is None:
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = int(n_fft / 2)
        tmp = librosa.feature.spectral_rolloff(y=self.audio,
                        sr=self.sr, n_fft=n_fft, hop_length=hop_length,
                        roll_percent=roll_percent)
        self.insertFeature(tmp, 'spectralRolloff', hop_length, full=False)
        self.insertExtractionParameters('spectralRolloff',
                                        dict(hop_length=hop_length,
										n_fft=n_fft, roll_percent=roll_percent,
										librosa=True, decomposition=True))
        return tmp

    def mfcc(self, n_mfcc=20, n_fft=None, hop_length=None):
        """
        Get the mel-frequency cepstral coefficients.
        :parameters:
            - n_mfcc : integer. Number of MFCCs to return.
            - n_fft : integer. Window size in [samples].
                Default is self.n_fft.
            - hop_length : integer. Overlap amount in [samples].
                Default is half overlap.
        """

        if n_fft is None:
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = int(n_fft / 2)
        tmp  = librosa.feature.mfcc(y=self.audio,
                sr=self.sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        self.insertFeature(tmp, 'MFCC', hop_length, full=False)
        self.insertExtractionParameters('MFCC',
                                        dict(hop_length=hop_length, n_fft=n_fft,
										decomposition=True, n_mfcc=n_mfcc, librosa=True))
        return tmp

    def melspectrogram(self, dB=True, n_fft=None, hop_length=None, n_mels=128):
        """
        Get the melspectrogram of the audiofile.
        :parameters:
            - n_fft : integer. The window length [samples]. For using with
                librosa. Default 2048 [samples].
            - dB : boolean. Whether or not to output in log (dB) scale.
                Default is True.
            - hop_length : integer. Hop length (i.e. overlap) between the
                frames in the time series analysis [samples].
                Default is 1024 [samples].
            - n_mels : integer. Number of mel frequency bands.
                Default is 128.
        """

        if n_fft is None:
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = int(n_fft / 2)
        tmp = librosa.feature.melspectrogram(y=self.audio, sr=self.sr,
                n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        if dB:
            tmp = librosa.logamplitude(tmp, ref_power=np.max)
        self.insertFeature(tmp, 'MELSPECT', hop_length, full=False)
        self.insertExtractionParameters('MELSPECT',
                                      dict(dB=dB, hop_length=hop_length,
									  n_fft=n_fft, n_mels=n_mels, librosa=True,
									  decomposition=True))
        return tmp

    def stft(self, dB=True, n_fft=None, hop_length=None):
        """
        Get the STFT of the audio file.
        :parameters:
            - dB : boolean. Whether or not to output power spectrogram.
                Default is True.
            - n_fft : integer. The window length [samples]. For using with
                librosa. Default 2048 [samples]. 
            - hop_length : integer. The amount of overlap between windows.
                in [samples]. Default is half overlap.
        """

        if n_fft is None:
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = int(n_fft / 2)
        S = librosa.core.stft(self.audio, n_fft=n_fft, hop_length=hop_length)
        if dB:
            S = librosa.logamplitude(np.abs(S)**2, ref_power=np.max)
        self.insertFeature(S, 'S', hop_length, full=False)
        self.insertExtractionParameters('S',
                                        dict(dB=dB, hop_length=hop_length, n_fft=n_fft,
										librosa=True, decomposition=True))
        return S
    
    def CQT(self, cqt_hop=None, seconds=2.0, n_bins=30, bins_per_octave=4, fmin=27.5,
        	use_han=False):
        """
        Get the constant-q transform of the audio file.
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
        """

        # cqt hop length needs to ~1024/44100 seconds by default; varies by sampling rate
        if cqt_hop is None:
        	cqt_hop = 2**(prevPow((1024/44100)*self.sr))
        CQTlog = extractor.CQT(self.audio, sr=self.sr, cqt_hop=cqt_hop, seconds=seconds, n_bins=n_bins,
        					   bins_per_octave=bins_per_octave, fmin=fmin, use_han=use_han)
        self.insertFeature(CQTlog, 'CQT', cqt_hop, full=False)
        self.insertExtractionParameters('CQT',
                                        dict(cqt_hop=cqt_hop, frame_seconds=seconds, n_bins=n_bins,
                                        bins_per_octave=bins_per_octave, fmin=fmin, librosa=True,
        								decomposition=True, use_han=use_han))
        return CQTlog
				
    ### TONAL FEATURES ###
    def chromagram(self, stft=True, S=None, n_fft=None, hop_length=None,
                        norm=np.inf, n_octaves=7, tuning=None, seconds=4,
                        center=True, use_librosa=True, **kwargs):
        """
        Get the chroma of the audio file.
        :parameters:
            - stft : boolean. Whether to calculate chroma using the short
                time fourier transform or constant-q. Default is True.
            - S : np.ndarray. Spectrogram output (to calculate from here), 
                rather than from audiofile. Only for STFT based method.
                Default is None.
            - n_fft : integer. The window length [samples] of the STFT.
                Default 2048 [samples].
            - hop_length : integer. Hop length (i.e. overlap) between the
                frames in the STFT analysis [samples].
                Default is 1024 [samples].
            - norm : float or None. Column-wise normalization. Default is
                np.inf.
            - n_octaves : integer. The number of octaves to analyze.
                Only for constant-q method. Default is 7.
            - tuning : Deviation in cents from A440 tuning. Default is
                None. Can be used with both methods.
            - seconds : integer. For extractor.chromagram(). The bin size
                for the spectrogram before calculating the chromagram.
            - center : boolean. For extractor.chromagram(). Whether or not
                to center the spectrogram before calculating the chromagram.
                Default is True.
            - librosa : boolean. Whether to use librosa's chromagram or
                extractor's chromagram. Default is True (librosa's).
            - kwargs : the arguments for librosa.filter.chroma()
        """

        if n_fft is None:
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = int(n_fft / 2)
        if stft:
            if use_librosa:
                chroma = \
                    librosa.feature.chroma_stft(y=self.audio, sr=self.sr, S=S,
                        norm=norm, n_fft=n_fft, hop_length=hop_length,
                        tuning=tuning, **kwargs)
            else:
                chroma = \
                    extractor.chromagram(y=self.audio, sr=self.sr, S=S, 
                        norm=norm, n_fft=n_fft, hop_length=None,
                        seconds=seconds, tuning=None, center=center,
                        **kwargs)

        else:
            chroma = \
                librosa.feature.chroma_cqt(y=self.audio, sr=self.sr,
                hop_length=hop_length, tuning=tuning, n_octaves=n_octaves)
        self.insertFeature(chroma, 'chroma', hop_length, full=False)
        self.insertExtractionParameters('chroma',
                                        dict(hop_length=hop_length, n_fft=n_fft,
										librosa=use_librosa, stft=stft, cqt=not stft,
										norm=norm, tuning=tuning, decomposition=True,
										n_octaves_ifcqt=n_octaves))
        return chroma

    def tonalCentroid(self, chroma=None):
        """
        Get the tonal centroid feature.
        :parameters:
            - chroma : Normalized energy for each chroma bin at each frameself.
                Default is None or self.returnFeature('chroma') if it exists.
        """

        if (chroma is None) and self.featureExists('chroma'):
            chroma = self.returnFeature('chroma')
        tmp = \
            librosa.feature.tonnetz(y=self.audio, sr=self.sr, chroma=chroma)
        self.insertFeature(tmp, 'tonalCentroid', None, full=False)
        self.insertExtractionParameters('tonalCentroid',
                                        dict(librosa=True,
										decomposition=True))
        return tmp

    def tonality(self, n_fft=None, hop_length=None, profiles='gomez',
                    seconds=4, center=True, use_librosa=True, full=False):
        """
        Get the chroma of the audio file.
        :parameters:
            - n_fft : integer. The window length [samples] for calculating
                the STFT. Default 2048 [samples].
            - hop_length : integer. Hop length (i.e. overlap) between the
                frames in the STFT analysis [samples].
                Default is 1024 [samples].
            - profiles : string. Which key profile to use
              Profiles available:
                - Gomez, 2006; Krumhansl, Cognitive Foundations of Pitch;
                Temperley The Krumhansl-Schmuckler Key-Finding Algorithm
                Revisted; Temperley, MIREX; Wei Chai MIT PhD Thesis
            - seconds : integer. For extractor.chromagram(). The bin size to
                bin the spectrogram frames with before calculating the chroma.
                Default is 4 seconds (rounded down to the nearest power of 2,
                ~ 3 seconds)
            - center : boolean. For extractor.chromagram(). Whether or not to
                center the spectrogram before calculating the chromagram.
                Default is True.
            - librosa : boolean. Whether to use librosa.chromagram() or
                extractor.chromagram(). Default is True.
            - full : boolean. Whether or not to also return key strengths
                for the entire piece vs. just the windowed/time-series analysis
                of the audio.
        """

        if n_fft is None:
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = int(n_fft / 2)
        tmp = extractor.tonality(self.audio, self.sr, profiles=profiles,
                                n_fft=n_fft, hop_length=hop_length,
                                seconds=seconds, center=center,
                                use_librosa=use_librosa, full=full)
        encodedKey = encodeKey(tmp[0])
        encodedMode = encodeMode(tmp[1])
        self.insertFeature(np.array(encodedKey), 'keyTimeSeries', hop_length, full=False)
        self.insertFeature(np.array(encodedMode), 'modeTimeSeries', hop_length, full=False)
        self.insertExtractionParameters('keyTimeSeries',
                                        dict(hop_length=hop_length, n_fft=n_fft,
										seconds=seconds, profiles=profiles, librosa=False,
										center=center, decomposition=True))
        self.insertExtractionParameters('modeTimeSeries',
                                        dict(hop_length=hop_length, n_fft=n_fft,
										seconds=seconds, profiles=profiles, librosa=False,
										center=center, decomposition=True))
        if full:
            self.insertFeature(tmp[2], 'majorKS', hop_length, full=True)
            self.insertExtractionParameters('majorKS',
                                            dict(hop_length=hop_length, n_fft=n_fft,
											seconds=seconds, profiles=profiles, librosa=False,
											center=center, decomposition=False))
            self.insertFeature(tmp[3], 'minorKS', hop_length, full=True)
            self.insertExtractionParameters('minorKS',
                                            dict(hop_length=hop_length, n_fft=n_fft,
											seconds=seconds, profiles=profiles, librosa=False,
											center=center, decomposition=False))
            # going to disregard 'keys' for the time being
            # self.insertFeature(tmp[4], 'keys', hop_length, full=True)
        return tmp

    ### RHYTHMIC FEATURES ###
    def fluctuationPatterns(self, n_fft=None, hop_length=None, mel_count=36,
                        seconds=None, band_num=12, max_freq=10, center=True,
                        padAmt=0.25, Pampalk=True, terhardt=False):
        """
        Extract the fluctuation patterns feature.
        :parameters:
            - n_fft : integer. The window length for the STFT [samples].
                For using with librosa. Default 2048 [samples].
            - hop_length : integer. Hop length (i.e. overlap) between the
                frames for the STFT calculation [samples].
                Default is 2048 [samples] (i.e. no overlap).
            - mel_count : integer. The number of mel frequency bands to
                consider in the STFT. Default is 36.
            - seconds : integer. The segment length for taking the second
                FFT in calculating the fluctuation patterns. Default is TR.
            - band_num : integer. The number of bands to merge the frequency
                bands in the STFT into for the second FFT. Default is 12.
            - max_freq : integer. The maximum fluctuation frequency to
                consider. Default is 10 Hz.
            - Pampalk : boolean. Whether or not to use Pampalk's method
                (i.e. hardcoded gaussian values) or to use the gaussian
                built in filter. Default is True.
            - terhardt : boolean. Whether or not to use terhardt's perception
                model weights. Default is False.
        """

        if n_fft is None:
            n_fft = self.n_fft # Originally default was 512
        if hop_length is None:
            hop_length = self.n_fft # Default is no overlap
        if seconds is None:
            seconds = self.TR
        tmp = \
                extractor.fluctuationPatterns(self.audio, sr=self.sr,
                                    n_fft=n_fft, hop_length=hop_length,
                                    mel_count=mel_count, seconds=seconds,
                                    band_num=band_num, max_freq=max_freq,
                                    center=center, padAmt=padAmt,
                                    Pampalk=Pampalk, terhardt=terhardt)
        # save and return transpose because that puts time along the x axis
        self.insertFeature(tmp.T, 'FP', hop_length, full=False)
        self.insertExtractionParameters('FP',
                                        dict(hop_length=hop_length, n_fft=n_fft,
										mel_count=mel_count, seconds=seconds,
										band_num=band_num, max_freq=max_freq,
										center=center, padAmt=padAmt,
										Pampalk=Pampalk, terhardt=terhardt,
										decomposition=True, librosa=False))
        return tmp.T

    def fluctuationFocus(self, n_fft=None, hop_length=None, mel_count=36,
                        seconds=None, band_num=12, max_freq=10, Pampalk=True,
                        terhardt=False, decomposition=True):
        """
        Calculate the fluctuation focus of the fluction patterns of the song.
        Can be done with or without having called self.fluctuationPatterns()
        first. Parameters same as self.fluctuationPatterns().
        """

        if n_fft is None:
            n_fft = self.n_fft # Originally default was 512
        if hop_length is None:
            hop_length = self.n_fft # Default is no overlap
        if seconds is None:
            seconds = self.TR
        if self.featureExists('FP'):
            fp = self.returnFeature('FP')
            # take tranpose of fp b/c calculations require time along y-axis
            tmp = extractor.fluctuationFocus(all_fp=fp.T,
                    seconds=seconds, n_fft=n_fft, hop_length=hop_length,
                    decomposition=decomposition)
        else:
            tmp = extractor.fluctuationFocus(self.audio, self.sr,
                        hop_length=hop_length, mel_count=mel_count,
                        seconds=seconds, band_num=band_num, max_freq=max_freq,
                        Pampalk=Pampalk, terhardt=terhardt,
                        decomposition=decomposition)
        self.insertFeature(tmp, 'FPfocus', hop_length, full=not decomposition)
        self.insertExtractionParameters('FPfocus',
                                        dict(hop_length=hop_length,
										n_fft=n_fft, seconds=seconds, decomposition=decomposition,
										librosa=False))
        return tmp

    def fluctuationCentroid(self, all_fp=None, n_fft=None,
                        hop_length=None, mel_count=36, seconds=None, band_num=12,
                        max_freq=10, Pampalk=True, terhardt=False,
                        decomposition=True):
        """
        Calculate the fluctuation centroid of the fluction patterns of the song.
        Can be done with or without having called self.fluctuationPatterns()
        first. Parameters same as self.fluctuationPatterns().
        """

        if n_fft is None:
            # Originally default was 512, TODO: check if accurate
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = self.n_fft # Default is no overlap
        if seconds is None:
            seconds = self.TR
        if self.featureExists('FP'):
            fp = self.returnFeature('FP')
            # calculations require time along the y axis, take transpose
            tmp = extractor.fluctuationCentroid(all_fp=fp.T,
                        seconds=seconds, n_fft=n_fft, hop_length=hop_length,
                        decomposition=decomposition)
        else:
            tmp = extractor.fluctuationCentroid(y=self.audio, sr=self.sr,
                        n_fft=n_fft, hop_length=hop_length, mel_count=mel_count,
                        seconds=seconds, band_num=band_num, max_freq=max_freq,
                        Pampalk=Pampalk, terhardt=terhardt,
                        decomposition=decomposition)
        self.insertFeature(tmp, 'FPcentroid', hop_length, full=not decomposition)
        self.insertExtractionParameters('FPcentroid',
                                        dict(hop_length=hop_length,
										n_fft=n_fft, seconds=seconds, librosa=False,
										decomposition=decomposition))
        return tmp

    def fluctuationEntropy(self, all_fp=None, n_fft=None,
                    hop_length=None, mel_count=36, seconds=None, band_num=12,
                    max_freq=10, Pampalk=True, terhardt=False,
                    decomposition=True):
        """
        Calculate the fluctuation entropy of the fluction patterns of the song.
        Can be done with or without having called self.fluctuationPatterns()
        first. Parameters same as self.fluctuationPatterns().
        """

        if n_fft is None:
            # Originally default was 512, TODO: check if accurate
            n_fft = self.n_fft
        if hop_length is None:
            hop_length = self.n_fft # Default is no overlap
        if seconds is None:
            seconds = self.TR
        if self.featureExists('FP'):
            fp = self.returnFeature('FP')
            # calculations require time along the y axis, take transpose
            tmp = extractor.fluctuationEntropy(all_fp=fp.T,
                        seconds=seconds, n_fft=n_fft, hop_length=hop_length,
                        decomposition=decomposition)
        else:
            tmp = extractor.fluctuationEntropy(y=self.audio, sr=self.sr,
                        n_fft=n_fft, hop_length=hop_length, mel_count=mel_count,
                        seconds=seconds, band_num=band_num, max_freq=max_freq,
                        Pampalk=Pampalk, terhardt=terhardt,
                        decomposition=decomposition)
        self.insertFeature(tmp, 'FPentropy', hop_length, full=not decomposition)
        self.insertExtractionParameters('FPentropy',
                                        dict(hop_length=hop_length, n_fft=n_fft,
										seconds=seconds, librosa=False,
										decomposition=decomposition))
        return tmp

    def MPS(self):
        pass
