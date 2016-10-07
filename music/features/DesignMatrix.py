from __future__ import division
import numpy as np
from scipy import signal
from math import floor
from music.features.util import interpdata
from music.features.util.utils import *

__all__ = ['DesignMatrix']

class DesignMatrix(object):
    def __init__(self, features, trfile=None, TR=2.0045,
                 sr=44100, interp="lanczos", **kwargs):
        """
            :parameters:
                - features : dictionary of extracted features. (Ouptut of features.py)
                - trfile : .report file that has TR times used to calculate the new
                    times the design matrix needs to be downsampled to.
                - TR : float. The TR length used for the fMRI scans.
                    Default is 2.0045 seconds.
                - sr : integer. The sampling rate to be used.
                    Default is 44100 Hz.
                - interp : string. The type of interpolation to be used.
                        Currently only one option, 'lanczos' (also default value).
                - [kwargs] are passed to the interpolation function.
                        window : integer. Lobs of  sinc function to be used.
                            Default is 3.  
                        cutoff_mult : float. Associated with cutoff frequency.
                            Default is 1.0 (Nyquist). <1.0 is low-pass filter.
                        rectify : boolean. Default is False.
        """
        self.all_features = features
        self.trfile = trfile
        self.TR = TR
        self.sr = sr

        self.interp = interp
        self.interpargs = kwargs

    def returnFeature(self, feature):
        """
        Returns the value of an extracted feature.
            :parameters:
                - feature : string. The feature of interest.
        """
        return self.all_features[feature]

    def assignMatrix(self, matrix):
        """
        Stores an input matrix as the design matrix attribute for this
        instance of the DesignMatrix object.
            :parameters:
                - matrix : np.ndarry [(time, feature_dims)]. The feature matrix to save.
        """
        self.feature_matrix = matrix

    def assignFeatsets(self, features_used):
        """
        Stores the features that are used to construct the particular
        design matrix stored in self.feature_matrix.
            :parameters:
                - features_used : list. The list of features used in
                    the design matrix.
        """
        self.featsets = features_used

    def returnFeatsets(self):
        """
        Returns the features used to construct the design matrix.
        """
        return self.featsets

    def buildMatrix(self, feat, time_length, retrieve=True):
        """
        Build a feature matrix row by columns (time by features).
            :parameters:
                - feat : array of strings or array of features (type np.ndarray)
                    A list of features with which to build the feature matrix.
                    If array of strings, retrieve parameter must be set to True.
                    Otherwise, retireve parameter must be set to False.
                - time_length : integer. The length of a feature generated
                    by a not padded version of the audiofile. (i.e. the length
                    of a feature extracted by extractor). Pass in the length of the
                    first feature in feat. (i.e, feat[0])
                - retrieve : boolean. Indicates whether the object will
                    need to retrieve the feature (if feat is a string array)
                    or if the features are provided (in feat). Default is True.
                    (i.e. this expects feat to be an array of strings)
        """

        if retrieve:
            self.feature_matrix = np.transpose(self.returnFeature(feat[0]))
        else:
            self.feature_matrix = np.transpose(feat[0])
        for i in range(1, len(feat)):
            if retrieve:
                curr = self.returnFeature(feat[i])
            else:
                curr = feat[i]
            # deal with features that are of incorrect length
            if curr.shape[1] != time_length:
                if retrieve:
                    print feat[i] + ' is of mismatched size compared to the rest'
                else:
                    print i + '-th feature is of mismatched size'
                continue
            self.feature_matrix = np.hstack((self.feature_matrix,
                                                np.transpose(curr)))
        return self.feature_matrix

    def downsampleMatrix(self, oldlen, audiolen=None, method='inter',
                         feature_matrix=None, **kwargs):
        """
        Downsamples the feature matrix using the settings specified
        in the initializer.
            :parameters:
                - oldlen : integer. The length of a feature generated
                    (i.e. the length of a feature extracted by extractor)
                - audiolen : integer. 
                - method : string. The downsampling method to be used when downsampling
                    the design matrix.
                - feature_matrix : np.ndarray. Feature_matrix. Rows correspond
                    to time, columns correspond to features. Default is None.
                    if None, will try to access object feature_matrix. If no
                    feature_matrix attribute exists, will raise exception.
                - nonlin : string. The nonlinearity to add to values. Choices
                    are 'squared', 'sqrt', None. Default is 'squared'.
                - **kwargs : The arguments for helper functions calculateOldtimes()
        """
        if feature_matrix is None and not hasattr(self, 'feature_matrix'):
            raise ValueError('Need a feature matrix to downsample')
        if feature_matrix is None and hasattr(self, 'feature_matrix'):
            feature_matrix = self.feature_matrix
        if method=='inter' and audiolen is None:
            raise ValueError('Need to pass in the length of the audio file.')
        if method == 'inter':
            self.feature_matrix = self.interpolateDownsample(feature_matrix,
                                        oldlen, audiolen, **kwargs)
        elif method == 'average':
            self.feature_matrix = self.averageDownsample(feature_matrix, **kwargs)
        elif method=='interCQT':
            # same as above, but designed specifically for the cqt feature
            self.feature_matrix = self.interpolateCQTDownsample(feature_matrix,
                                        oldlen, **kwargs)
        self.feature_matrix = np.nan_to_num(self.feature_matrix)
        return self.feature_matrix

    def downsampleFeature(self, vals, oldlen, audiolen=None, method='inter', **kwargs):
        """
        Downsamples a specific feature to be the same length as the responses.
            :parameters:
                - vals : np.ndarray. The feature values that need to be
                    downsampled.
                - oldlen : integer. The old length of the features.
                - audiolen : integer. The length of the audio data file from which the
                    acoustic features were extracted. This parameter is only needed when
                    using the 'interp' method of downsampling. Default None.
                - method : string. The method of downsampling to use.
                    Choices are 'inter' (interpolate), 'man' (manual),
                    'resample' (sp.signal.resample()), or 'interCQT'
                    (same as 'inter' but specifically for CQT feature).
                    Default is 'inter.'
                - **kwargs : arguments for the resampling method to be used
                    and for calculateOldtimes()
        """
        # list of features that need to be manually downsampled
        # this is more for internal record keeping rather than for implementation
        man = ['modeTimeSeries', 'keyTimeSeries']
        if method=='inter' and audiolen is None:
            raise ValueError('Need to pass in the length of the audio file.')
        if method=='man':
            return np.nan_to_num(self.manualDownsample(vals, oldlen, **kwargs))
        elif method=='resample':
            return np.nan_to_num(self.resampDownsample(vals, **kwargs))
        elif method=='inter':
            # transpose vals b/c interpolation function expects time x features
            tmp = self.interpolateDownsample(vals.T, oldlen, audiolen, **kwargs)
            # transpose back to feature x time
            return np.nan_to_num(tmp.T)
        elif method=='interCQT':
            # same as above, but designed specifically for the cqt feature
            tmp = self.interpolateCQTDownsample(vals.T, oldlen, **kwargs)
            return np.nan_to_num(tmp.T)

    def applyNonlinearity(self, vals, nonlin):
        """
        This is used as a helper function to the interpolation functions.
        Function applies  a desired nonlinearity to the values before downsampling them.
            :parameters:
                - vals : np.ndarray. The values that need to be downsampled.
                - nonlin : string. Type of nonlinearity to apply.
            :returns:
                - vals : np.ndarray. The result of applying the input nonlinearity
        """
        if nonlin == 'squared':
            vals = np.square(vals)
        elif nonlin == 'sqrt':
            vals = np.sqrt(vals)
        return vals

    def interpolateDownsample(self, vals, oldlen, audiolen, nonlin=None,
                              endTrim=False, newLen=None, **kwargs):
        """
        Uses interpolation to downsample the signal.
            :parameters:
                - vals : np.ndarray. The values that need to be downsampled.
                - oldlen : integer. The old length of the values.
                -  audiolen : integer. The length of the audio data file from which the
                    acoustic features were extracted.
                - nonlin : string. The nonlinearity to add to the signal.
                    Default is 'squared'
                - endTrim : boolean. Whether or not to trim the last 5 TRs for the new
                    set of time points (i.e., the new length) for the features. This is
                    used when using TR reports for calculate the new length/timepoints
                    and if the experimental setup had 5 TRs of no stimulus at the end of
                    the run. Default False.
                - newlen : integer. The new feature length to downsample to.
                    Used when no TR report available for the data to compute the new
                    time length of the features. Default None.
                - **kwargs : arguments of calculateOldtimes()
        """
        oldtime = self.calculateOldtimes(oldlen, audiolen, **kwargs)
        newtime = self.calculateTRtimes(endTrim=endTrim, newLen=newLen)
        # Add nonlinearity to make sure downsampling retains shape
        if nonlin is not None:
            vals = self.applyNonlinearity(vals, nonlin=nonlin)
        if self.interp == 'lanczos':
            newvals = interpdata.lanczosinterp2D(vals,
                                oldtime, newtime, **self.interpargs)
        return newvals

    def interpolateCQTDownsample(self, vals, oldlen, nonlin=None,
                                endTrim=False, newLen=None, **kwargs):
        """
        Uses interpolation to downsample a constant-Q transform feature matrix.
            :parameters:
                - vals : np.ndarray. The values that need to be downsampled.
                - oldlen : integer. The old length of the values.
                - nonlin : string. The nonlinearity to add to the signal.
                    Default is 'squared'
                - endTrim : boolean. Whether or not to trim the last 5 TRs for the new
                    set of time points (i.e., the new length) for the features. This is
                    used when using TR reports for calculate the new length/timepoints
                    and if the experimental setup had 5 TRs of no stimulus at the end of
                    the run. Default False.
                - newlen : integer. The new feature length to downsample to.
                    Used when no TR report available for the data to compute the new
                    time length of the features. Default None.
                - **kwargs : arguments of calculateOldtimes()
        """
        oldtime = self.calculateOldTimesCQT(oldlen, **kwargs)
        newtime = self.calculateTRtimes(endTrim=endTrim, newLen=newLen)
        # Add nonlinearity to make sure downsampling retains shape
        if nonlin is not None:
            vals = self.applyNonlinearity(vals, nonlin=nonlin)
        if self.interp == 'lanczos':
            newvals = interpdata.lanczosinterp2D(vals,
                                oldtime, newtime, **self.interpargs)
        return newvals

    def manualDownsample(self, vals, oldlen, newlen=None):
        """
        Manually downsamples a 1D array by tossing out values.
            :parameters:
                - vals : np.ndarray. The values that need to be
                    manually downsampled.
                - oldlen : integer. The old length of the array.
                - newlen : integer. The desired new downsampled
                    length of the array. Default is None, which
                    returns the original values, unchanged.
        """
        # TODO: increase functionality to work with 2D arrays
        if newlen is None:
            print 'No new length value provided, not downsampling...'
            return vals
        ds_rate = int(floor(oldlen/newlen))
        downsampled = [np.squeeze(vals)[ds_rate*i] for i in range(newlen)]
        assert len(downsampled) == newlen
        return np.tile(np.array(downsampled),(1,1))

    def resampDownsample(self, vals, newlen=None, axis=1):
        """
        Downsample by using scipy.signal.resample().
            :parameters:
                - vals : np.ndarray. The values to be downsampled.
                - newlen : integer. The new length the downsampled
                    signal should be. Default is None, returning
                    original values, untouched.
                - axis : integer. The axis along which to resample the
                    signal. Default is axis = 1.
        """
        if newlen is None:
            print 'New length value not provided, will not resample...'
            return vals
        return signal.resample(vals, newlen, axis=axis)

    def averageDownsample(self, vals, newlen=None, overlapsize=1):
        """
        Downsample feature matrix by taking a moving window overage of some overlap.
            :parameters:
                - vals : np.ndarray. The values to be downsampled. (size: time x feature num)
                - newlen : integer. The new length the downsampled signal should be.
                           Default is None, reutrning original values untouched.
                - overlapsize: integer. The amount of overlap when computing the averages.
                           Default is 1.
        """
        if newlen is None:
            print 'New length value not provided, will not average and downsample...'
            return vals
        newarr = np.zeros((newlen, vals.shape[1]))
        #winsize = vals.shape[0]/newlen + overlapsize
        winsize = vals.shape[0]//newlen + overlapsize
        padWidth = ((0, winsize - vals.shape[0] % newlen), (0,0))
        # make sure there is enough array to actually take window averages over
        toDownsample = np.pad(vals, pad_width=padWidth, mode='reflect')
        for i in range(newlen):
            start = i*(winsize-overlapsize)
            end = start + winsize
            newarr[i,:] = np.mean(toDownsample[start:end,:], axis=0)
        return newarr

    def calculateOldtimes(self, oldlen, audiolen, t='sc', start=None, end=None,
    							n_fft=2048, hop_length=None, padded=None):
        """
        Calculate the time points for all the bins created when extracting
        features.
            :parameters:
                - oldlen : integer. The length of a feature generated
                    by a not padded version of the audiofile. (i.e. the length
                    of a feature extracted by extractor)
                - audiolen : interger. The length of the audio signal (samples).
                - t : string. Either 'sc' or 'sp'. Indicates the unit of time
                    to use when downsampling the feature matrix.
                - start : integer. The first sample to start calculating the old
                    times from. Default: n_fft / 2
                - end : integer. The last sample to be included/considered for
                    calculating the old times. Default: len(audio) - n_fft/2
                - n_fft : integer. The window length used for the time-series
                    analysis of *all* features. Default is 2048 samples.
                - hop_length : integer. Overlap length used when extracting
                    features. Default is half of n_fft value (in samples)
                - padded : integer. The amount the signal was padded on one side
                    of the signal (assuming mirror padding to center signal).
                    Used to correct the samples. Default is None.
        """
        # if n_fft is None:
        #     n_fft = self.n_fft
        if hop_length is None:
            hop_length = int(n_fft / 2)
        if padded is None:
            padded = int(n_fft // 2)
        if start is None:
            start = n_fft / 2
        if end is None:
            # end = len(self.audio) + 2 * padded - int(n_fft / 2)
            end = audiolen + 2 * padded - int(n_fft / 2)
        # assumption here is that after cropping padded extracted values,
        # the array 'starts' at 0 seconds; could be several milliseconds off
        # if something other than the default padding amount is used
        time = np.arange(start, end+1, hop_length)
        if hop_length == 1: #TODO: find a better way to calculate time w/ hop == 1
            time = time[:-1]
        assert len(time) == oldlen # correct number of time bins
        # shift the 'times' to account for padding
        # TODO: account for padding by default; is accurate?
        time -= padded
        if t == 'sc':
            # convert to seconds
            return time / self.sr
        return time

    def calculateOldTimesCQT(self, oldlen, t='sc', cqt_hop=1024,
                             seconds=2.0, sr=44100):
        """
        This function is used to calculate the time points for the samples of an
        extracted feature prior to downsampling.
            :parameters:
                - oldlen : integer. The length of a feature prior to downsampling
                    (i.e. the length of a feature extracted by extractor).
                - t : string. Either 'sc' or 'sp'. Indicates the unit of time
                    to use when downsampling the feature matrix.
                - cqt_hop : integer. The extraction parameter value corresponding to
                    cqt_hop when extract CQT feature. Default 1024 [samples].
                - seconds : float. The number of seconds used as the time window for the
                    initial chunking of the soundfile before computing the CQT. Default 2.0.
                - sr : integer. Sampling rate of the audiofile from which CQT was extracted.
                    Default 44100 [samples per second].
            :returns:
                - oldtimes : np.ndarry. The time points of the extracted time series.
        """
        # TODO: find a way to integrate this with the pre-existing downsampling stuff,
        # (maybe via variables/parameters)
        
        # compute framelength, the same way computed in cqt based off cqt extraction params
        frame_length = seconds * sr
        frame_length = (frame_length//cqt_hop) * cqt_hop

        num_frames = oldlen/(frame_length/cqt_hop - 1) # CQT.shape[1]
        sig_len = num_frames * frame_length - (num_frames - 1) * cqt_hop
        time_per_cqt_samp = (sig_len / oldlen) #/sr # CQT.shape[1]
        oldtimes = np.arange(0.0, sig_len, time_per_cqt_samp) #sig_len/sr
        if len(oldtimes) < oldlen:
            oldtimes = np.append(oldtimes, oldtimes[-1]+time_per_cqt_samp)
        elif len(oldtimes) > oldlen:
            oldtimes = oldtimes[:-1]

        if t == 'sc':
            return oldtimes / self.sr
        return oldtimes

    def calculateTRtimes(self, trim=5, endTrim=False, newLen=None):
        """
        Calculate the TR start times in seconds. Uses TRFile object class.
        Returns an array of start times in seconds.
            :parameters:
                - trim : integer. How much to trim the beginning of
                    the TRtimes by. Depends on how many TRs passed before the
                    stimulus began.
                - endTrim : boolean. Whether or not to trim the last 5 TRs for the new
                    set of time points (i.e., the new length) for the features. This is
                    used when using TR reports for calculate the new length/timepoints
                    and if the experimental setup had 5 TRs of no stimulus at the end of
                    the run. Default False.
                - newlen : integer. The new feature length to downsample to.
                    Used when no TR report available for the data to compute the new
                    time length of the features. Default None.
        """
        if self.trfile is not None:
            tr = TRFile(self.trfile)
            newtimes = tr.get_reltriggertimes()
            # Discard the first to get song-only times
            if endTrim:
                return newtimes[trim:-trim]
            return newtimes[trim:]
        else:
            if newLen is None:
                raise ValueError('Require a new length to downsample to.')
            return np.r_[0.0:self.TR*newLen:self.TR]

    def saveMatrix(self):
        """
        Returns the feature matrix to be saved by user.
            :returns:
                - feature_matrix : np.ndarray [shape=].
        """
        return self.feature_matrix
