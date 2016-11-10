from __future__ import division
import numpy as np
from math import ceil, floor, log
# import librosa

__all__ = ['TRFile',
	  	   'prevPow',
	   	   'nextPow',
	   	   'loadProfiles',
	   	   'framewise',
	   	   'calculateStartEnd',
	   	   'calculateSegLen',
	   	   'encodeMode',
	   	   'encodeKey']

class TRFile(object):
    def __init__(self, trfilename, expectedtr=2.0045):
        """Loads data from [trfilename], should be output from stimulus
        presentation code.
        TRFile was written by FD (original source stimulus_utils_fi.py)
        """
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr

        if trfilename is not None:
            self.load_from_file(trfilename)

    def load_from_file(self, trfilename):
        """Loads TR data from report with given [trfilename].
        """
        # Read the report file and populate the datastructure
        for ll in open(trfilename):
            timestr = ll.split()[0]
            label = " ".join(ll.split()[1:])
            time = float(timestr)

            if label in ("init-trigger", "trigger"):
                self.trtimes.append(time)

            elif label == "sound-start":
                self.soundstarttime = time

            elif label == "sound-stop":
                self.soundstoptime = time

            else:
                self.otherlabels.append((time, label))

        # Fix weird TR times
        itrtimes = np.diff(self.trtimes)
        badtrtimes = np.nonzero(itrtimes > (itrtimes.mean()*1.5))[0]
        newtrs = []
        for btr in badtrtimes:
            # Insert new TR where it was missing..
            newtrtime = self.trtimes[btr]+self.expectedtr
            newtrs.append((newtrtime, btr))

        for ntr, btr in newtrs:
            self.trtimes.insert(btr+1, ntr)

    def simulate(self, ntrs):
        """Simulates [ntrs] TRs that occur at the expected TR.
        """
        self.trtimes = list(np.arange(ntrs)*self.expectedtr)

    def get_reltriggertimes(self):
        """Returns the times of all trigger events relative to the sound.
        """
        return np.array(self.trtimes)-self.soundstarttime

    @property
    def avgtr(self):
        """Returns the average TR for this run.
        """
        return np.diff(self.trtimes).mean()

def nextPow(x):
    """
    Calculate the nearest power of two that is greater than this number.
        :usage:
            >>> ans = 2 ** nextPow(flt)
    """
    return ceil(log(x, 2))

def prevPow(x):
    """
    Calculate the nearest power of two that is less than this number.
        :usage:
            >>> ans = 2 ** nextPow(flt)
    """
    return floor(log(x, 2))

def loadProfiles(name="gomez"):
    """Minor and Major profiles."""

    if name == "gomez":
        # Computed by Emilia Gomez, PhD 2006 (calculated using alpha matrices)
        major = np.array([0.904679376083, 0.0, 0.449740034662, 0.0,
                          0.550259965338, 0.354419410745, 0.0, 1.0,
                          0.0, 0.354419410745, 0.0, 0.449740034662])
        minor = np.array([0.889891696751, 0.0, 0.428700361011,
                          0.571299638989, 0.0, 0.318592057762, 0.0, 1.0,
                          0.318592057762, 0.0, 0.0, 0.428700361011])

    elif name == "gomezMIR":
        # Gomez_according to MIRtoolbox
        major = np.array([1.561306527900001, 0.839633522400000,
                          1.192863059100000, 0.758618294400000,
                          1.415716274700000, 1.030016677500000,
                          0.875592679500000, 1.536103663200000,
                          0.841472563500000, 1.214095687200000,
                          0.723516215400000, 1.133100316800000])
        minor = np.array([1.622160757800000, 0.802739838600000,
                          1.158742943100000, 1.381344982200000,
                          0.979464973500000, 1.048636970100000,
                          0.859499275500000, 1.546622578800000,
                          1.052296601400000, 1.017899085600000,
                          1.127755566000000, 0.986193468900000])

    elif name == "diatonic":
        # Traditional diatonic key profiles
        major = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1])

    elif name == "tonicTriad":
        # Tonic Triad Chords (i.e. I, III, V)
        major = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        minor = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

    elif name == "krumhansl":
        # Cognitive Foundations of Pitch, Carol L. Krumhansl
        # Chapter: Quantifying tonal hierarchies and key distances
        major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                          5.19, 2.39, 3.66, 2.29, 2.88])
        minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                          4.75, 3.98, 2.69, 3.34, 3.17])

    elif name == "temperley":
        # The Krumhansl-Schmuckler Key-Finding Algorithm Revisted, David
        # Temperley
        major = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5,
                          2.0, 3.5, 1.5, 4.0])
        minor = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5,
                          3.5, 2.0, 1.5, 4.0])

    elif name == "temperleyMIR":
        # Temperley MIREX 2005
        major = np.array([0.748, 0.060, 0.488, 0.082, 0.67, 0.46, 0.096,
                          0.715, 0.104, 0.366, 0.057, 0.4])
        minor = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.46, 0.105,
                          0.747, 0.404, 0.067, 0.133, 0.33])

    elif name == "wei":
        # Wei Chai MIT PhD thesis
        major = np.array([81302, 320, 65719, 1916, 77469, 40928, 2223, 83997,
                          1218, 39853, 1579, 28908])
        minor = np.array([39853, 1579, 28908, 81302, 320, 65719, 1916, 77469,
                          40928, 2223, 83997, 1218])
    return major, minor

def calculateStartEnd(hop_length, segment_length, iterV=None):
    '''
    Functino used to calculate the start and end indices when taking
    chunks from a long array.
        :parameters:
            - hop_length : integer. The amount of overlap between frames.
            - segment_length : integer. Frame length.
            - iterV : integer. Which iteration of a for loop or which chunck
                currently on. Default is None.
    '''
    start = hop_length * iterV
    end = segment_length + hop_length * iterV
    return start, end

def framewise(func, y, win_length, hop_length, padAmt=None, **kwargs):
    '''
    Internal helper function to be used in all feature decomposition functions.
    Computes feature according to frames.
        :parameters:
            - func : function name. Feacture extraction function to be applied
                     to each frame.
            - y : np.ndarray [shape=(n,)]. Time series to calculate the RMS of.
            - win_length : integer. Length (in samples) of each frame.
            - hop_length : integer. Overlapping samples between each frame
        :returns:
            - A numpy array with the result of function 'func' applied to each
              frame.
    '''
    assert len(y) >= win_length, 'win_length may not be less than the length of time series'
    # padding to prevent loss of information
    if padAmt is None:
        if len(y) % hop_length != 0:
            padAmt = hop_length - len(y) % hop_length
        else:
            padAmt = 0
    y = np.pad(y, (padAmt//2, padAmt - padAmt//2), mode='reflect')
    windows = np.lib.stride_tricks.as_strided(
        y,
        shape=(1+(len(y)-win_length)//hop_length, win_length),
        strides=(hop_length*y.itemsize, y.itemsize))
    vals = [func(window, **kwargs) for window in windows]
    return np.array(vals)

def encodeKey(vals):
    """
    Encode key values; switch from string to integer. Used by
    feature extraction object (features.py).
    code = {'C':0, 'C#':1, 'D':2, 'D#':3, 'E':4, 'F':5,
            'F#':6, 'G':7, 'G#':8, 'A':9, 'A#':10, 'B':11}
        :parameters:
            - vals : np.ndarray. The key values to encode.
    """
    code = {'C':0, 'C#':1, 'D':2, 'D#':3, 'E':4, 'F':5,
            'F#':6, 'G':7, 'G#':8, 'A':9, 'A#':10, 'B':11}
    newvals = np.empty(len(vals))
    for i in range(len(vals)):
        newvals[i] = code[vals[i]]
    return newvals

def encodeMode(vals):
    """
    Encode mode values; switch from string to integer. Used by
    feature extraction object (features.py).
    code = {'Major':0, 'Minor':1}
        :parameters:
            - vals : np.ndarray. The mode values to encode.
    """
    code = {'Major':0, 'Minor':1}
    newvals = np.empty(len(vals))
    for i in range(len(vals)):
        newvals[i] = code[vals[i]]
    return newvals

def calculateSegLen(sr, seconds, ds_rate, use='prev'):
    """
    Calculates the segment length, corresponding to secondary
    window length. Used with MPS and FP in features.py.
        :parameters:
            - sr : integer. The sampling rate used.
            - second : float. The number of seconds the secondary
                window is supposed to be.
            - ds_rate : float. The chunking used in the middle step.
                (i.e hop_length) (TBD if hop_length more accurate than
                n_fft)
            - use : string. Whether or not to use either prevPow or nextPow
                in the calculations. Default is 'prev'.
                Choices: 'prev', 'next', None.
    """
    tmp = sr * seconds / ds_rate
    if use == 'prev':
        tmp = 2**prevPow(tmp)
    elif use == 'next':
        tmp = 2**nextPow(tmp)
    return tmp
