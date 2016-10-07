## script for extracting other acoustic features
## will need to be modified before using with new python library

import numpy as np
import music.features.features as ft
import os
from math import ceil, log, floor

### PARAMETERS ###

sr = 44100 # sampling rate
use_prev = False # calculate the next lowest power of 2 
use_next = False # calculate the next largest power of 2

# short feature
short_nfft = 2048 # ~50 ms granularity
short_hop = int(short_nfft / 2) # ~25 ms overlap between windows

# long feature
seconds = 1.34 # maximal window size librosa stft will take
long_nfft = int(seconds * sr) # convert the seconds to samples
if use_prev:
  long_nfft = 2**ft.prevPow(long_nfft) # ~0.743 sec
elif use_next:
  long_nfft = 2**ft.nextPow(long_nfft)
long_hop = int(long_nfft / 2) # half overlap

# binning spectrogram
seconds = 4 # rounded down to nearest power of 2

# the intermediate step for FP and MPS
FP_nfft = 512 # number from Pampalks PhD Thesis (2006)
FP_hop = 512 # number from Pampalks PhD Thesis (2006); no overlap

# Specify output directories and paths to necessary data.
output_dir = '/auto/k7/lucine/projects/' # to save features.py objects
project = 'speech' # the project under which to save the extracted feature objects
fpath = '/auto/data/stimuli/speech_wavs/wavs' # location of sound (.wav) files
tpath = '/auto/data/archive/mri/stimreports/' # location of trfiles

### STIMULUS INFORMATION. TO BE FORMATTED LATER ###

stimuli = ['alternateithicatom',
            'wheretheressmoke', # validation
            'avatar',
            'legacy',
            'odetostepfather',
            'souls',
          ]

wavnames = ['alternateithicatom.wav',
            'wheretheressmoke-norm-1.wav', # validation
            'avatar-norm.wav',
            'legacy.wav',
            'odetostepfather.wav',
            'souls.wav',
           ]

trfiles = ['20150801AN_alternateithicatom.report',
           '20150801AN_wheretheressmoke-0.report',
           '20150801AN_avatar.report',
           '20150801AN_legacy.report',
           '20150801AN_odetostepfather.report',
           '20150801AN_souls.report'
          ]


### ACTUAL FEATURE EXTRACTION SCRIPT ###

for i in range(len(stimuli)):
    fname = os.path.join(fpath, wavnames[i])
    tname = os.path.join(tpath, trfiles[i])
    feature_gen = ft.Features(fname, trfile=tname, window=3, cutoff_mult=1.0, rectify=False)

    # Extract all features
    # short length features
    feature_gen.rms()
    feature_gen.temporalFlatness()
    feature_gen.zcr()
    feature_gen.spectralCentroid()
    feature_gen.spectralSpread()
    feature_gen.spectralFlatness()
    feature_gen.spectralContrast()
    feature_gen.spectralRolloff()
    feature_gen.mfcc()
    feature_gen.melspectrogram()
    feature_gen.stft()

    # long length feature
    feature_gen.chromagram(stft=True, n_fft=long_nfft, hop_length=long_hop,
                            seconds=seconds, use_librosa=True)
    feature_gen.tonalCentroid(chroma=feature_gen.returnFeature('chroma'))
    feature_gen.tonality(n_fft=long_nfft, hop_length=long_hop,
                          seconds=seconds, use_librosa=True)
    feature_gen.fluctuationPatterns(n_fft=FP_nfft, hop_length=FP_hop, seconds=seconds)
    feature_gen.fluctuationCentroid(n_fft=FP_nfft, hop_length=FP_hop, seconds=seconds)
    feature_gen.fluctuationFocus(n_fft=FP_nfft, hop_length=FP_hop, seconds=seconds)
    feature_gen.fluctuationEntropy(n_fft=FP_nfft, hop_length=FP_hop, seconds=seconds)

    print 'Saving extracted features for {0}...'.format(stimuli[i])
    feature_gen.saveObject(fpath=output_dir, name=stimuli[i], project=project)

print 'Saved features for all songs...'
