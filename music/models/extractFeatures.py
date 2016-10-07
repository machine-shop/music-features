## script for extracting features from both sessions of music

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
fpath = '/auto/data/stimuli/' # location of sound (.wav) files
tpath = '/auto/data/archive/mri/stimreports/' # location of trfiles

### STIMULUS INFORMATION. TO BE FORMATTED LATER ###

stimuli = ['Beethoven_Op027_No1-01', 'Beethoven_WoO080', 'Beethoven_Op031No2-01',
           'Brahms_Op010No1', 'Brahms_Op010No2', 'Chopin_Op026No1', 'Chopin_Op026No2',
           'Chopin_Op066','Rachmaninoff_Op036-03_Op039-01', # session 1 stimuli
           'Bach_BWV875_0102_Beethoven_Op27No1_03','Brahms_Op005_01',
           'Chopin_Op010_03_Op028_04_Op29', 'Chopin_Op10_04_Op48No1',
           'Haydn_HobXVINo52_01', 'Ravel_JeuxDeau_Skyrabin_Op008No8' # session 2 stimuli
          ]

wavnames = ['session1/Beethoven_Op027No1-01_003_20090916-SMD-norm.wav',
            'session1/Beethoven_WoO080_001_20081107-SMD-norm.wav',
            'session1/Beethoven_Op031No2-01_002_20090916-SMD-norm.wav',
            'session1/Brahms_Op010No1_003_20090916-SMD-norm.wav',
            'session1/Brahms_Op010No2_003_20090916-SMD-norm.wav',
            'session1/Chopin_Op026No1_003_20100611-SMD-norm.wav',
            'session1/Chopin_Op026No2_005_20100611-SMD-norm.wav',
            'session1/Chopin_Op066_006_20100611-SMD-norm.wav',
            'session1/Rachmaninoff_Op036-03_Op039-01-SMD-norm.wav', # end sesion 1
            'session2/Bach_BWV875-01-02_Beethoven_Op27-No1-03-SMD-norm.wav',
            'session2/Brahms_Op005-01_002_20110315-SMD-norm.wav',
            'session2/Chopin_Op010-03_Op028-04_Op029-SMD-norm.wav',
            'session2/Chopin_Op010-04_Op048No1-SMD-norm.wav',
            'session2/Haydn_HobXVINo52-01_008_20110315-SMD-norm.wav',
            'session2/Ravel_JeuxDEau_Skyrabin_Op008No8-SMD-norm.wav' # end session 2
           ]

trfiles = ['20150608AN-Beethoven_Op027_No1-01.report',
           '20150608AN-Beethoven_WoO080-0.report',
           '20150608AN-Beethoven_Op031No2-01.report',
           '20150608AN-Brahms_Op010No1.report',
           '20150608AN-Brahms_Op010No2.report',
           '20150608AN-Chopin_Op026No1.report',
           '20150608AN-Chopin_Op026No2.report',
           '20150608AN-Chopin_Op066.report',
           '20150608AN-Rachmaninoff_Op036-03_Op039-01.report', # end session 1
           '20150618AN-Bach_BWV875_0102_Beethoven_Op27No1_03.report',
           '20150618AN-Brahms_Op005_01.report',
           '20150618AN-Chopin_Op010_03_Op028_04_Op29.report',
           '20150618AN-Chopin_Op10_04_Op48No1.report',
           '20150618AN-Haydn_HobXVINo52_01.report',
           '20150618AN-Ravel_JeuxDeau_Skyrabin_Op008No8.report' # end session 2
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
    feature_gen.saveObject(fpath=output_dir, name=stimuli[i])

print 'Saved features for all songs...'
